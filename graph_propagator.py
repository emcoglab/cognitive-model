"""
===========================
Base class for all components.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

import json
from abc import ABC
from collections import namedtuple, defaultdict, deque
from typing import Dict, DefaultDict, Optional, List, Callable, Deque

from .basic_types import ActivationValue, ItemIdx, ItemLabel, Item, Component
from .events import ModelEvent, ItemActivatedEvent
from .graph import Graph
from .utils.exceptions import ItemNotFoundError
from .utils.maths import make_decay_function_constant


# Maps an item and its activation to a new, modulated activation
Modulation = Callable[[ItemIdx, ActivationValue], ActivationValue]
# Maps an item and its activation to whether something is permitted to happen,
# i.e. whether the guard is passed
# (True => it is allowed to happen; False => it is not allowed to happen)
Guard = Callable[[ItemIdx, ActivationValue], bool]
# Maps an elapsed time and and initial activation to a final activation
DecayFunction = Callable[[int, ActivationValue], ActivationValue]

IMPULSE_PRUNING_THRESHOLD = 0.05


class ActivationRecord(namedtuple('ActivationRecord', ['activation', 'time_activated'])):
    """
    ActivationRecord stores a historical node activation.

    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.

    Don't thoughtlessly change this class as it probably needs to remain a small namedtuple for performance reasons.
    """
    __slots__ = ()


def blank_node_activation_record() -> ActivationRecord:
    """A record for an unactivated node."""
    return ActivationRecord(activation=0, time_activated=-1)


class GraphPropagator(ABC):

    def __init__(self,
                 graph: Graph,
                 idx2label: Dict[ItemIdx, ItemLabel],
                 impulse_pruning_threshold: ActivationValue,
                 node_decay_function: Optional[DecayFunction],
                 edge_decay_function: Optional[DecayFunction],
                 component: Component,
                 shelf_life: Optional[int],
                 ):
        """
        Underlying shared code between model components which operate via propagation of activation on a graph.
        :param graph:
            `graph` should be an undirected, weighted graph with the following data:
                On Edges:
                    weight
                    length
        :param idx2label:
            A dictionary whose keys are item indexes, and whose values are item labels.
            This lets us keep the graph data stored throughout as ints rather than strings, saving a bunch of memory.
        :param impulse_pruning_threshold
            Any impulse which decays to less than this threshold before reaching its destination will be deleted.
            This is primarily an optimisation facility, to stop trickle impulses flooding the graph.
        :param node_decay_function:
            A function governing the decay of activations on nodes.
            Use the decay_function_*_with_params methods to create these.
            If None is supplied, a constant function will be used by default (i.e. no decay).
        :param edge_decay_function:
            A function governing the decay of activations in connections.
            Use the decay_function_*_with_* methods to create these.
            If None is supplied, a constant function will be used by default (i.e. no decay).
        :param shelf_life:
            For optimisation purposes only!
            If supplied and not None, any activations which would have been scheduled after this point are ignored.
            Use ONLY when you know output won't be examined after this point in time.
        """

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Don't reset
        self.idx2label: Dict[ItemIdx, ItemLabel] = idx2label
        self.label2idx: Dict[ItemLabel, ItemIdx] = {v: k for k, v in idx2label.items()}

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Thresholds
        # Use >= and < to test for above/below

        # Optimisation threshold which prevents minute impulses from ricocheting around forever.
        # This applies to IMPULSES ONLY (i.e. activations scheduled elsewhere in the future).
        # It should be applied when impulses are generated, as it will not be applied when they arrive.
        # This also means that it only applies to presynnaptic (arrival) activations, and is not affected by
        # presynnaptic modulations.
        self.impulse_pruning_threshold: ActivationValue = impulse_pruning_threshold

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        # Use a constant function by default
        self.node_decay_function: DecayFunction = (node_decay_function
                                                   if node_decay_function is not None
                                                   else make_decay_function_constant())
        self.edge_decay_function: DecayFunction = (edge_decay_function
                                                   if edge_decay_function is not None
                                                   else make_decay_function_constant())

        # Modulations and guards are applied in sequence
        # The output of one modulation is the input to the next; the output of the final is the result.  If there are
        # none, no modulation is applied.
        # If any guard in the sequence returns False, the sequence terminates with False; else we get True.

        # presynaptic_modulations:
        #     Modulates the incoming activations to items. E.g. by scaling incoming activation by some property of the
        #     item. Applies to the sum total of all converging activation, not to each individual incoming activation
        #     (this isn't the same unless the modulation is linear).
        # :param idx:
        #     The item receiving the activation.
        # :param activation:
        #     The unmodified presynaptic activation.
        # :return:
        #     The modified presynaptic activation.
        self.presynaptic_modulations: Deque[Modulation] = deque()
        # presynaptic_guards:
        #     Guards a node's accumulation (and hence also its firing) based on its activation before incoming
        #     activation has accumulated.  (E.g. making sufficiently-activated nodes into sinks until they decay.)
        #     See "Guard" below for signature.
        #     argument `activation` is the activation level of the item BEFORE accumulation.
        # :param idx:
        #     The item receiving the activation.
        # :param activation:
        #     The activation level of the item before accumulation.
        # :return:
        #     True if the node should be allowed accumulate, else False.
        self.presynaptic_guards: Deque[Guard] = deque()
        # postsynaptic_modulations:
        #     Modulates the activations of items after accumulation, but before firing.
        #     (E.g. applying an activation cap).
        # Modulates the activations of items after accumulation, but before firing.
        # (E.g. applying an activation cap).
        # :param idx:
        #     The item receiving the activation.
        # :param activation:
        #     The unmodified postsynaptic activation.
        # :return:
        #     The modified postsynaptic activation.
        self.postsynaptic_modulations: Deque[Modulation] = deque()
        # postsynaptic_guards:
        #     Guards a node's firing based on its activation after incoming activation has accumulated.
        #     (E.g. applying a firing threshold.)
        #     argument `activation` is the activation level of the item AFTER accumulation
        # Guards a node's firing based on its activation after incoming activation has accumulated.
        # (E.g. applying a firing threshold.)
        # :param idx:
        #     The item receiving the activation.
        # :param activation:
        #     The activation level of the item after accumulation.
        # :return:
        #     True if the node should be allowed to fire, else False.
        self.postsynaptic_guards: Deque[Guard] = deque()

        # endregion

        # region Resettable
        # These fields are reinitialised in .reset()

        # Zero-indexed tick counter.
        # The clock should be updated as the final step in a tick(). Thus everything that happens during a tick is
        # stamped with the time at the start of the tick(). This means that if activations are made externally to a
        # component at any time before or during a tick(), they will have the same timestamp, and will be processed
        # during the tick(). Furthermore, other events which relate to the activation of items, etc, which are produced
        # as items are activated or during the tick() will have the same timestamp.
        self.clock: int = 0

        # A node-keyed dictionaries of node ActivationRecords.
        # Stores the most recent activation of each node, if any.
        self._activation_records: DefaultDict[ItemIdx, ActivationRecord] = defaultdict(blank_node_activation_record)

        # Impulses are stored in an arrival-time-keyed dict of destination-idx-keyed dicts of cumulative activation
        # scheduled for arrival.
        # This way, when an arrival time is reached, we can .pop() a destination-idx-keyed dict of activations to
        # process.  Nice!
        # ACTUALLY we'll use a defaultdict here, so we can quickly and easily add a scheduled activation in the right
        # place without verbose checks.
        #
        # arrival-time → destination-item → activation-to-apply
        self._scheduled_activations: DefaultDict[int, DefaultDict[ItemIdx, ActivationValue]] = defaultdict(
            # In case the aren't any scheduled activations due to arrive at a particular time, we'll just find an empty
            # defaultdict
            lambda: defaultdict(
                # In case there aren't any scheduled activations due to arrive at a particular node, we'll just find
                # 0 activation, which allows for handy use of +=
                lambda: ActivationValue(0)
            ))

        self.component: Component = component

        self._shelf_life: Optional[int] = shelf_life

        # endregion

    def reset(self):
        """Resets the spreading to its initial state without having to reload any data."""
        self.clock = 0
        self._activation_records = defaultdict(blank_node_activation_record)
        self._scheduled_activations = defaultdict(lambda: defaultdict(lambda: ActivationValue(0)))

    # region tick()

    def tick(self) -> List[ModelEvent]:
        """
        Performs the spreading activation algorithm for one tick of the clock based on the current state.

        Modifications to the model, such as activating items or scheduling activations, should be applied BEFORE calling
        .tick().

        .tick() will:

            -   Apply all activations scheduled for the CURRENT time (before .tick())
            -   Increment the clock.

        When modifying .tick() in an override, instead override _evolve_model().

        EXAMPLE
        -------

        On __init__() the clock is 0.
        Then some items are activated (i.e. activations scheduled for t=0).
        .tick() is called.
        The scheduled activations are applied and events returned.
        The clock is now 1.

        :return:
            List of events.
        """

        # Do the work
        events = self._evolve_propagator()

        # Advance the clock
        self.clock += 1

        return events

    def _evolve_propagator(self) -> List[ModelEvent]:
        """
        Do the work of tick() before the clock is advanced.
        Override this instead of .tick()
        """
        events = self.__apply_activations()

        # There will be at most one event for each item which has an event
        assert len(events) == len(set(e.item.idx for e in events))

        return events

    def __apply_activations(self) -> List[ItemActivatedEvent]:
        """
        Apply scheduled activations for the current time.
        :return:
            Activation events
        """
        activation_events = []
        if self.clock in self._scheduled_activations:

            # This is an item_idx-keyed dict of activation ready to arrive
            scheduled_activations: DefaultDict[ItemIdx, ActivationValue] = self._scheduled_activations.pop(self.clock)

            # TODO optimisation: sort into numpy.array and apply presynaptic modulation in a vectorised manner
            if len(scheduled_activations) > 0:
                for destination_item, activation in scheduled_activations.items():
                    # Because self._scheduled_activations is a defaultdict, it's possible that checking for a
                    # non-existent destination at some time will produce a scheduled 0 activation at that time.
                    # This should not happen in ordinary operation, but can happen during debugging etc.
                    # These should not affect the model's behaviour, so we manually skip them here.
                    if activation == 0:
                        continue
                    activation_event = self.__apply_activation_to_item_with_idx(destination_item, activation)
                    if activation_event:
                        activation_events.append(activation_event)
        return activation_events

    def __apply_activation_to_item_with_idx(self, idx: ItemIdx, activation: ActivationValue) -> Optional[ItemActivatedEvent]:
        """
        Apply (scheduled) activation to an item.
        :param idx:
            Item to activate.
        :param activation:
            Activation to apply (presynaptic).
            May have presynaptic and postynaptic modulation applied, and activation may or may not be prevented.
        :return:
            ItemActivatedEvent if the item did activate.
            None if not.
        """

        current_activation = self.activation_of_item_with_idx(idx)

        # Check if something will prevent the activation from occurring
        if not self.__apply_presynaptic_guards(idx, current_activation):
            # If activation was blocked, node didn't activate (or fire)
            return None

        # Otherwise, we proceed with the activation:

        activation = self.__apply_presynaptic_modulation(idx, activation)

        # We don't check for resultant activation beneath self.impulse_pruning_threshold here, as it would prevent
        # manual external activation beneath the threshold. Instead we must rely on the threshold being applied when
        # *impulses* (i.e. activations scheduled elsewhere in the future) are generated.
        if activation == 0:
            return None

        # Accumulate activation
        new_activation = current_activation + activation

        # Apply postsynaptic modulations to accumulated value
        new_activation = self.__apply_postsynaptic_modulation(idx, new_activation)

        # The item activated, so an activation event occurs
        event = ItemActivatedEvent(time=self.clock, item=Item(idx=idx, component=self.component),
                                   activation=new_activation,
                                   fired=False)

        # Record the activation
        self._activation_records[idx] = ActivationRecord(new_activation, self.clock)

        # Check if the postsynaptic firing guard is passed
        if not self.__apply_postsynaptic_guards(idx, new_activation):
            # If not, stop here
            return event

        # If we did, not only did this node activate, it fired as well, so we upgrade the event
        event.fired = True
        self.__fire_node(source_idx=idx, source_activation=new_activation)

        return event

    # Separated out from __apply_activation_to_item_with_idx for profiling purposes

    def __apply_presynaptic_guards(self, idx, activation):
        for guard in self.presynaptic_guards:
            if not guard(idx, activation):
                return False
        return True

    def __apply_postsynaptic_guards(self, idx, activation):
        for guard in self.postsynaptic_guards:
            if not guard(idx, activation):
                return False
        return True

    def __apply_presynaptic_modulation(self, idx, activation):
        for modulation in self.presynaptic_modulations:
            activation = modulation(idx, activation)
        return activation

    def __apply_postsynaptic_modulation(self, idx, activation):
        for modulation in self.postsynaptic_modulations:
            activation = modulation(idx, activation)
        return activation

    def __fire_node(self, source_idx, source_activation):
        # For each incident edge
        for edge in self.graph.incident_edges(source_idx):

            # Find which node in the edge is the source node and which is the target
            n1, n2 = edge
            target_idx = n2 if source_idx == n1 else n1

            length = self.graph.edge_lengths[edge]

            arrival_activation = self.edge_decay_function(length, source_activation)

            # Skip any impulses which will be too small on arrival
            if arrival_activation < self.impulse_pruning_threshold:
                continue

            # Accumulate activation at target node at time when it's due to arrive
            self._schedule_activation_of_item_with_idx(idx=target_idx, activation=arrival_activation,
                                                       arrival_time=self.clock + length)

    # endregion

    # region Get activations

    def activation_of_item_with_idx(self, idx: ItemIdx) -> ActivationValue:
        """
        Returns the current activation of a node.
        Call this AFTER .tick() to see effect of activations applied since .tick() was last called.
        :raises ItemNotFoundError
        """
        return self.activation_of_item_with_idx_at_time(idx, time=self.clock)

    def activation_of_item_with_idx_at_time(self, idx: ItemIdx, time: int) -> ActivationValue:
        """
        Returns the current activation of a node.
        Call this AFTER .tick() to see effect of activations applied since .tick() was last called.
        :raises ItemNotFoundError
        """
        if idx not in self.graph.nodes:
            raise ItemNotFoundError(idx)
        if idx not in self._activation_records:
            return ActivationValue(0)
        activation_record: ActivationRecord = self._activation_records[idx]
        # If the last known activation is zero, or we know it's never been activated, we don't need to compute decay
        if (activation_record.activation == 0) or (activation_record.time_activated == -1):
            return ActivationValue(0)
        relative_age = time - activation_record.time_activated
        if relative_age < 0:
            raise ValueError("Can't check activation levels before node was last activated.")
        if relative_age == 0:
            # We don't need to compute decay if we're checking the activation on the tick it was activated
            return activation_record.activation
        # If we get this far, we actually have to compute decay
        return self.node_decay_function(
            relative_age,
            activation_record.activation)

    def activation_of_item_with_label(self, label: ItemLabel) -> ActivationValue:
        """
        Returns the current activation of a node.
        Call this AFTER .tick() to see effect of activations applied since .tick() was last called.
        :raises: ItemNotFoundError
        """
        try:
            return self.activation_of_item_with_idx(self.label2idx[label])
        except KeyError as e:
            raise ItemNotFoundError from e

    # endregion

    # region Set activations

    def activate_item_with_idx(self, idx: ItemIdx, activation: ActivationValue):
        """
        Activate an item.
        Call this BEFORE .tick().
        """
        self._schedule_activation_of_item_with_idx(idx, activation, self.clock)

    def activate_items_with_labels(self, labels: List[ItemLabel], activation: ActivationValue):
        """
        Activate a list of items.
        Call this BEFORE .tick().
        :raises ItemNotFoundError
        """
        try:
            idxs = [self.label2idx[label] for label in labels]
        except KeyError as e:
            raise ItemNotFoundError() from e
        for idx in idxs:
            self.activate_item_with_idx(idx, activation)

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue):
        """
        Activate an item.
        Call this BEFORE .tick().
        :raises ItemNotFoundError
        """
        try:
            idx = self.label2idx[label]
        except KeyError as e:
            raise ItemNotFoundError(label) from e
        self.activate_item_with_idx(idx, activation)

    def _schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        """
        Schedule an item to receive activation at a future time.
        Call this BEFORE .tick().
        """
        # Inequality here because the clock hasn't advanced yet for this .tick().
        # We want to be able to schedule activations for the final tick of the shelf life.
        if (self._shelf_life is not None) and (arrival_time > self._shelf_life):
            return
        self._scheduled_activations[arrival_time][idx] += activation

    def schedule_activation_of_item_with_label(self, label: ItemLabel, activation: ActivationValue, arrival_time: int):
        """
        Schedule an item to receive activation at a future time.
        Call this BEFORE .tick().
        :raises: ItemNotFoundError
        """
        try:
            idx = self.label2idx[label]
        except KeyError as e:
            raise ItemNotFoundError(label) from e
        self._schedule_activation_of_item_with_idx(idx=idx, activation=activation, arrival_time=arrival_time)

    # endregion

    def __str__(self):
        string_builder = f"CLOCK = {self.clock}\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.idx2label[node]}: {self.activation_of_item_with_idx(node)}\n"
        return string_builder

    def scheduled_activation_count(self) -> int:
        return sum([1
                    for tick, schedule_activation in self._scheduled_activations.items()
                    for idx, activation in schedule_activation.items()
                    if activation > 0])


def _load_labels(node_label_path: str) -> Dict[ItemIdx, ItemLabel]:
    with open(node_label_path, mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    node_labelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_labelling_dictionary[ItemIdx(k)] = v
    return node_labelling_dictionary
