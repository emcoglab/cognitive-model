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
from abc import ABCMeta
from collections import namedtuple, defaultdict
from os import path
from typing import Dict, DefaultDict, Optional, List, Callable

import yaml

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.events import ModelEvent, ItemActivatedEvent
from model.graph import Graph
from model.utils.maths import make_decay_function_constant


class ActivationRecord(namedtuple('ActivationRecord', ['activation',
                                                       'time_activated'])):
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


class GraphPropagation(metaclass=ABCMeta):

    def __init__(self,
                 graph: Graph,
                 idx2label: Dict[ItemIdx, ItemLabel],
                 impulse_pruning_threshold: ActivationValue,
                 node_decay_function: Callable[[int, ActivationValue], ActivationValue] = None,
                 edge_decay_function: Callable[[int, ActivationValue], ActivationValue] = None,
                 ):
        """
        Underlying shared code between model components which operate via spreading activation on a graph.
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
        self.impulse_pruning_threshold: ActivationValue = impulse_pruning_threshold

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        # Use a constant function by default
        self.node_decay_function: Callable[[int, ActivationValue], ActivationValue] = node_decay_function if node_decay_function is not None else make_decay_function_constant()
        self.edge_decay_function: Callable[[int, ActivationValue], ActivationValue] = edge_decay_function if edge_decay_function is not None else make_decay_function_constant()

        # endregion

        # region Resettable
        # These fields are reinitialised in .reset()

        # Zero-indexed tick counter.
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
                ActivationValue
            ))

        # endregion

    def reset(self):
        """Resets the spreading to its initial state without having to reload any data."""
        self.clock = 0
        self._activation_records = defaultdict(blank_node_activation_record)
        self._scheduled_activations = defaultdict(lambda: defaultdict(ActivationValue))

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
        events = self._evolve_model()

        # Advance the clock
        self.clock += 1

        return events

    def _evolve_model(self) -> List[ModelEvent]:
        """
        Do the work of tick() before the clock is advanced.
        Override this intead of .tick()
        """
        events = self.__apply_activations()
        return events

    def __apply_activations(self):
        """Apply activations for the current time."""
        activation_events = []
        if self.clock in self._scheduled_activations:

            # This is an item-keyed dict of activation ready to arrive
            scheduled_activations: DefaultDict = self._scheduled_activations.pop(self.clock)

            if len(scheduled_activations) > 0:
                for destination_item, activation in scheduled_activations.items():
                    # Because self._scheduled_activations is a defaultdict, it's possible that checking for a
                    # non-existent destination at some time will produce a scheduled 0 activation at that time.
                    # This should not happen in ordinary operation, but can happen during debugging etc.
                    # These should not affect the model's behaviour, so we manually skip them here.
                    if activation < self.impulse_pruning_threshold:
                        continue
                    activation_event = self.__apply_activation_to_item_with_idx(destination_item, activation)
                    if activation_event:
                        activation_events.append(activation_event)
        return activation_events

    def __apply_activation_to_item_with_idx(self, idx: ItemIdx, activation: ActivationValue) -> Optional[ItemActivatedEvent]:
        """
        Apply activation to an item.
        :param idx:
            Item to activate.
        :param activation:
            Activation to apply (presynaptic).
            May have presynaptic and postynaptic modulation applied, and activation may or may not be prevented.
        :return:
            ItemActivatedEvent if the item did activate.
            None if not.
        """
        assert idx in self.graph.nodes

        current_activation = self.activation_of_item_with_idx(idx)

        # Check if something will prevent the activation from occurring
        if not self._presynaptic_guard(idx, current_activation):
            # If activation was blocked, node didn't activate or fire
            return None

        # Otherwise, we proceed with the activation:

        # Apply presynaptic modulation
        activation = self._presynaptic_modulation(idx, activation)

        # Accumulate activation
        new_activation = current_activation + activation

        # Apply postsynaptic modulation to accumulated value
        new_activation = self._postsynaptic_modulation(idx, new_activation)

        # The item activated, so an activation event occurs
        event = ItemActivatedEvent(time=self.clock, item=idx, activation=new_activation, fired=False)

        # Check if the postsynaptic firing guard is passed
        if self._postsynaptic_guard(idx, new_activation):

            # If we did, not only did this node activated, it fired as well, so we upgrade the event
            event.fired = True

            # Fire and rebroadcast!
            source_idx = idx

            # For each incident edge
            for edge in self.graph.incident_edges(source_idx):

                # Find which node in the edge is the source node and which is the target
                n1, n2 = edge
                if source_idx == n1:
                    target_idx = n2
                elif source_idx == n2:
                    target_idx = n1
                else:
                    raise ValueError()

                length = self.graph.edge_lengths[edge]

                arrival_activation = self.edge_decay_function(length, new_activation)

                # Skip any impulses which will be too small on arrival
                if arrival_activation < self.impulse_pruning_threshold:
                    continue

                # Accumulate activation at target node at time when it's due to arrive
                self.schedule_activation_of_item_with_idx(idx=target_idx,
                                                          activation=arrival_activation,
                                                          arrival_time=self.clock + length)

        # Record the activation
        self._activation_records[idx] = ActivationRecord(new_activation, self.clock)

        return event

    # endregion

    # region Get activations

    def activation_of_item_with_idx(self, idx: ItemIdx) -> ActivationValue:
        """
        Returns the current activation of a node.
        Call this AFTER .tick() to see effect of activations applied since .tick() was last called.
        """
        assert idx in self.graph.nodes

        activation_record: ActivationRecord = self._activation_records[idx]
        # If the last known activation is zero, we don't need to compute decay
        if activation_record.activation == 0:
            return ActivationValue(0)
        else:
            return self.node_decay_function(
                # node age
                self.clock - activation_record.time_activated,
                activation_record.activation)

    def activation_of_item_with_label(self, label: ItemLabel) -> ActivationValue:
        """
        Returns the current activation of a node.
        Call this AFTER .tick() to see effect of activations applied since .tick() was last called.
        """
        return self.activation_of_item_with_idx(self.label2idx[label])

    # endregion

    # region Set activations

    def activate_items_with_idxs(self, idxs: List[ItemIdx], activation: ActivationValue):
        """
        Activate a list of items.
        Call this BEFORE .tick().
        """
        for idx in idxs:
            self.activate_item_with_idx(idx, activation)

    def activate_item_with_idx(self, idx: ItemIdx, activation: ActivationValue):
        self.schedule_activation_of_item_with_idx(idx, activation, self.clock)

    def activate_items_with_labels(self, labels: List[ItemLabel], activation: ActivationValue):
        """
        Activate a list of items.
        Call this BEFORE .tick().
        """
        self.activate_items_with_idxs([self.label2idx[label] for label in labels], activation)

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue):
        """
        Activate an item.
        Call this BEFORE .tick().
        """
        self.activate_item_with_idx(self.label2idx[label], activation)

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        """
        Schedule an item to receive activation at a future time.
        Call this BEFORE .tick().
        """
        self._scheduled_activations[arrival_time][idx] += activation

    # endregion

    def _presynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """
        Modulates the incoming activations to items.
        (E.g. scaling incoming activation by some property of the item).
        :param idx:
            The item receiving the activation.
        :param activation:
            The unmodified presynaptic activation.
        :return:
            The modified presynaptic activation.
        """
        # Default implementation: no modification
        return activation

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """
        Modulates the activations of items after accumulation, but before firing.
        (E.g. applying an activation cap).
        :param idx:
            The item receiving the activation.
        :param activation:
            The unmodified presynaptic activation.
        :return:
            The modified presynaptic activation.
        """
        # Default implementation: no modification
        return activation

    def _presynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        """
        Guards a node's accumulation (and hence also its firing) based on its activation before incoming activation has
        accumulated.  (E.g. making sufficiently-activated nodes into sinks until they decay.)
        :param idx:
            The item receiving the activation.
        :param activation:
            The activation level of the item before accumulation.
        :return:
            True if the node should be allowed fire, else False.
        """
        # Default implementation: never prevent firing.
        return True

    def _postsynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        """
        Guards a node's firing based on its activation after incoming activation has accumulated.
        (E.g. applying a firing threshold.)
        :param idx:
            The item receiving the activation.
        :param activation:
            The activation level of the item after accumulation.
        :return:
            True if the node should be allowed to fire, else False.
        """
        # Default implementation: never prevent firing.
        return True

    def __str__(self):
        string_builder = f"CLOCK = {self.clock}\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.idx2label[node]}: {self.activation_of_item_with_idx(node)}\n"
        return string_builder

    # TODO: It's weird that the spec dict is built outside the class.
    # TODO: There must be a more sensible way to do this.
    @classmethod
    def save_model_spec(cls, spec, response_dir):
        """
        Save the model spec to the `response_dir`.
        """
        with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
            yaml.dump(spec, spec_file, yaml.SafeDumper)

    @classmethod
    def load_model_spec(cls, response_dir) -> dict:
        with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
            return yaml.load(spec_file, yaml.SafeLoader)


def _load_labels(nodelabel_path: str) -> Dict[ItemIdx, ItemLabel]:
    with open(nodelabel_path, mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    node_labelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_labelling_dictionary[ItemIdx(k)] = v
    return node_labelling_dictionary
