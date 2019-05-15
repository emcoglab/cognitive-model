"""
===========================
Common aspects of model components.
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
from typing import Dict, Set, DefaultDict

import yaml

from model.graph import Node, Graph
from model.utils.maths import make_decay_function_constant

ActivationValue = float
ItemIdx = Node
ItemLabel = str


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


class ItemActivatedEvent:
    """
    A node activation event.
    Used to pass out of TSA.tick().
    """

    def __init__(self, label: str, activation: ActivationValue, time_activated: int):
        self.label = label
        # Use an ActivationRecord to store this so we don't have repeated code
        self._record = ActivationRecord(activation=activation, time_activated=time_activated)

    @property
    def activation(self) -> ActivationValue:
        return self._record.activation

    @property
    def time_activated(self) -> int:
        return self._record.time_activated

    def __repr__(self) -> str:
        return f"<'{self.label}' ({self.activation}) @ {self.time_activated}>"


class GraphPropagationComponent(metaclass=ABCMeta):

    def __init__(self,
                 graph: Graph,
                 idx2label: Dict[ItemIdx, ItemLabel],
                 impulse_pruning_threshold: ActivationValue,
                 node_decay_function: callable = None,
                 edge_decay_function: callable = None,
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
        self.idx2label = idx2label
        self.label2idx = {v: k for k, v in idx2label.items()}

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Thresholds
        # Use >= and < to test for above/below
        self.impulse_pruning_threshold: ActivationValue = impulse_pruning_threshold

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        # Use a constant function by default
        self.node_decay_function: callable = node_decay_function if node_decay_function is not None else make_decay_function_constant()
        self.edge_decay_function: callable = edge_decay_function if edge_decay_function is not None else make_decay_function_constant()

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
        """Resets the spreading to its initial state without having to reload any data"""
        self.clock = 0
        self._activation_records: DefaultDict[ItemIdx, ActivationRecord] = defaultdict(blank_node_activation_record)
        self._scheduled_activations: DefaultDict[int, DefaultDict[ItemIdx, ActivationValue]] = defaultdict(lambda: defaultdict(ActivationValue))

    def tick(self) -> Set[ItemActivatedEvent]:
        """
        Performs the spreading activation algorithm for one tick of the clock.
        :return:
            Set of nodes which became activated.
        """
        self.clock += 1

        nodes_which_became_active = self._apply_activations()

        return set(
            ItemActivatedEvent(label=self.idx2label[node],
                               activation=self.activation_of_item_with_idx(node),
                               time_activated=self.clock)
            for node in nodes_which_became_active)

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue) -> bool:
        """
        Activate a node.
        :param label:
        :param activation:
        :return:
            True if the item was activated, else false.
        """
        return self.activate_item_with_idx(self.label2idx[label], activation)

    def _apply_activations(self) -> Set:
        """
        Applies scheduled all scheduled activations.
        :return:
            Set of nodes which became activated.
        """

        items_which_became_activated = set()

        if self.clock in self._scheduled_activations:

            # This should be a item-keyed dict of activation ready to arrive
            scheduled_activation: DefaultDict = self._scheduled_activations.pop(self.clock)

            if len(scheduled_activation) > 0:
                for destination_item, activation in scheduled_activation.items():
                    item_did_become_activated = self.activate_item_with_idx(destination_item, activation)
                    if item_did_become_activated:
                        items_which_became_activated.add(destination_item)

        return items_which_became_activated

    def activation_of_item_with_idx(self, n: ItemIdx) -> ActivationValue:
        """Returns the current activation of a node."""
        assert n in self.graph.nodes

        activation_record: ActivationRecord = self._activation_records[n]
        return self.node_decay_function(
            self.clock - activation_record.time_activated,  # node age
            activation_record.activation)

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        """Schedule an item to receive activation at a future time."""
        self._scheduled_activations[arrival_time][idx] += activation

    def activation_of_item_with_label(self, label: ItemLabel) -> ActivationValue:
        """Returns the current activation of a node."""
        return self.activation_of_item_with_idx(self.label2idx[label])

    def activate_item_with_idx(self, n: ItemIdx, activation: ActivationValue) -> bool:
        """
        Activate a node.
        :param n:
        :param activation:
        :return:
            True if the node fired, else false.
        """
        assert n in self.graph.nodes

        current_activation = self.activation_of_item_with_idx(n)

        if not self._presynaptic_guard(n, current_activation):
            # Node didn't fire
            return False

        # Otherwise, we proceed with the activation:

        # Apply presynaptic function
        activation = self._presynaptic_modulation(n, activation)

        # Accumulate activation
        new_activation = current_activation + activation

        # Apply postsynaptic function
        new_activation = self._postsynaptic_modulation(n, new_activation)

        self._activation_records[n] = ActivationRecord(new_activation, self.clock)

        # Check if we reached the firing threshold.
        if self._postsynaptic_guard(n, new_activation):

            # If so, fire and rebroadcast!
            rebroadcast_source_node = n

            # For each incident edge
            for edge in self.graph.incident_edges(rebroadcast_source_node):

                # Find which node in the edge is the source node and which is the target
                n1, n2 = edge
                if rebroadcast_source_node == n1:
                    target_node = n2
                elif rebroadcast_source_node == n2:
                    target_node = n1
                else:
                    raise ValueError()

                length = self.graph.edge_lengths[edge]

                arrival_activation = self.edge_decay_function(length, new_activation)

                # Skip any impulses which will be too small on arrival
                if arrival_activation < self.impulse_pruning_threshold:
                    continue

                # Accumulate activation at target node at time when it's due to arrive
                self.schedule_activation_of_item_with_idx(target_node,
                                                          arrival_activation,
                                                          arrival_time=self.clock + length)

            # Node did fire
            return True

        else:
            # If not, we're done
            # Node didn't fire
            return False

    def _presynaptic_modulation(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """
        Modulates the incoming activations to items.
        (E.g. scaling incoming activation by some property of the item).
        :param item:
            The item receiving the activation.
        :param activation:
            The unmodified presynaptic activation.
        :return:
            The modified presynaptic activation.
        """
        # Default implementation: no modification
        return activation

    def _postsynaptic_modulation(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """
        Modulates the activations of items after accumulation, but before firing.
        (E.g. applying an activation cap).
        :param item:
            The item receiving the activation.
        :param activation:
            The unmodified presynaptic activation.
        :return:
            The modified presynaptic activation.
        """
        # Default implementation: no modification
        return activation

    def _presynaptic_guard(self, item: ItemIdx, activation: ActivationValue) -> bool:
        """
        Guards a node's accumulation (and firing) based on its activation before incoming activation has accumulated.
        (E.g. making sufficiently-activated nodes into sinks until they decay.)
        :param item:
            The item receiving the activation.
        :param activation:
            The activation level of the item before accumulation.
        :return:
            True if the node should be allowed fire, else False.
        """
        # Default implementation: never prevent firing.
        return True

    def _postsynaptic_guard(self, item: ItemIdx, activation: ActivationValue) -> bool:
        """
        Guards a node's firing based on its activation after incoming activation has accumulated.
        (E.g. applying a firing threshold.)
        :param item:
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


def load_model_spec(response_dir) -> dict:
    with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
        return yaml.load(spec_file, yaml.SafeLoader)


def _load_labels(nodelabel_path: str) -> Dict[ItemIdx, ItemLabel]:
    with open(nodelabel_path, mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    # TODO: this isn't a great way to do this
    node_labelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_labelling_dictionary[ItemIdx(k)] = v
    return node_labelling_dictionary
