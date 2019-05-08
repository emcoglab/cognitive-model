"""
===========================
Temporal spreading activation.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import logging
from collections import defaultdict
from os import path
from typing import Set, Dict, DefaultDict

from numpy import Infinity

from model.component import ActivationValue, ItemIdx, ItemLabel, ActivationRecord, ItemActivatedEvent, _load_labels, \
    blank_node_activation_record
from model.graph import Graph
from model.utils.maths import make_decay_function_constant
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class TemporalSpreadingActivation:

    def __init__(self,
                 graph: Graph,
                 item_labelling_dictionary: Dict,
                 firing_threshold: ActivationValue,
                 impulse_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue = Infinity,
                 node_decay_function: callable = None,
                 edge_decay_function: callable = None):
        """
        :param graph:
            `graph` should be an undirected, weighted graph with the following data:
                On Edges:
                    weight
                    length
        :param item_labelling_dictionary:
            A dictionary whose keys are nodes, and whose values are node labels.
            This lets us keep the graph data stored throughout as ints rather than strings, saving a bunch of memory.
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        :param impulse_pruning_threshold
            Any impulse which decays to less than this threshold before reaching its destination will be deleted.
        :param activation_cap:
            Any node which would be activated above this gets its activation clamped at this level.
            Default: Infinity.
        :param node_decay_function:
            A function governing the decay of activations on nodes.
            Use the decay_function_*_with_params methods to create these.
            If None is supplied, a constant function will be used by default (i.e. no decay).
        :param edge_decay_function:
            A function governing the decay of activations in connections.
            Use the decay_function_*_with_* methods to create these.
            If None is supplied, a constant function will be used by default (i.e. no decay).
        """

        self.idx2label = item_labelling_dictionary
        self.label2idx = {v: k for k, v in item_labelling_dictionary.items()}

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

        # Zero-indexed tick counter.
        self.clock: int = int(0)

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Thresholds
        # Use >= and < to test for above/below
        self.firing_threshold: ActivationValue = firing_threshold
        self.impulse_pruning_threshold: ActivationValue = impulse_pruning_threshold

        # Optional activation cap.  Any time a node would become activated more than this, it gets this activation.
        self.activation_cap: ActivationValue = activation_cap

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        # Use a constant function by default
        self.node_decay_function: callable = node_decay_function if node_decay_function is not None else make_decay_function_constant()
        self.edge_decay_function: callable = edge_decay_function if edge_decay_function is not None else make_decay_function_constant()

        # Validations
        if self.activation_cap < self.firing_threshold:
            raise ValueError(f"activation cap {self.activation_cap} cannot be less than the firing threshold {self.firing_threshold}")

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

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        self._scheduled_activations[arrival_time][idx] += activation

    def activation_of_item_with_label(self, label: ItemLabel) -> ActivationValue:
        """Returns the current activation of a node."""
        return self.activation_of_item_with_idx(self.label2idx[label])

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue) -> bool:
        """
        Activate a node.
        :param label:
        :param activation:
        :return:
            True if the node fired, else false.
        """
        return self.activate_item_with_idx(self.label2idx[label], activation)

    def suprathreshold_items(self) -> Set[ItemIdx]:
        """
        Items which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return set(
            n
            for n in self.graph.nodes
            if self.activation_of_item_with_idx(n) >= self.firing_threshold
        )

    def impulses_headed_for(self, n: ItemIdx) -> Dict[int, float]:
        """A time-keyed dict of cumulative activation due to arrive at a node."""
        return {
            t: activation_arriving_at_time_t[n]
            for t, activation_arriving_at_time_t in self._scheduled_activations.items()
            if n in activation_arriving_at_time_t.keys()
        }

    def activation_of_item_with_idx(self, n: ItemIdx) -> ActivationValue:
        """Returns the current activation of a node."""
        assert n in self.graph.nodes

        activation_record: ActivationRecord = self._activation_records[n]
        return self.node_decay_function(
            self.clock - activation_record.time_activated,  # node age
            activation_record.activation)

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

        # If this node is currently suprathreshold, it acts as a sink.
        # It doesn't accumulate new activation and cannot fire.
        if current_activation >= self.firing_threshold:
            # Node didn't fire
            return False

        # Otherwise, we proceed with the activation:

        # Accumulate activation
        new_activation = current_activation + activation

        # If using an activation cap, apply this here.
        # The activation cap, if used, MUST be greater than the firing threshold, so applying the cap does not effect
        # whether the node will fire or not.
        if new_activation > self.activation_cap:
            new_activation = self.activation_cap

        self._activation_records[n] = ActivationRecord(new_activation, self.clock)

        # Check if we reached the firing threshold.
        if new_activation < self.firing_threshold:
            # If not, we're done
            # Node didn't fire
            return False

        else:
            # If so, fire and rebroadcast!
            source_node = n

            # For each incident edge
            for edge in self.graph.incident_edges(source_node):

                # Find which node in the edge is the source node and which is the target
                n1, n2 = edge
                if source_node == n1:
                    target_node = n2
                elif source_node == n2:
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

    def __str__(self):

        string_builder = f"CLOCK = {self.clock}\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.idx2label[node]}: {self.activation_of_item_with_idx(node)}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]


def load_labels_from_corpus(corpus, n_words):
    return _load_labels(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))


def load_labels_from_sensorimotor():
    return _load_labels(path.join(Preferences.graphs_dir, "sensorimotor words.nodelabels"))


