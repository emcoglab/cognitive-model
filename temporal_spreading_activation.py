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
from typing import Set, Dict, DefaultDict, Tuple

from model.common import ActivationRecord, ItemActivatedEvent, blank_node_activation_record, ActivationValue, Label
from model.graph import Graph, Node

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class TemporalSpreadingActivation(object):

    def __init__(self,
                 graph: Graph,
                 node_relabelling_dictionary: Dict,
                 firing_threshold: ActivationValue,
                 conscious_access_threshold: ActivationValue,
                 impulse_pruning_threshold: float,
                 node_decay_function: callable,
                 edge_decay_function: callable):
        """
        :param graph:
            `graph` should be an undirected, weighted graph with the following data:
                On Edges:
                    weight
                    length
        :param node_relabelling_dictionary:
            A dictionary whose keys are nodes, and whose values are node labels.
            This lets us keep the graph data stored throughout as ints rather than strings, saving a bunch of memory.
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        :param conscious_access_threshold:
            Conscious access threshold.
            A node will be listed as activated if its activation reaches this threshold.
        :param impulse_pruning_threshold
            Any impulse which decays to less than this threshold before reaching its destination will be deleted.
        :param node_decay_function:
            A function governing the decay of activations on nodes.
            Use the decay_function_*_with_params methods to create these.
        :param edge_decay_function:
            A function governing the decay of activations in connections.
            Use the decay_function_*_with_* methods to create these.
        """

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Thresholds
        # Use >= and < to test for above/below
        self.firing_threshold: ActivationValue = firing_threshold
        self.conscious_access_threshold: ActivationValue = conscious_access_threshold
        self.impulse_pruning_threshold: float = impulse_pruning_threshold

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        self.node_decay_function: callable = node_decay_function
        self.edge_decay_function: callable = edge_decay_function

        # Node label dictionaries
        # node ↦ label
        self.node2label: Dict = node_relabelling_dictionary
        # label ↦ node
        self.label2node: Dict = dict((v, k) for k, v in node_relabelling_dictionary.items())

        # Zero-indexed tick counter.
        self.clock: int = int(0)

        # Graph data:

        # A node-keyed dictionaries of node activations.
        # Stores the most recent activation of each node, if any.
        self._node_activation_records: DefaultDict = defaultdict(blank_node_activation_record)

        # Impulses are stored in an arrival-time-keyed dict of destination-node-keyed dicts of cumulative activation
        # scheduled for arrival.
        # This way, when an arrival time is reached, we can .pop() a destination-node-keyed dict of impulses to process.
        # Nice!
        # ACTUALLY we'll use a defaultdict here, so we can quickly and easily add an impulse in the right place without
        # verbose checks.
        self._impulses: DefaultDict[int, DefaultDict[Node, float]] = defaultdict(
            # In case the aren't any impulses due to arrive at a particular time, we'll just find an empty dict
            lambda: defaultdict(
                # In case there aren't any impulses due to arrive at a particular node, we'll just find 0 activation,
                # which allows for handy use of +=
                float
            ))

    def schedule_activation(self, at_node: Node, activation: ActivationValue, arrival_time: int):
        self._impulses[arrival_time][at_node] += activation

    def n_suprathreshold_nodes(self) -> int:
        """
        The number of nodes which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return len([
            n
            for n in self.graph.nodes
            if self.activation_of_node(n) >= self.firing_threshold
        ])

    def impulses_headed_for(self, n: Node) -> Dict[int, float]:
        """A time-keyed dict of cumulative activation due to arrive at a node."""
        return {
            t: activation_arriving_at_time_t[n]
            for t, activation_arriving_at_time_t in self._impulses.items()
            if n in activation_arriving_at_time_t.keys()
        }

    def activation_of_node(self, n: Node) -> float:
        """Returns the current activation of a node."""
        assert n in self.graph.nodes

        activation_record: ActivationRecord = self._node_activation_records[n]
        return self.node_decay_function(
            self.clock - activation_record.time_activated,  # node age
            activation_record.activation)

    def activation_of_node_with_label(self, n: Label) -> float:
        """Returns the current activation of a node."""
        return self.activation_of_node(self.label2node[n])

    def activate_node_with_label(self, label: Label, activation: float) -> Tuple[bool, bool]:
        return self.activate_node(self.label2node[label], activation)

    def activate_node(self, n: Node, activation: float) -> Tuple[bool, bool]:
        """
        Activate a node.
        :param n:
        :param activation:
        :return:
            A 2-tuple of bools: (
                Node did fire,
                Node's activation did cross conscious-access threshold
            )
        """
        assert n in self.graph.nodes

        current_activation = self.activation_of_node(n)

        currently_below_conscious_access_threshold = current_activation < self.conscious_access_threshold

        # If this node is currently suprathreshold, it acts as a sink.
        # It doesn't accumulate new activation and cannot fire.
        if current_activation >= self.firing_threshold:
            return (
                # Node didn't fire
                False,
                # Node's activation didn't change so it definitely didn't cross the conscious-access threshold
                False
            )

        # Otherwise, we proceed with the activation:

        # Accumulate activation
        new_activation = current_activation + activation
        self._node_activation_records[n] = ActivationRecord(new_activation, self.clock)

        # Check if we reached the conscious-access threshold
        did_cross_conscious_access_threshold = currently_below_conscious_access_threshold and (new_activation > self.conscious_access_threshold)

        # Check if we reached the firing threshold.

        if new_activation < self.firing_threshold:
            # If not, we're done
            return (
                # Node didn't fire
                False,
                did_cross_conscious_access_threshold
            )

        else:
            # If so, Fire!

            # Fire and rebroadcast
            source_node = n

            # For each incident edge
            for edge in self.graph.incident_edges(source_node):

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

                arrival_time = int(self.clock + length)

                # Accumulate activation at target node at time when it's due to arrive
                self.schedule_activation(target_node, arrival_activation, arrival_time)

            return (
                # Node did fire
                True,
                did_cross_conscious_access_threshold and (new_activation > self.conscious_access_threshold)
            )

    def _propagate_impulses(self) -> Set:
        """
        Propagates impulses along connections.
        :return:
            Set of nodes which became consciously active.
        """

        # "Propagation" happens by just incrementing the global clock.
        # But we have to check if any impulses have reached their destination.

        nodes_which_fired = set()
        nodes_which_crossed_conscious_access_threshold = set()

        if self.clock in self._impulses:

            # This should be a destination-node-keyed dict of activation ready to arrive
            activation_at_destination: DefaultDict = self._impulses.pop(self.clock)

            if len(activation_at_destination) > 0:
                # Each such impulse activates its target node
                for destination_node, activation in activation_at_destination.items():
                    node_did_fire, node_did_cross_conscious_access_threshold = self.activate_node(destination_node, activation)
                    if node_did_fire:
                        nodes_which_fired.add(destination_node)
                    if node_did_cross_conscious_access_threshold:
                        nodes_which_crossed_conscious_access_threshold.add(destination_node)

        return nodes_which_crossed_conscious_access_threshold

    def tick(self) -> Set[ItemActivatedEvent]:
        """
        Performs the spreading activation algorithm for one tick of the clock.
        :return:
            Set of nodes which became active.
        """
        self.clock += 1

        nodes_which_became_consciously_active = self._propagate_impulses()

        return set(ItemActivatedEvent(self.node2label[node], self.activation_of_node(node), self.clock) for node in nodes_which_became_consciously_active)

    def __str__(self):

        string_builder = f"CLOCK = {self.clock}\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._node_activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.node2label[node]}: {self.activation_of_node(node)}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]
