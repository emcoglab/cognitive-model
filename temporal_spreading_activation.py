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
from typing import Set, Dict, Tuple

from model.component import ModelComponent, ActivationValue, ActivationRecord, \
    ItemActivatedEvent
from model.graph import Graph, Node

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class TemporalSpreadingActivation(ModelComponent):

    def __init__(self,
                 graph: Graph,
                 item_labelling_dictionary: Dict,
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
        :param item_labelling_dictionary:
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

        super().__init__(item_labelling_dictionary=item_labelling_dictionary)

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

    def n_suprathreshold_nodes(self) -> int:
        """
        The number of nodes which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return len([
            n
            for n in self.graph.nodes
            if self.activation_of_item_with_idx(n) >= self.firing_threshold
        ])

    def impulses_headed_for(self, n: Node) -> Dict[int, float]:
        """A time-keyed dict of cumulative activation due to arrive at a node."""
        return {
            t: activation_arriving_at_time_t[n]
            for t, activation_arriving_at_time_t in self._scheduled_activations.items()
            if n in activation_arriving_at_time_t.keys()
        }

    def activation_of_item_with_idx(self, n: Node) -> ActivationValue:
        """Returns the current activation of a node."""
        assert n in self.graph.nodes

        activation_record: ActivationRecord = self._activation_records[n]
        return self.node_decay_function(
            self.clock - activation_record.time_activated,  # node age
            activation_record.activation)

    def activate_item_with_idx(self, n: Node, activation: ActivationValue) -> Tuple[bool, bool]:
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

        current_activation = self.activation_of_item_with_idx(n)

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
        self._activation_records[n] = ActivationRecord(new_activation, self.clock)

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
                self.schedule_activation_of_item_with_idx(target_node, arrival_activation, arrival_time)

            return (
                # Node did fire
                True,
                did_cross_conscious_access_threshold and (new_activation > self.conscious_access_threshold)
            )

    def tick(self) -> Set[ItemActivatedEvent]:
        """
        Performs the spreading activation algorithm for one tick of the clock.
        :return:
            Set of nodes which became active.
        """
        self.clock += 1

        nodes_which_became_consciously_active = self._apply_activations()

        return set(ItemActivatedEvent(self.idx2label[node], self.activation_of_item_with_idx(node), self.clock) for node in nodes_which_became_consciously_active)

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
