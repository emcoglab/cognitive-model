from typing import Dict, Set

from model.basic_types import ActivationValue, ItemIdx
from model.graph import Graph
from model.graph_propagator import GraphPropagator


class LinguisticPropagator(GraphPropagator):
    """
    Spreading activation on a graph over time.
    Nodes have a firing threshold and an activation cap.
    """

    def __init__(self,
                 graph: Graph,
                 idx2label: Dict,
                 firing_threshold: ActivationValue,
                 impulse_pruning_threshold: ActivationValue,
                 node_decay_function: callable = None,
                 edge_decay_function: callable = None):
        """
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        """

        super(LinguisticPropagator, self).__init__(
            graph=graph,
            idx2label=idx2label,
            impulse_pruning_threshold=impulse_pruning_threshold,
            node_decay_function=node_decay_function,
            edge_decay_function=edge_decay_function,
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Thresholds
        # Use >= and < to test for above/below
        self.firing_threshold: ActivationValue = firing_threshold

        # endregion

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

    def _postsynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Activation must exceed a firing threshold to cause further propagation.
        return activation >= self.firing_threshold

    def _presynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # If this node is currently suprathreshold, it acts as activation sink.
        # It doesn't accumulate new activation and cannot fire.
        return activation < self.firing_threshold
