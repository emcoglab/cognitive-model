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

from typing import Set, Dict

from model.common import ActivationValue, ItemIdx, GraphPropagationComponent
from model.graph import Graph


class TemporalSpreadingActivation(GraphPropagationComponent):
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

        super(TemporalSpreadingActivation, self).__init__(
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

    def impulses_headed_for(self, n: ItemIdx) -> Dict[int, float]:
        """A time-keyed dict of cumulative activation due to arrive at a node."""
        return {
            t: activation_arriving_at_time_t[n]
            for t, activation_arriving_at_time_t in self._scheduled_activations.items()
            if n in activation_arriving_at_time_t.keys()
        }

    def _postsynaptic_guard(self, activation: ActivationValue) -> bool:
        # Activation must exceed a firing threshold to cause further propagation.
        return activation >= self.firing_threshold

    def _presynaptic_guard(self, activation: ActivationValue) -> bool:
        # If this node is currently suprathreshold, it acts as activation sink.
        # It doesn't accumulate new activation and cannot fire.
        return activation < self.firing_threshold
