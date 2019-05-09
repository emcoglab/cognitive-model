"""
===========================
Continuous propagation of activation in a vector space.
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

from typing import Dict

from model.common import ActivationValue, GraphPropagationComponent, ItemIdx
from model.graph import Graph


class TemporalSpatialPropagation(GraphPropagationComponent):
    """
    Propagate activation by expanding spheres through space, where spheres have a maximum radius.
    Implemented by using the underlying graph of connections between points which are mutually within the maximum sphere
    radius.
    """

    def __init__(self,
                 underlying_graph: Graph,
                 idx2label: Dict,
                 impulse_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 node_decay_function: callable):
        """
        :param activation_cap:
            If None is supplied, no cap is used.
        """

        super(TemporalSpatialPropagation, self).__init__(
            graph=underlying_graph,
            idx2label=idx2label,
            node_decay_function=node_decay_function,
            # Once pruning has been done, we don't need to decay activation in edges, as target items should receive the
            # full activations of their source items at the time they were last activated.
            # The maximal sphere radius is achieved by the initial graph pruning.
            edge_decay_function=None,
            impulse_pruning_threshold=impulse_pruning_threshold,
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Cap on a node's total activation after receiving incoming.
        self.activation_cap = activation_cap

        # endregion
    def _postsynaptic_modulation(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:

        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__,
        # so applying the cap does not effect whether the node will fire or not.
        return activation if activation <= self.activation_cap else self.activation_cap
