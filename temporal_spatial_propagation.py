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

from model.common import ActivationValue, GraphPropagationComponent
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
                 node_decay_function: callable):

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
