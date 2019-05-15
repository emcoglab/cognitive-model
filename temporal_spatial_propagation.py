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

from model.common import GraphPropagationComponent
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
                 node_decay_function: callable):
        """
        :param underlying_graph:
            The graph of distances between items, where there is an edge iff the distance is within the max sphere
            radius.
        """

        super(TemporalSpatialPropagation, self).__init__(
            graph=underlying_graph,
            idx2label=idx2label,
            node_decay_function=node_decay_function,
            # Once pruning has been done, we don't need to decay activation in edges, as target items should receive the
            # full activations of their source items at the time they were last activated.
            # The maximal sphere radius is achieved by the initial graph pruning.
            edge_decay_function=None,
            # Impulses reach their destination iff their destination is within the max sphere radius.
            # The max sphere radius is baked into the underlying graph.
            impulse_pruning_threshold=0,
        )
