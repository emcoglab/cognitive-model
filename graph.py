"""
===========================
For working with graphs.
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

from networkx import Graph, from_numpy_matrix, selfloop_edges, write_edgelist, read_edgelist
from numpy import ones_like
from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil


class EdgeDataKey(object):
    """Column names for edge data."""
    WEIGHT = "weight"
    LENGTH = "length"


def graph_from_distance_matrix(distance_matrix: ndarray,
                               length_granularity: int,
                               weighted_graph: bool,
                               prune_connections_longer_than: int = None) -> Graph:
    """
    Produces a Graph of the correct format to underlie a TemporalSpreadingActivation.

    Nodes will be numbered according to the row/column indices of weight_matrix (and so can
    be relabelled accordingly).

    Distances will be converted to weights using x ↦ 1-x.

    Distances will be converted to integer lengths using the supplied scaling factor.

    :param distance_matrix:
    A symmetric distance matrix in numpy format.
    :param length_granularity:
    Distances will be scaled into integer connection lengths using this granularity scaling factor.
    :param weighted_graph:
    Whether to use weights on the edges.
    If True, distances will be converted to weights using x ↦ 1-x.
        (This means it's only suitable for things like cosine and correlation distances, not Euclidean.)
    If False, all edges get the same weight.
    :param prune_connections_longer_than:
    (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be severed.
    :return:
    A Graph of the correct format.
    """

    length_matrix = ceil(distance_matrix * length_granularity)

    if weighted_graph:
        weight_matrix = ones_like(distance_matrix) - distance_matrix
    else:
        weight_matrix = ones_like(distance_matrix)

    graph = from_numpy_matrix(weight_matrix)

    # Converting from a distance matrix creates self-loop edges, which we have to remove
    graph.remove_edges_from(selfloop_edges(graph))

    # Add lengths to graph data
    for n1, n2, e_data in graph.edges(data=True):
        e_data[EdgeDataKey.LENGTH] = int(length_matrix[n1][n2])

    # Prune long connections
    if prune_connections_longer_than is not None:
        long_edges = [
            (n1, n2)
            for n1, n2, e_data in graph.edges(data=True)
            if e_data[EdgeDataKey.LENGTH] > prune_connections_longer_than
        ]
        graph.remove_edges_from(long_edges)

    return graph


def save_graph(graph: Graph, file_path: str):
    write_edgelist(graph, file_path,
                   data=[EdgeDataKey.WEIGHT,
                         EdgeDataKey.LENGTH])


def load_graph(file_path: str) -> Graph:
    return read_edgelist(file_path,
                         nodetype=int,
                         data=[(EdgeDataKey.WEIGHT, float),
                               (EdgeDataKey.LENGTH, int)])
