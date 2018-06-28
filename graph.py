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

import logging
import os

from networkx import Graph, from_numpy_matrix, selfloop_edges, write_edgelist, read_edgelist
from numpy import ones_like
from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil

logger = logging.getLogger()


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


def save_graph_from_distance_matrix(file_path: str,
                                    distance_matrix: ndarray,
                                    length_granularity: int,
                                    weighted_graph: bool,
                                    prune_connections_longer_than: int = None):
    """
    Saves a graph of the correct form to underlie a TemporalSpreadingActivation.
    Saved as a networkx edgelist format.

    This can be loaded using `load_graph`.

    It is often faster (and more memory efficient) to save this way than building the graph and then saving it.

    :param file_path:
    :param distance_matrix:
    :param length_granularity:
    :param weighted_graph:
    :param prune_connections_longer_than:
    :return:
    """

    temp_file_path = file_path + ".incomplete"

    # Log progress every time we reach a percentage milestone
    # Record here the most recently logged milestone
    logged_percent_milestone = 0

    with open(temp_file_path, mode="w", encoding="utf8") as temp_file:
        for i in range(0, distance_matrix.shape[0]):
            # Log progress
            percent_done = int(ceil(100 * i / distance_matrix.shape[0]))
            if (percent_done % 10 == 0) and (percent_done > logged_percent_milestone):
                logger.info(f"\t{percent_done}% done")
                logged_percent_milestone = percent_done
            for j in range(i+1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                weight   = float(1.0 - distance if weighted_graph else 1.0)
                length   = int(ceil(distance * length_granularity))
                if (prune_connections_longer_than is not None) and (length > prune_connections_longer_than):
                    continue
                # Write edge to file
                temp_file.write(f"{i} {j} {weight} {length}\n")
    os.rename(temp_file_path, file_path)
