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
from collections import namedtuple, defaultdict, Sequence
from typing import Dict, Set, Tuple, Iterator, DefaultDict

from numpy import percentile
from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil

logger = logging.getLogger()

Node = int
EdgeData = namedtuple('EdgeData', ['length'])


class Edge(frozenset):
    def __init__(self, seq=()):
        assert len(seq) == 2
        frozenset.__init__(seq)

    @property
    def nodes(self) -> Tuple[Node, Node]:
        return tuple(self)


class GraphError(Exception):
    pass


class Graph:
    """
    This is a fragile class that needs to be made more robust.
    Right now Graph.nodes and Graph.edges are Dicts that can be modified at will.
    They should be
    """

    # TODO: Make it more robust by protecting dictionaries from editing outside of the add_* methods.

    def __init__(self, nodes: Set[Node] = None, edges: Dict[Edge, EdgeData] = None):
        # The set of nodes of the graph
        # If modifying, you must also modify .edge_data, else we'll end up with edges without endpoints
        self.nodes: Set[Node] = set()
        # The data associated with each edge.
        # If modifying this, you must also modify ._incident_edges, which caches incidence information.
        self.edge_data: Dict[Edge, EdgeData] = dict()
        # Node-keyed dict of sets of incident edges
        # Redundant cache for fast lookup.
        self._incident_edges: DefaultDict[Node, Set[Edge]] = defaultdict(set)

        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        if edges is not None:
            for edge, edge_data in edges.items():
                self.add_edge(edge, edge_data)

    @property
    def edges(self):
        return self.edge_data.keys()

    def add_edge(self, edge: Edge, edge_data: EdgeData = None):
        # Check if edge already added
        if edge in self.edges:
            raise GraphError(f"Edge {edge} already exists!")
        # Add endpoint nodes
        for node in edge:
            if node not in self.nodes:
                self.add_node(node)
        # Add edge
        self.edge_data[edge] = edge_data
        # Add incident edges information
        nodes = list(edge)
        self._incident_edges[nodes[0]].add(edge)
        self._incident_edges[nodes[1]].add(edge)

    def add_node(self, node: Node):
        if node not in self.nodes:
            self.nodes.add(node)

    def incident_edges(self, node: Node) -> Iterator[Edge]:
        """The edges which have `node` as an endpoint."""
        for edge in self._incident_edges[node]:
            yield edge

    def neighbourhood(self, node: Node) -> Iterator[Node]:
        """The nodes which are connected to `node` by exactly one edge."""
        assert node in self._incident_edges.keys()
        for edge in self._incident_edges[node]:
            for n in edge:
                # Skip the source node
                if n == node:
                    continue
                yield n

    # region IO

    def save_as_edgelist(self, file_path: str):
        """Saves a Graph as an edgelist. Disconnected nodes will not be included."""
        with open(file_path, mode="w", encoding="utf-8") as edgelist_file:
            for edge, edge_data in self.edge_data.items():
                n1, n2 = sorted(edge)
                length = int(edge_data.length)
                edgelist_file.write(f"{Node(n1)} {Node(n2)} {length}\n")

    @classmethod
    def load_from_edgelist(cls, file_path: str, ignore_edges_longer_than: int = None) -> 'Graph':
        """
        Loads a Graph from an edgelist file.
        :param file_path:
        :param ignore_edges_longer_than:
            If provided and not None, edges longer than this will not be included in the graph (but the endpoint nodes will).
        :return:
        """
        graph = cls()
        for edge, edge_data in edge_data_from_edgelist(file_path):
            if ignore_edges_longer_than is not None and edge_data.length > ignore_edges_longer_than:
                n1, n2 = edge.nodes
                # Add nodes but not edge
                graph.add_node(n1)
                graph.add_node(n2)
                continue
            graph.add_edge(edge, edge_data)
        return graph

    @classmethod
    def from_distance_matrix(cls,
                             distance_matrix: ndarray,
                             length_granularity: int,
                             ignore_edges_longer_than: int = None) -> 'Graph':
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
            Whether to use weights on the edges.
            If True, distances will be converted to weights using x ↦ 1-x.
                (This means it's only suitable for things like cosine and correlation distances, not Euclidean.)
            If False, all edges get the same weight.
        :param ignore_edges_longer_than:
            (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be severed.
        :return:
            A Graph of the correct format.
        """

        graph = cls()

        n_nodes = distance_matrix.shape[0]

        for n1 in range(0, n_nodes):
            graph.add_node(n1)
            for n2 in range(n1 + 1, n_nodes):
                distance = distance_matrix[n1, n2]
                length = int(ceil(distance * length_granularity))
                # Skip the edge if we're pruning and it's too long
                if (ignore_edges_longer_than is not None) and (length > ignore_edges_longer_than):
                    continue
                # Add the edge
                graph.add_edge(Edge((n1, n2)), EdgeData(length=length))

        return graph

    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix: ndarray, length: int = None) -> 'Graph':
        graph = cls()

        n_nodes = adjacency_matrix.shape[0]

        for n1 in range(0, n_nodes):
            graph.add_node(n1)
            for n2 in range(n1 + 1, n_nodes):
                if adjacency_matrix[n1, n2]:
                    if length is not None:
                        graph.add_edge(Edge((n1, n2)), EdgeData(length=length))
                    else:
                        graph.add_edge(Edge((n1, n2)))

        return graph

    # endregion IO

    # region topology

    def is_connected(self) -> bool:
        """Returns True if the graph is connected, and False otherwise."""
        # We pick a node at random, and see how many other nodes we can visit from it, then see if we've got everywhere.
        # Use a breadth-first search
        visited_nodes = set()
        search_queue = set()
        starting_node = list(self.nodes)[0]
        visited_nodes.add(starting_node)
        search_queue.add(starting_node)
        while len(search_queue) > 0:
            current_node = search_queue.pop()
            neighbouring_nodes = set(node for edge in self._incident_edges[current_node] for node in edge)
            for node in neighbouring_nodes:
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    search_queue.add(node)
        # Check if we visited all the nodes
        if len(visited_nodes) == len(self.nodes):
            return True
        else:
            return False

    def has_orphaned_nodes(self) -> bool:
        """Returns True if the graph has an orphaned node, and False otherwise."""
        for node in self.nodes:
            # Orphaned nodes have no incident edges
            if len(self._incident_edges[node]) == 0:
                return True
        return False

    # endregion

    # region topography and length metrics

    def edge_length_quantile(self, quantile):
        """
        Return the quantile(s) at a specified length.
        :param quantile:
            float in range of [0,1] (or sequence of floats)
        :return:
            length (or sequence of lengths) marking specified quantile.
            Returned lengths will not be interpolated - nearest lengths to the quantile will be given.
        """
        # If one quantile provided
        if isinstance(quantile, float):
            centile = 100 * quantile
        # If sequence of quantiles provided
        elif isinstance(quantile, Sequence):
            centile = [100 * q for q in quantile]
        else:
            raise TypeError()

        length = percentile([self.edge_data[edge].length for edge in self.edges], centile, interpolation="nearest")

        return length

    # endregion

    # region pruning

    def prune_longest_edges_by_length(self, length_threshold: int):
        """
        Prune the longest edges in the graph by length.
        :param length_threshold:
            Edges will be pruned if they are strictly longer than this threshold.
        :return:
        """
        edges_to_prune = []
        for edge in self.edges:
            length = self.edge_data[edge].length
            if length > length_threshold:
                edges_to_prune.append(edge)
        for edge in edges_to_prune:
            self.remove_edge(edge)

    def remove_edge(self, edge: Edge):
        """Remove an edge from the graph. Does not remove endpoint nodes."""
        # Remove from edge dictionary
        self.edge_data.pop(edge)
        # Remove from redundant adjacency dictionary
        n1, n2 = edge.nodes
        self._incident_edges[n1].remove(edge)
        self._incident_edges[n2].remove(edge)

    def prune_longest_edges_by_quantile(self, quantile: float):
        """
        Prune the longest edges in the graph by quantile.
        :param quantile:
            The quantile by which to prune the graph.
            So a value of 0.1 will result in the longest 10% of edges being pruned.
        :return:
        """
        # We invert the quantile, so that if `quantile` is 0.1, we prune the TOP 10% (i.e. prune at the 90th centile)
        pruning_quantile = 1 - quantile
        pruning_length = self.edge_length_quantile(pruning_quantile)
        self.prune_longest_edges_by_length(pruning_length)

    # endregion


def save_edgelist_from_distance_matrix(file_path: str,
                                       distance_matrix: ndarray,
                                       length_granularity: int):
    """
    Saves a graph of the correct form to underlie a TemporalSpreadingActivation.
    Saved as a networkx-compatible edgelist format.

    This can be loaded using `load_graph`.

    It is often faster (and more memory efficient) to save this way than building the graph and then saving it.

    :param file_path:
    :param distance_matrix:
    :param length_granularity:
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
            for j in range(i + 1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                length = int(ceil(distance * length_granularity))
                # Write edge to file
                temp_file.write(f"{i} {j} {length}\n")
    os.rename(temp_file_path, file_path)


def edge_data_from_edgelist(file_path: str) -> Iterator[Tuple[Edge, EdgeData]]:
    """Yields tuples of (edge: Edge, edge_data: EdgeData) from an edgelist file."""
    with open(file_path, mode="r", encoding="utf-8") as edgelist_file:
        for line in edgelist_file:
            n1, n2, length = line.split()
            yield Edge((Node(n1), Node(n2))), EdgeData(length=int(length))
