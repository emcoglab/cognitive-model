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
from collections import defaultdict, Sequence
from numbers import Real
from typing import Dict, Set, Tuple, Iterator, DefaultDict, Callable

from numpy import percentile
from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil
from sortedcontainers import SortedSet

logger = logging.getLogger()

Node = int
Length = int


class Edge(tuple):
    def __new__(cls, seq=()):
        assert len(seq) == 2
        # By sorting on init, we guarantee that two edges are equal if their nodes are equal, regardless of order.
        return tuple.__new__(tuple, sorted(seq))


class GraphError(Exception):
    pass


class EdgeExistsError(GraphError):
    pass


class EdgeNotExistsError(GraphError):
    pass


class Graph:
    """
    This is a fragile class that needs to be made more robust.
    Right now Graph.nodes and Graph.edges are Dicts that can be modified at will.
    They should be
    """

    # TODO: Make it more robust by protecting dictionaries from editing outside of the add_* methods.

    def __init__(self, nodes: Set[Node] = None, edges: Dict[Edge, Length] = None):
        # The set of nodes of the graph
        # If modifying, you must also modify .edge_lengths, else we'll end up with edges without endpoints
        self.nodes: Set[Node] = set()
        # The length associated with each edge.
        # If modifying this, you must also modify ._incident_edges, which caches incidence information.
        self.edge_lengths: Dict[Edge, Length] = dict()
        # Node-keyed dict of sets of incident edges
        # Redundant cache for fast lookup.
        self._incident_edges: DefaultDict[Node, Set[Edge]] = defaultdict(set)

        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        if edges is not None:
            for edge, length in edges.items():
                self.add_edge(edge, length)

    @property
    def edges(self):
        return self.edge_lengths.keys()

    def add_edge(self, edge: Edge, length: Length = None):
        """
        Add an edge to the graph, and endpoint nodes.
        :param edge:
        :param length:
        :raises EdgeExistsError if edge already exists in graph.
        :return:
        """
        # Check if edge already added
        if edge in self.edges:
            raise EdgeExistsError(f"Edge {edge} already exists!")
        # Add endpoint nodes
        for node in edge:
            if node not in self.nodes:
                self.add_node(node)
        # Add edge
        self.edge_lengths[edge] = length
        # Add incident edges information
        nodes = list(edge)
        self._incident_edges[nodes[0]].add(edge)
        self._incident_edges[nodes[1]].add(edge)

    def add_node(self, node: Node):
        """Add a bare node to the graph if it's not already there."""
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
            for edge, length in self.edge_lengths.items():
                n1, n2 = sorted(edge)
                edgelist_file.write(f"{Node(n1)} {Node(n2)} {length}\n")

    @classmethod
    def load_from_edgelist_with_arbitrary_stat(cls,
                                               file_path: str,
                                               stat_from_length: Callable[[Edge, Length], Real],
                                               ignore_edges_with_stat_greater_than=None,
                                               keep_at_least_n_edges: int = 0):
        """
        Loads a Graph from an edgelist file, allowing for pruning using an arbitrary statistic derived from length.

        :param file_path:
        :param stat_from_length:
            A function which converts (edge, length) pairs to the statistic based on which edges will be pruned.
        :param ignore_edges_with_stat_greater_than:
            If provided and not None, edges with stat greater than this will not be included in the graph (but the
            endpoint nodes will).
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :return:
        """
        ignoring_outlier_edges = (ignore_edges_with_stat_greater_than is not None)
        if not ignoring_outlier_edges and keep_at_least_n_edges:
            logger.warning(
                f"Requested to keep {keep_at_least_n_edges} but not pruning. This parameter is therefore being ignored.")

        # Keep some edges, selected by the stat
        edges_to_keep = defaultdict(lambda: SortedSet(key=lambda edge_length_pair: stat_from_length(*edge_length_pair)))

        graph = cls()
        for edge, length in iter_edges_from_edgelist(file_path):
            stat = stat_from_length(edge, length)
            if ignoring_outlier_edges and stat > ignore_edges_with_stat_greater_than:
                # Add nodes but not edge
                for node in edge:
                    graph.add_node(node)

                # Keep some edges around to avoid orphans
                if keep_at_least_n_edges:
                    for node in edge:
                        edges_to_keep[node].add((edge, length))
                        # We only want to force-keep the n smallest edges per node, so discard the largest ones once we
                        # have too many
                        if len(edges_to_keep[node]) > keep_at_least_n_edges:
                            edges_to_keep[node].pop(-1)

            else:
                graph.add_edge(edge, length)

        # Add in the edges we decided to keep anyway
        if keep_at_least_n_edges:
            graph.__add_kept_edges(edges_to_keep, keep_at_least_n_edges)

        return graph

    @classmethod
    def load_from_edgelist(cls,
                           file_path: str,
                           ignore_edges_longer_than: Length = None,
                           keep_at_least_n_edges: int = 0) -> 'Graph':
        """
        Loads a Graph from an edgelist file.
        :param file_path:
        :param ignore_edges_longer_than:
            If provided and not None, edges longer than this will not be included in the graph (but the endpoint nodes
            will).
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :return:
        """
        return cls.load_from_edgelist_with_arbitrary_stat(
            file_path=file_path,
            # The stat is just the length
            stat_from_length=lambda edge, length: length,
            ignore_edges_with_stat_greater_than=ignore_edges_longer_than,
            keep_at_least_n_edges=keep_at_least_n_edges
        )

    def __add_kept_edges(self, edges_to_keep_buffer: DefaultDict[Node, SortedSet], keep_at_least_n_edges: int):
        """
        When not keeping all edges, we will want to add some back in, but not all of them.
        This reusable code keeps the logic of which edges we actually want to keep.
        :param edges_to_keep_buffer:
            Node-keyed defaultdict of sortedsets of (edge, stat) tuples
        :param keep_at_least_n_edges:
        :return:
        """
        for node, edges_to_keep_this_node in edges_to_keep_buffer.items():
            # We only want to force-keep *up-to* n edges, so if we've already got some, we don't need to force-add
            # all n in the buffer.
            # So we first forget to add any edges the node already has...
            forget = []
            for edge, length in edges_to_keep_this_node:
                if edge in self.incident_edges(node):
                    forget.append((edge, length))
            for f in forget:
                edges_to_keep_this_node.remove(f)
            # ... and then forget as many others as necessary.
            n_excess_kept_edges = (len(self._incident_edges[node])
                                   + len(edges_to_keep_this_node)
                                   - keep_at_least_n_edges)
            for _ in range(n_excess_kept_edges):
                try:
                    edges_to_keep_this_node.pop(-1)
                # In case many edges have already been added back for this node as the endpoint for edges incident
                # to other nodes, we may have already exceeded our quotient
                # In this case we will try to over-empty this set
                except IndexError:
                    break

            # Finally we add the remaining ones
            for edge, length in edges_to_keep_this_node:
                try:
                    self.add_edge(edge, length)
                # Each edge will end up being recorded twice
                except EdgeExistsError:
                    pass

    @classmethod
    def from_distance_matrix(cls,
                             distance_matrix: ndarray,
                             length_granularity: int,
                             ignore_edges_longer_than: Length = None,
                             keep_at_least_n_edges: int = 0) -> 'Graph':
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
            (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be
            severed.
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :return:
            A Graph of the correct format.
        """
        ignoring_long_edges = (ignore_edges_longer_than is not None)
        if not ignoring_long_edges:
            assert not keep_at_least_n_edges

        edges_to_keep = defaultdict(lambda: SortedSet(key=lambda x: x[1]))

        graph = cls()

        n_nodes = distance_matrix.shape[0]

        for n1 in range(0, n_nodes):
            graph.add_node(n1)
            for n2 in range(n1 + 1, n_nodes):
                graph.add_node(n2)
                edge = Edge((n1, n2))
                distance = distance_matrix[n1, n2]
                length = Length(ceil(distance * length_granularity))
                # Skip the edge if we're pruning and it's too long
                if not ignoring_long_edges or length <= ignore_edges_longer_than:
                    graph.add_edge(edge, length)
                else:
                    # But keep a few around so we don't get orphans
                    if keep_at_least_n_edges:
                        edges_to_keep[n1].add((edge, length))
                        edges_to_keep[n2].add((edge, length))
                        # But don't keep too many
                        if len(edges_to_keep[n1]) > keep_at_least_n_edges:
                            edges_to_keep[n1].pop(-1)
                        if len(edges_to_keep[n2]) > keep_at_least_n_edges:
                            edges_to_keep[n2].pop(-1)

        # Add in the edges we decided to keep anyway
        if keep_at_least_n_edges:
            graph.__add_kept_edges(edges_to_keep, keep_at_least_n_edges)

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
                        graph.add_edge(Edge((n1, n2)), length)
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

    def _is_orphaned(self, node: Node) -> bool:
        return len(self._incident_edges[node]) == 0

    def _iter_orphaned_nodes(self) -> Iterator[Node]:
        """Iterator of orphaned nodes."""
        for node in self.nodes:
            if self._is_orphaned(node):
                yield node

    def orphaned_nodes(self) -> Set[Node]:
        """The set of orphaned nodes."""
        return set(self._iter_orphaned_nodes())

    def has_orphaned_nodes(self) -> bool:
        """Returns True if the graph has an orphaned node, and False otherwise."""
        return any(True for _ in self._iter_orphaned_nodes())

    # endregion

    # region pruning

    def prune_longest_edges_by_length(self, length_threshold: Length, keep_at_least_n_edges: int = 0):
        """
        Prune the longest edges in the graph by length.
        :param length_threshold:
            Edges will be pruned if they are strictly longer than this threshold.
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :return:
        """

        edges_to_prune = set()
        edges_to_keep = defaultdict(lambda: SortedSet(key=lambda x: x[1]))

        for edge in self.edges:
            length = self.edge_lengths[edge]
            if length > length_threshold:
                edges_to_prune.add(edge)

                if keep_at_least_n_edges:
                    for node in edge:
                        edges_to_keep[node].add((edge, length))
                        # If we've got too many edges to keep now, drop the largest
                        if len(edges_to_keep[node]) > keep_at_least_n_edges:
                            edges_to_keep[node].pop(-1)

        # Prune the edges
        for edge in edges_to_prune:
            self.remove_edge(edge)

        # Add back in the edges we wanted to keep anyway
        if keep_at_least_n_edges:
            self.__add_kept_edges(edges_to_keep, keep_at_least_n_edges)

    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the graph. Does not remove endpoint nodes.
        :param edge:
        :raises EdgeNotExistsError if edge does not exist in the graph.
        :return:
        """
        if edge not in self.edges:
            raise EdgeNotExistsError(f"Edge {edge} does not exist.")
        # Remove from edge dictionary
        self.edge_lengths.pop(edge)
        # Remove from redundant adjacency dictionary
        n1, n2 = edge
        self._incident_edges[n1].remove(edge)
        self._incident_edges[n2].remove(edge)

    def prune_longest_edges_by_quantile(self, quantile: float, keep_at_least_n_edges: int = 0):
        """
        Prune the longest edges in the graph by quantile.
        :param quantile:
            The quantile by which to prune the graph.
            So a value of 0.1 will result in the longest 10% of edges being pruned.
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :return:
        """
        # We invert the quantile, so that if `quantile` is 0.1, we prune the TOP 10% (i.e. prune at the 90th centile)
        pruning_quantile = 1 - quantile
        pruning_length = edge_length_quantile([length for edge, length in self.edge_lengths], pruning_quantile)
        self.prune_longest_edges_by_length(pruning_length, keep_at_least_n_edges)

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
            # Write
            for j in range(i + 1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                length = Length(ceil(distance * length_granularity))
                # Write edge to file
                temp_file.write(f"{i} {j} {length}\n")
    os.rename(temp_file_path, file_path)


def edge_length_quantile(lengths, quantile):
    """
    Return the quantile(s) at a specified length.
    :param lengths
        Sequence of lengths from which to form a distribution.
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
    # noinspection PyTypeChecker
    length = percentile(lengths,
                        # I don't know why Pycharm thinks its expecting an int here; it shouldn't be
                        centile, interpolation="nearest")
    return length


def iter_edges_from_edgelist(file_path: str) -> Iterator[Tuple[Edge, Length]]:
    """Yields tuples of (edge: Edge, edge_lengths: EdgeData) from an edgelist file."""
    with open(file_path, mode="r", encoding="utf-8") as edgelist_file:
        for line in edgelist_file:
            n1, n2, length = line.split()
            yield Edge((Node(n1), Node(n2))), Length(length)
