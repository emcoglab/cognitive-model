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

from __future__ import annotations

import os
from collections import defaultdict
from enum import Enum, auto
from numbers import Real
from typing import Dict, Set, Tuple, Iterator, DefaultDict, List

from numpy import nan
from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import percentileofscore
from sortedcontainers import SortedSet

from .ldm.utils.log import print_progress
from .basic_types import Node, Length
from .utils.maths import mean, nearest_value_at_quantile, distance_from_similarity
from .utils.log import logger


class Edge(tuple):
    """
    Edge in the graph.
    A (sorted) tuple of Nodes.
    """
    def __new__(cls, seq=()):
        assert len(seq) == 2
        # By sorting on init, we guarantee that two edges are equal iff their nodes are equal, regardless of order.
        return tuple.__new__(tuple, sorted(seq))


def edgelist_line(from_edge: Edge, with_length: Length):
    """
    Converts an edge and a length into a line to be written into an edgelist file,
    complete with trailing newline.
    """
    n1, n2 = sorted(from_edge)
    return f"{Node(n1)} {Node(n2)} {with_length}\n"


class GraphError(Exception):
    pass


class EdgeExistsError(GraphError):
    pass


class EdgeNotExistsError(GraphError):
    pass


class Graph:

    # TODO: This is a fragile class that needs to be made more robust.
    #  Right now Graph.nodes and Graph.edges are Dicts that can be modified at will.
    #  It should be made more robust by protecting dicts and defaultdicts from editing outside of the add_* methods.

    def __init__(self, nodes: Set[Node] = None, edges: Dict[Edge, Length] = None):
        # The set of nodes of the graph
        # If modifying, you must also modify .edge_lengths, else we'll end up with edges without endpoints
        self.nodes: Set[Node] = set()
        # The length associated with each edge.
        # If modifying this, you must also modify ._incident_edges, which caches incidence information.
        self.edge_lengths: Dict[Edge, Length] = dict()
        # Cache of Node-keyed dict of lists of incident edges.
        # When modifying the graph using add_edge or remove_edge this will become invalid and must be rebuilt before use
        # So it's a good idea to batch modifications.
        self.__incident_edges: DefaultDict[Node, List[Edge]] = defaultdict(list)  # Backing for self._incident_edges
        self.__incident_edges_cache_is_valid: bool = False

        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        if edges is not None:
            for edge, length in edges.items():
                self.add_edge(edge, length)

        self.__rebuild_incident_edges_cache()

    @property
    def _incident_edges(self) -> DefaultDict[Node, List[Edge]]:
        """Prefer to use self.edges_incident_to(node)."""
        if not self.__incident_edges_cache_is_valid:
            self.__rebuild_incident_edges_cache()
        return self.__incident_edges

    def edges_incident_to(self, node: Node) -> List[Edge]:
        """The edges which have `node` as an endpoint."""
        return self._incident_edges[node]

    def __rebuild_incident_edges_cache(self):
        logger.info("Rebuilding incident edge cache")
        # Use sets for quick deduplication
        incident_edges = defaultdict(set)
        for edge in self.edges:
            # Add incident edges information
            for node in edge:
                incident_edges[node].add(edge)
        # Convert back to list
        self.__incident_edges = defaultdict(list, {
            node: list(edges)
            for node, edges in incident_edges.items()
        })
        self.__incident_edges_cache_is_valid = True

    @property
    def edges(self):
        return self.edge_lengths.keys()

    def add_edge(self, edge: Edge, length: Length = None):
        """
        Add an edge to the graph, and endpoint nodes.
        Invalidates the cache.
        :param edge:
        :param length:
        :raises EdgeExistsError if edge already exists in graph.
        :return:
        """
        self.__incident_edges_cache_is_valid = False
        # Check if edge already added
        if edge in self.edges:
            raise EdgeExistsError(f"Edge {edge} already exists!")
        # Add endpoint nodes
        for node in edge:
            if node not in self.nodes:
                self.add_node(node)
        # Add edge
        self.edge_lengths[edge] = length

    def add_node(self, node: Node):
        """Add a bare node to the graph if it's not already there."""
        # Adding a node does not itself invalidate the cache
        if node not in self.nodes:
            self.nodes.add(node)

    def neighbourhood(self, node: Node) -> Iterator[Node]:
        """The nodes which are connected to `node` by exactly one edge."""
        assert node in self.nodes
        for edge in self.edges_incident_to(node):
            for n in edge:
                # Skip the source node
                if n == node:
                    continue
                yield n

    # region IO

    def save_as_pickle(self, file_path: str, validate_cache: bool = False):
        """Pickles this Graph object."""
        if validate_cache and not self.__incident_edges_cache_is_valid:
            self.__rebuild_incident_edges_cache()
        import pickle
        with open(file_path, mode="wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_pickle(cls, file_path: str) -> Graph:
        """Unpickles a Graph object."""
        import pickle
        with open(file_path, mode="rb") as file:
            graph = pickle.load(file)
        return graph

    def save_as_edgelist(self, file_path: str):
        """Saves a Graph as an edgelist. Disconnected nodes will not be included."""
        with open(file_path, mode="w", encoding="utf-8") as edgelist_file:
            for edge, length in self.edge_lengths.items():
                edgelist_file.write(edgelist_line(from_edge=edge, with_length=length))

    @classmethod
    def load_from_edgelist_with_importance_pruning(cls,
                                                   file_path: str,
                                                   ignore_edges_with_importance_greater_than: Real = None,
                                                   keep_at_least_n_edges: int = 0):
        """
        The importance of an edge is the average percentile score of its length in the distribution of edge lengths
        incident to each of its endpoint nodes.
        """
        ignoring_outlier_edges = (ignore_edges_with_importance_greater_than is not None)
        if not ignoring_outlier_edges and keep_at_least_n_edges:
            logger.warning(
                f"Requested to keep {keep_at_least_n_edges} edges but not pruning. "
                f"This parameter is therefore being ignored.")

        # Run through edgelist first, and build distributions of lengths
        edge_length_distributions = defaultdict(list)
        for edge, length in iter_edges_from_edgelist(file_path):
            for node in edge:
                edge_length_distributions[node].append(length)

        # Run through distributions, compute length -> per-node percentile mapping
        # node -> length -> percentile
        length_percentile_mapping = defaultdict(lambda: defaultdict(float))
        for node, length_list in edge_length_distributions.items():
            for length in length_list:
                # only work with keys not yet processed
                if length not in length_percentile_mapping[node].keys():
                    length_percentile_mapping[node][length] = percentileofscore(length_list, length)
        # freemem
        del edge_length_distributions

        def local_importance(e: Edge, l: Length):
            n1, n2 = e
            return mean(length_percentile_mapping[n1][l], length_percentile_mapping[n2][l])

        # Keep some edges, selected by importance
        edges_to_keep = defaultdict(lambda: SortedSet(key=lambda edge_importance_pair: edge_importance_pair[1]))

        graph = cls()
        for edge, length in iter_edges_from_edgelist(file_path=file_path):
            if not ignoring_outlier_edges:
                graph.add_edge(edge, length)
            else:
                i = local_importance(edge, length)
                if i <= ignore_edges_with_importance_greater_than:
                    graph.add_edge(edge, length)
                else:
                    for node in edge:
                        graph.add_node(node)
                    if keep_at_least_n_edges:
                        for node in edge:
                            edges_to_keep[node].add((edge, length))
                            if len(edges_to_keep[node]) > keep_at_least_n_edges:
                                edges_to_keep[node].pop(-1)

        # add edges we decided to keep
        if keep_at_least_n_edges:
            graph.__add_kept_edges(edges_to_keep, keep_at_least_n_edges)

        graph.__rebuild_incident_edges_cache()

        return graph

    @classmethod
    def load_from_edgelist(cls,
                           file_path: str,
                           ignore_edges_longer_than: Length = None,
                           keep_at_least_n_edges: int = 0,
                           with_feedback: bool = False) -> Graph:
        """
        Loads a Graph from an edgelist file.
        :param file_path:
        :param ignore_edges_longer_than:
            If provided and not None, edges longer than this will not be included in the graph (but the endpoint nodes
            will).
        :param keep_at_least_n_edges:
            Default 0.
            Make sure each node keeps at least this number of edges.
        :param with_feedback:
            If true, logs feedback
        :return:
        """
        ignoring_long_edges = (ignore_edges_longer_than is not None)
        if not ignoring_long_edges and keep_at_least_n_edges:
            logger.warning(f"Requested to keep {keep_at_least_n_edges} but not pruning. "
                           f"This parameter is therefore being ignored.")

        edges_to_keep = defaultdict(lambda: SortedSet(key=lambda x: x[1]))

        graph = cls()
        for i, (edge, length) in enumerate(iter_edges_from_edgelist(file_path)):
            if with_feedback and i > 0 and i % 1_000_000 == 0:
                logger.info(f"Read {i:,} edges")
            if ignoring_long_edges and length > ignore_edges_longer_than:
                # Add nodes but not edge
                # This is the tricky bit which means we can't use prepruned graphs
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

                continue
            graph.add_edge(edge, length)

        # Add in the edges we decided to keep anyway
        if keep_at_least_n_edges:
            graph.__add_kept_edges(edges_to_keep, keep_at_least_n_edges)

        graph.__rebuild_incident_edges_cache()

        return graph

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
            # all n in the working_memory_buffer.
            # So we first forget to add any edges the node already has...
            forget = []
            for edge, length in edges_to_keep_this_node:
                if edge in self.edges_incident_to(node):
                    forget.append((edge, length))
            for f in forget:
                edges_to_keep_this_node.remove(f)
            # ... and then forget as many others as necessary.
            n_excess_kept_edges = (len(self.edges_incident_to(node))
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
                             keep_at_least_n_edges: int = 0) -> Graph:
        """
        Produces a Graph of the correct format to underlie a GraphPropagator.

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

        graph.__rebuild_incident_edges_cache()

        return graph

    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix: ndarray, length: int = None) -> Graph:
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

        graph.__rebuild_incident_edges_cache()

        return graph

    # endregion IO

    # region topology

    def is_connected(self) -> bool:
        """Returns True if the graph is connected, and False otherwise."""
        # We pick a node at random, and see how many other nodes we can visit from it, then see if we've got everywhere.
        # Use a breadth-first search.
        visited_nodes = set()
        search_queue = set()
        starting_node = list(self.nodes)[0]
        visited_nodes.add(starting_node)
        search_queue.add(starting_node)
        while len(search_queue) > 0:
            current_node = search_queue.pop()
            neighbouring_nodes = set(node for edge in self.edges_incident_to(current_node) for node in edge)
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
        return len(self.edges_incident_to(node)) == 0

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

        self.__rebuild_incident_edges_cache()

    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the graph. Does not remove endpoint nodes.
        Invalidates the cache
        :param edge:
        :raises EdgeNotExistsError if edge does not exist in the graph.
        :return:
        """
        self.__incident_edges_cache_is_valid = False
        if edge not in self.edges:
            raise EdgeNotExistsError(f"Edge {edge} does not exist.")
        # Remove from edge dictionary
        self.edge_lengths.pop(edge)

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
        pruning_length = nearest_value_at_quantile([length for edge, length in self.edge_lengths], pruning_quantile)
        self.prune_longest_edges_by_length(pruning_length, keep_at_least_n_edges)

    # endregion

    def print(self):
        print(f"{len(self.nodes)} nodes")
        print(f"{len(self.edges)} edges")
        for (n1, n2), l in self.edge_lengths.items():
            print(f"{n1} → {n2}: {l}")


def length_from_distance(distance: float, length_factor: int) -> Length:
    return Length(ceil(distance * length_factor))


def save_edgelist_from_distance_matrix(file_path: str,
                                       distance_matrix: ndarray,
                                       length_factor: int):
    """
    Saves a graph of the correct form to underlie a GraphPropagator.
    Saved as a networkx-compatible edgelist format.

    This can be loaded using `load_graph`.

    It is often faster (and more memory efficient) to save this way than building the graph and then saving it.

    :param file_path:
    :param distance_matrix:
    :param length_factor:
    :return:
    """

    temp_file_path = file_path + ".incomplete"

    with open(temp_file_path, mode="w", encoding="utf8") as temp_file:
        i_max = distance_matrix.shape[0]
        j_max = distance_matrix.shape[1]
        for i in range(0, i_max):
            for j in range(i + 1, j_max):
                distance = distance_matrix[i, j]
                length = length_from_distance(distance, length_factor)
                assert length > 0
                # Write edge to file
                temp_file.write(f"{i} {j} {length}\n")
            print_progress(i, i_max-1)

    # When done writing to the temp file, rename it to the finished file
    os.rename(temp_file_path, file_path)


def save_edgelist_from_similarity_matrix(file_path: str,
                                         similarity_matrix: csr_matrix,
                                         filtered_node_ids: List[int],
                                         length_factor: int):
    """
    Saves a graph of the correct form to underlie a GraphPropagator.
    Saved as a networkx-compatible edgelist format.

    This can be loaded using `load_graph`.

    It is often faster (and more memory efficient) to save this way than building the graph and then saving it.

    :param file_path:
    :param similarity_matrix:
    :param filtered_node_ids:
    :param length_factor:
    :return:
    """

    temp_file_path = file_path + ".incomplete"

    # Determine max and min similarities over WHOLE similarity matrix, before filtering

    # Drop zeros to make sure the min is non-zero
    similarity_matrix.eliminate_zeros()
    if similarity_matrix.shape == (0, 0):
        logger.warning("Empty matrix encountered")
        max_value = min_value = nan
    else:
        max_value = similarity_matrix.data.max()
        min_value = similarity_matrix.data.min()

    # Filter similarity matrix rows and columns by supplied ids
    similarity_matrix = similarity_matrix.tocsr()[filtered_node_ids, :].tocsc()[:, filtered_node_ids]

    # number of non-zero values, used for logging progress
    n_values_for_logging_progress = similarity_matrix.nnz

    # Convert to coo for fast iteration
    similarity_matrix = coo_matrix(similarity_matrix)

    with open(temp_file_path, mode="w", encoding="utf8") as temp_file:

        n_values_considered = 0

        # Iterate over non-zero entries, which are the ones which should correspond to edges in the matrix
        for i, j, v in zip(similarity_matrix.row, similarity_matrix.col, similarity_matrix.data):
            # only want half of the symmetric matrix, and no diagonal
            if j <= i:
                continue
            length: Length = length_from_distance(distance=distance_from_similarity(v, max_value, min_value),
                                                  length_factor=length_factor)
            assert length > 0
            # Write edge to file
            temp_file.write(f"{i} {j} {length}\n")

            # Log occasionally
            n_values_considered += 1
            if (n_values_considered == 0
                    or n_values_considered == n_values_for_logging_progress
                    or n_values_considered % 100 == 0):
                # Double the % done as we only look at one half of the symmetric matrix (making this value approx, as we
                # ignore diagonal entries).
                print_progress(n_values_considered * 2, n_values_for_logging_progress)
    # make sure we get the 100%
    print_progress(n_values_for_logging_progress, n_values_for_logging_progress)

    # When done writing to the temp file, rename it to the finished file
    os.rename(temp_file_path, file_path)


def iter_edges_from_edgelist(file_path: str) -> Iterator[Tuple[Edge, Length]]:
    """Yields tuples of (edge: Edge, length: Length) from an edgelist file."""
    with open(file_path, mode="r", encoding="utf-8") as edgelist_file:
        for line in edgelist_file:
            n1, n2, length = line.split()
            assert Length(length) > 0
            yield Edge((Node(n1), Node(n2))), Length(length)


def iter_edges_from_edgelist_with_pruning(file_path: str,
                                          ignore_edges_longer_than: Length) -> Iterator[Tuple[Edge, Length]]:
    """Yields tuples of (edge: Edge, length: Length) from an edgelist file."""
    for edge, length in iter_edges_from_edgelist(file_path):
        if length > ignore_edges_longer_than:
            continue
        yield edge, length


def log_graph_topology(graph) -> Tuple[bool, bool]:
    """
    :param graph:
    :return: graph.is_connected, graph.has_orphans
    """
    logger.info(f"Graph has {len(graph.edges):,} edges")
    connected = graph.is_connected()
    orphans = graph.has_orphaned_nodes()
    if orphans:
        logger.info("Graph has orphaned nodes.")
    else:
        logger.info("Graph does not have orphaned nodes")
    if connected:
        logger.info("Graph is connected")
    else:
        logger.info("Graph is not connected")
    return connected, orphans


class EdgePruningType(Enum):
    Length     = auto()
    Percent    = auto()
    Importance = auto()

    @property
    def name(self) -> str:
        if self == EdgePruningType.Length:
            return "Length"
        elif self == EdgePruningType.Percent:
            return "Percent"
        elif self == EdgePruningType.Importance:
            return "Importance"
        else:
            raise NotImplementedError()

    @classmethod
    def from_name(cls, name: str) -> EdgePruningType:
        for t in EdgePruningType:
            if name.lower() == t.name.lower():
                return t
        raise NotImplementedError()
