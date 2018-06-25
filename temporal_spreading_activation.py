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

import logging
from collections import namedtuple, defaultdict
from typing import Set, Dict, DefaultDict

from networkx import Graph, from_numpy_matrix, selfloop_edges
from numpy import exp, ndarray, ones_like, ceil, float_power

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class EdgeDataKey(object):
    """Column names for edge data."""
    WEIGHT = "weight"
    LENGTH = "length"


class NodeActivationRecord(namedtuple('NodeActivationRecord', ['activation',
                                                               'time_activated'
                                                               ])):
    """
    NodeActivationRecord stores a historical node activation event.
    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.
    """
    # Tell Python no more fields can be added to this class, so it stays small in memory.
    __slots__ = ()


def blank_node_activation_record() -> NodeActivationRecord:
    """A record for an unactivated node."""
    return NodeActivationRecord(activation=0, time_activated=-1)


class Impulse(namedtuple("Impulse", ['source_node',
                                     'target_node',
                                     'departure_time',
                                     'arrival_time',
                                     'departure_activation',
                                     'arrival_activation'
                                     ])):
    """
    Stores information about an impulse.
    Impulses are what I'm calling activation that is spreading along a connection.
    """
    # Tell Python no more fields can be added to this class, so it stays small in memory.
    __slots__ = ()

    def age_at_time(self, t) -> int:
        """The age of this impulse at a specified time."""
        return t - self.departure_time


class TemporalSpreadingActivation(object):

    def __init__(self,
                 graph: Graph,
                 node_relabelling_dictionary: Dict,
                 activation_threshold: float,
                 impulse_pruning_threshold: float,
                 node_decay_function: callable,
                 edge_decay_function: callable):
        """
        :param graph:
            `graph` should be an undirected, weighted graph with the following data:
                On Edges:
                    weight
                    length
        :param node_relabelling_dictionary:
            A dictionary whose keys are nodes, and whose values are node labels.
            This lets us keep the graph data stored throughout as ints rather than strings, saving a bunch of memory.
        :param activation_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        :param impulse_pruning_threshold
            Any impulse which decays to less than this threshold before reaching its destination will be deleted.
        :param node_decay_function:
            A function governing the decay of activations on nodes.
            Use the decay_function_*_with_params methods to create these.
        :param edge_decay_function:
            A function governing the decay of activations in connections.
            Use the decay_function_*_with_* methods to create these.
        """

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Thresholds
        # Use < and >= to test for above/below
        self.activation_threshold = activation_threshold
        self.impulse_pruning_threshold = impulse_pruning_threshold

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        self.node_decay_function: callable = node_decay_function
        self.edge_decay_function: callable = edge_decay_function

        # Node label dictionaries
        # node ↦ label
        self.node2label: Dict = node_relabelling_dictionary
        # label ↦ node
        self.label2node: Dict = dict((v, k) for k, v in node_relabelling_dictionary.items())

        # Zero-indexed tick counter.
        self.clock: int = 0

        # Graph data:

        # A node-keyed dictionaries of node activations
        # Stores the most recent activation of each node, if any
        self._node_activation_records = defaultdict(blank_node_activation_record)

        # Impulses are stored in an arrival-time-keyed dict of destination-node-keyed dicts of lists of impulses
        # scheduled for arrival.
        # This way, when an arrival time is reached, we can .pop() a destination-node-keyed dict of impulses to process.
        # Nice!
        # TODO: if this is still too slow, we could just store arrival_activations in the inner dict. This would give us
        # TODO: everything we *need*, but would make display and tracking a bit harder.
        # ACTUALLY we'll use a defaultdict here, so we can quickly and easily add an impulse in the right place without
        # verbose checks
        self._impulses: DefaultDict = defaultdict(
            # In case the aren't any impulses due to arrive at a particular time, we'll just find an empty dict
            lambda: defaultdict(
                # In case there aren't any impulses due to arrive at a particular node, we'll just find an empty list
                list
            ))

    def n_suprathreshold_nodes(self) -> int:
        """
        The number of nodes which are above the activation threshold.
        May take a long time to compute.
        :return:
        """
        return len([
            n
            for n in self.graph.nodes
            if self.activation_of_node(n) >= self.activation_threshold
        ])

    def impulses_by_edge(self, n1, n2) -> Set:
        """The set of impulses in the (undirected) edge with endpoints n1, n2."""
        d = defaultdict(set)
        for t, impulse_dict in self._impulses:
            for destination_node, impulses in impulse_dict:
                for i in impulses:
                    d[(i.source_node, destination_node)].add(i)
        return d[(n1, n2)].union(d[(n2, n1)])

    def activation_of_node(self, n) -> float:
        """Returns the current activation of a node."""
        activation_record: NodeActivationRecord = self._node_activation_records[n]
        return self.node_decay_function(
            # node age
            self.clock - activation_record.time_activated,
            activation_record.activation)

    def activate_node(self, n, activation: float):
        """Activates a node."""

        current_activation = self.activation_of_node(n)

        # If this node is currently suprathreshold, it acts as a sink.
        # It doesn't accumulate new activation and cannot fire.
        if current_activation >= self.activation_threshold:
            return

        # Otherwise, we proceed with the activation:

        # Accumulate activation
        new_activation = current_activation + activation
        self._node_activation_records[n] = NodeActivationRecord(new_activation, self.clock)

        # Check if we reached the threshold
        if new_activation >= self.activation_threshold:

            # If so, Fire!

            # Fire and rebroadcast
            source_node = n

            # For each incident edge
            for n1, n2, e_data in self.graph.edges(source_node, data=True):
                # Determine source and target node
                if source_node == n1:
                    target_node = n2
                elif source_node == n2:
                    target_node = n1
                else:
                    raise ValueError()

                edge_length = e_data[EdgeDataKey.LENGTH]
                departure_activation = e_data[EdgeDataKey.WEIGHT] * new_activation
                arrival_activation = self.edge_decay_function(edge_length, departure_activation)

                # Skip any impulses which will be too small on arrival
                if arrival_activation < self.impulse_pruning_threshold:
                    continue

                arrival_time = self.clock + edge_length

                # We pre-compute the impulses now rather than decaying them over time.
                # Intermediate activates can be computed for display purposes if necessary.
                impulse = Impulse(
                    source_node=source_node,
                    target_node=target_node,
                    departure_time=self.clock,
                    arrival_time=arrival_time,
                    departure_activation=departure_activation,
                    arrival_activation=arrival_activation
                )

                # Since a node can only fire once when it first passes threshold, it should be logically impossible for
                # there to be an existing impulse with the same age and target released from this node.
                # This means we can just remember ALL impulses that are ever released, without fear that they'll ever be
                # overlapping.
                self._impulses[arrival_time][target_node].append(impulse)

    def _propagate_impulses(self):
        """Propagates impulses along connections."""

        # "Propagation" happens by just incrementing the global clock.

        # But we have to check if any impulses have reached their destination.
        # This should be a destination-node-keyed dict of lists of impulses
        if self.clock in self._impulses:
            impulses_at_destination: DefaultDict = self._impulses.pop(self.clock)

            if len(impulses_at_destination) > 0:
                # Each such impulse activates its target node
                for destination_node, impulses in impulses_at_destination:
                    for impulse in impulses:
                        self.activate_node(destination_node, impulse.arrival_activation)

    def tick(self):
        """Performs the spreading activation algorithm for one tick of the clock."""
        self.clock += 1
        self._propagate_impulses()

    def __str__(self):

        string_builder = f"CLOCK = {self.clock}\n"
        string_builder += "Nodes:\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._node_activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.node2label[node]}: {self.activation_of_node(node)}\n"

        string_builder += "Edges:\n"
        for n1, n2 in self.graph.edges():
            impulses_this_edge = self.impulses_by_edge(n1, n2)
            if len(impulses_this_edge) == 0:
                continue
            string_builder += f"\t{self.node2label[n1]}–{self.node2label[n2]}:\n"
            for impulse in impulses_this_edge:
                string_builder += f"\t\t{impulse}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]


def decay_function_exponential_with_decay_factor(decay_factor) -> callable:
    # Decay formula for activation a, original activation a_0, decay factor d, time t:
    #   a = a_0 d^t
    #
    # In traditional formulation of exponential decay, this is equivalent to:
    #   a = a_0 e^(-λt)
    # where λ is the decay constant.
    #
    # I.e.
    #   d = e^(-λ)
    #   λ = - ln d
    assert 0 < decay_factor <= 1

    def decay_function(age, original_activation):
        return original_activation * (decay_factor ** age)

    return decay_function


def decay_function_exponential_with_half_life(half_life) -> callable:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return decay_function_exponential_with_decay_factor(decay_factor)


def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
    """Gaussian decay with sd specifying the number of ticks."""
    assert height_coef > 0
    assert sd > 0

    def decay_function(age, original_activation):
        height = original_activation * height_coef
        return height * exp((-1) * (((age - centre) ** 2) / (2 * sd * sd)))

    return decay_function


def decay_function_gaussian_with_sd_fraction(sd_frac: float, granularity: int, height_coef=1, centre=0) -> callable:
    """Gaussian decay with sd as a fraction of the granularity."""
    sd = sd_frac * granularity
    return decay_function_gaussian_with_sd(
        sd=sd,
        height_coef=height_coef,
        centre=centre)


def graph_from_distance_matrix(distance_matrix: ndarray,
                               length_granularity: int,
                               weighted_graph: bool,
                               weight_factor: float = 1,
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
    If False, all edges get the same weight.
    :param weight_factor:
    (Default 1.)
    If `weighted_graph` is True, this factor is multiplied by all weights.
    If `weighted_graph` is false, this fixed weight given to each edge in the graph.
    :param prune_connections_longer_than:
    (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be severed.
    :return:
    A Graph of the correct format.
    """

    length_matrix = ceil(distance_matrix * length_granularity)

    if weighted_graph:
        weight_matrix = weight_factor * (ones_like(distance_matrix) - distance_matrix)
    else:
        weight_matrix = weight_factor * ones_like(distance_matrix)

    graph = from_numpy_matrix(weight_matrix)

    # Converting from a distance matrix creates self-loop edges, which we have to remove
    graph.remove_edges_from(selfloop_edges(graph))

    # Add lengths to graph data
    for n1, n2, e_data in graph.edges(data=True):
        e_data[EdgeDataKey.LENGTH] = length_matrix[n1][n2]

    # Prune long connections
    if prune_connections_longer_than is not None:
        long_edges = [
            (n1, n2)
            for n1, n2, e_data in graph.edges(data=True)
            if e_data[EdgeDataKey.LENGTH] > prune_connections_longer_than
        ]
        graph.remove_edges_from(long_edges)

    return graph
