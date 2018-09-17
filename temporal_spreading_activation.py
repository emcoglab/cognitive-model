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
from typing import Set, Dict, DefaultDict, NamedTuple, Tuple

from numpy import exp, float_power
from scipy.stats import norm

from model.graph import Graph, Node

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class NodeActivationRecord(namedtuple('NodeActivationRecord', ['activation',
                                                               'time_activated'])):
    """
    NodeActivationRecord stores a historical node activation event.
    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.
    """
    # Tell Python no more fields can be added to this class, so it stays small in memory.
    __slots__ = ()


class ActivatedNodeEvent(NamedTuple):
    """
    A node activation event.
    Used to pass out of TSA.tick().
    Should be used for display and logging only, nothing high-performance!
    """
    node: str
    activation: float
    tick_activated: int

    def __repr__(self) -> str:
        return f"<'{self.node}' ({self.activation}) @ {self.tick_activated}>"


def blank_node_activation_record() -> NodeActivationRecord:
    """A record for an unactivated node."""
    return NodeActivationRecord(activation=0, time_activated=-1)


class TemporalSpreadingActivation(object):

    def __init__(self,
                 graph: Graph,
                 node_relabelling_dictionary: Dict,
                 firing_threshold: float,
                 conscious_access_threshold: float,
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
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        :param conscious_access_threshold:
            Conscious access threshold.
            A node will be listed as activated if its activation reaches this threshold.
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
        # Use >= and < to test for above/below
        self.firing_threshold: float = firing_threshold
        self.conscious_access_threshold: float = conscious_access_threshold
        self.impulse_pruning_threshold: float = impulse_pruning_threshold

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
        self.clock: int = int(0)

        # Graph data:

        # A node-keyed dictionaries of node activations.
        # Stores the most recent activation of each node, if any.
        self._node_activation_records: DefaultDict = defaultdict(blank_node_activation_record)

        # Impulses are stored in an arrival-time-keyed dict of destination-node-keyed dicts of cumulative activation
        # scheduled for arrival.
        # This way, when an arrival time is reached, we can .pop() a destination-node-keyed dict of impulses to process.
        # Nice!
        # ACTUALLY we'll use a defaultdict here, so we can quickly and easily add an impulse in the right place without
        # verbose checks.
        self._impulses: DefaultDict[int, DefaultDict[Node, float]] = defaultdict(
            # In case the aren't any impulses due to arrive at a particular time, we'll just find an empty dict
            lambda: defaultdict(
                # In case there aren't any impulses due to arrive at a particular node, we'll just find 0 activation,
                # which allows for handy use of +=
                float
            ))

    def n_suprathreshold_nodes(self) -> int:
        """
        The number of nodes which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return len([
            n
            for n in self.graph.nodes
            if self.activation_of_node(n) >= self.firing_threshold
        ])

    def impulses_headed_for(self, n) -> Dict[int, float]:
        """A time-keyed dict of cumulative activation due to arrive at a node."""
        return {
            t: activation_arriving_at_time_t[n]
            for t, activation_arriving_at_time_t in self._impulses.items()
            if n in activation_arriving_at_time_t.keys()
        }

    def activation_of_node(self, n) -> float:
        """Returns the current activation of a node."""
        assert n in self.graph.nodes

        activation_record: NodeActivationRecord = self._node_activation_records[n]
        return self.node_decay_function(
            self.clock - activation_record.time_activated,  # node age
            activation_record.activation)

    def activation_of_node_with_label(self, n) -> float:
        """Returns the current activation of a node."""
        return self.activation_of_node(self.label2node[n])

    def activate_node_with_label(self, n, activation: float) -> Tuple[bool, bool]:
        """
        Activate a node.
        :param n:
        :param activation:
        :return:
            A 2-tuple of bools: (
                Node did fire,
                Node's activation did cross conscious-access threshold
            )
        """
        return self.activate_node(self.label2node[n], activation)

    def activate_node(self, n, activation: float) -> Tuple[bool, bool]:
        """
        Activate a node.
        :param n:
        :param activation:
        :return:
            A 2-tuple of bools: (
                Node did fire,
                Node's activation did cross conscious-access threshold
            )
        """
        assert n in self.graph.nodes

        current_activation = self.activation_of_node(n)

        currently_below_conscious_access_threshold = current_activation < self.conscious_access_threshold

        # If this node is currently suprathreshold, it acts as a sink.
        # It doesn't accumulate new activation and cannot fire.
        if current_activation >= self.firing_threshold:
            return (
                # Node didn't fire
                False,
                # Node's activation didn't change so it definitely didn't cross the conscious-access threshold
                False
            )

        # Otherwise, we proceed with the activation:

        # Accumulate activation
        new_activation = current_activation + activation
        self._node_activation_records[n] = NodeActivationRecord(new_activation, self.clock)

        # Check if we reached the conscious-access threshold
        did_cross_conscious_access_threshold = currently_below_conscious_access_threshold and (new_activation > self.conscious_access_threshold)

        # Check if we reached the firing threshold.

        if new_activation < self.firing_threshold:
            # If not, we're done
            return (
                # Node didn't fire
                False,
                did_cross_conscious_access_threshold
            )

        else:
            # If so, Fire!

            # Fire and rebroadcast
            source_node = n

            # For each incident edge
            for edge in self.graph.incident_edges(source_node):

                n1, n2 = edge
                if source_node == n1:
                    target_node = n2
                elif source_node == n2:
                    target_node = n1
                else:
                    raise ValueError()

                length = self.graph.edge_lengths[edge]

                arrival_activation = self.edge_decay_function(length, new_activation)

                # Skip any impulses which will be too small on arrival
                if arrival_activation < self.impulse_pruning_threshold:
                    continue

                arrival_time = int(self.clock + length)

                # Accumulate activation at target node at time when it's due to arrive
                self._impulses[arrival_time][target_node] += arrival_activation

            return (
                # Node did fire
                True,
                did_cross_conscious_access_threshold and (new_activation > self.conscious_access_threshold)
            )

    def _propagate_impulses(self) -> Set:
        """
        Propagates impulses along connections.
        :return:
            Set of nodes which became active.
        """

        # "Propagation" happens by just incrementing the global clock.
        # But we have to check if any impulses have reached their destination.

        nodes_which_fired = set()
        nodes_which_crossed_conscious_access_threshold = set()

        if self.clock in self._impulses:

            # This should be a destination-node-keyed dict of activation ready to arrive
            activation_at_destination: DefaultDict = self._impulses.pop(self.clock)

            if len(activation_at_destination) > 0:
                # Each such impulse activates its target node
                for destination_node, activation in activation_at_destination.items():
                    node_did_fire, node_did_cross_conscious_access_threshold = self.activate_node(destination_node, activation)
                    if node_did_fire:
                        nodes_which_fired.add(destination_node)
                    if node_did_cross_conscious_access_threshold:
                        nodes_which_crossed_conscious_access_threshold.add(destination_node)

        return nodes_which_crossed_conscious_access_threshold

    def tick(self) -> Set[ActivatedNodeEvent]:
        """
        Performs the spreading activation algorithm for one tick of the clock.
        :return:
            Set of nodes which became active.
        """
        self.clock += 1
        nodes_which_became_active = self._propagate_impulses()

        return set(ActivatedNodeEvent(self.node2label[node], self.activation_of_node(node), self.clock) for node in nodes_which_became_active)

    def __str__(self):

        string_builder = f"CLOCK = {self.clock}\n"
        for node in self.graph.nodes:
            # Skip unactivated nodes
            if self._node_activation_records[node].time_activated == -1:
                continue
            string_builder += f"\t{self.node2label[node]}: {self.activation_of_node(node)}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]


def decay_function_exponential_with_decay_factor(decay_factor) -> callable:
    """
    Exponential decay function specified by decay factor.
    :param decay_factor:
    :return:
    """
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
    """
    Exponential decay function specified by half-life.
    :param half_life:
    :return:
    """
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return decay_function_exponential_with_decay_factor(decay_factor)


def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
    """
    Gaussian survival decay function with sd specifying the number of ticks.
    :param sd:
    :param height_coef:
    :param centre:
    :return:
    """
    assert height_coef > 0
    assert sd > 0

    def decay_function(age, original_activation):
        return original_activation * height_coef * norm.sf(age, loc=centre, scale=sd)

    return decay_function


def decay_function_gaussian_with_sd_fraction(sd_frac: float, granularity: int, height_coef=1, centre=0) -> callable:
    """
    Gaussian survival decay function with sd as a fraction of the granularity.
    :param sd_frac:
    :param granularity:
    :param height_coef:
    :param centre:
    :return:
    """
    sd = sd_frac * granularity
    return decay_function_gaussian_with_sd(
        sd=sd,
        height_coef=height_coef,
        centre=centre)
