"""
===========================
Spreading activation classes
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
from typing import Dict

from model.graph import Graph, Edge

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class SpreadingActivationCleglowski(object):
    """
    Spreading activation "energize" algorithm.
    Original algorithm from Cleglowski, Coburn, Cuadrado (2003) "Semantic Search of Unstructured Data using Contextual
    Network Graphs". Whitepaper published for the National Institute for Technology and Liberal Education.
    Modified to use global constant decay factor rather than valency-based decay, and also to allow breadth-first
    n-steps termination rather than depth-first recursion.
    """

    def __init__(self,
                 graph: Graph,
                 decay_factor: float,
                 firing_threshold: float,
                 energy_cap: float = None):

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Parameters
        self.decay_factor = decay_factor
        self.firing_threshold = firing_threshold
        self.energy_cap = energy_cap

        # Data for each node

        # The current energy of each node
        self.energy: Dict = dict()
        # Energy received last cycle
        self.recently_received_energy: Dict = dict()
        # The energy each node is ready to receive on the next cycle
        self.pending_energy: Dict = dict()

        self.initialise()

    def initialise(self):
        for node in self.graph.nodes:
            self.energy[node] = 0
            self.pending_energy[node] = 0
            self.recently_received_energy[node] = 0

    def activate_node(self, n):
        """Fully activate a node."""
        self.pending_energy[n] = 1

    def spread_once(self, verbose: bool = False):
        """One iteration of the spreading loop."""

        # Each node gets its pending energy
        for node in self.graph.nodes:

            # Accumulate pending energy
            self.energy[node] += self.pending_energy[node]

            # Remember what was received last cycle
            self.recently_received_energy[node] = self.pending_energy[node]
            self.pending_energy[node] = 0

            # constrain at cap
            if self.energy_cap is not None and self.energy[node] > self.energy_cap:
                self.energy[node] = self.energy_cap

        # Spread new energy to other nodes on next cycle
        # TODO: this loop might be parallelisable??
        for node in self.graph.nodes:
            neighbourhood = self.graph.neighbourhood(node)

            # Each node only passes on the energy it recieved last cycle.
            #
            # In the original Cleglowski algorithm, this is scaled by the degree of the node rather than a global decay
            # factor. However since our graph is complete, they would amount to the same thing, and this is simpler.
            #
            # As an alternative, Ziegler & Lausen (2004; IEEE-EEE) scale the energy by the sum of the weights for this
            # node's incident edges.
            presynaptic_energy = self.recently_received_energy[node]
            self.recently_received_energy[node] = 0

            # In the original Cleglowski algorithm, the threshold is applied to the a presynaptic energy which has
            # already been scaled by the decay factor or node valency. Moving the decay into the energy transfer amounts
            # to multiplying the firing threshold by a constant, so doesn't make much difference (well, a few tens of
            # thousands of multiplications) and will be a bit simpler to debug).
            #
            #  /!\  HOWEVER THIS WOULD HAVE TO CHANGE IF USING VALENCY-BASED DECAY, AS THE DECAY FACTOR WILL NOT BE
            #       CONSTANT OVER ALL NODES, WHEREAS THE THRESHOLD IS.
            #       If this is the plan, best go back to the source to verify the algorithm.
            if presynaptic_energy > self.firing_threshold:
                if verbose:
                    logger.info(f"'{node}' is above threshold!")
                for neighbour in neighbourhood:
                    edge_data = self.graph.edge_data[Edge((node, neighbour))]
                    energy_transfer = presynaptic_energy * edge_data.weight * self.decay_factor
                    self.pending_energy[neighbour] += energy_transfer

    def spread_n_times(self, n):
        """N iterations of the spreading loop."""
        for i in range(n):
            self.spread_once()

    def print_graph(self):
        """Print all nodes in the graph, and their output."""
        # Sort by energy.
        for n, e in sorted(self.energy.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{n}, {self.energy[n]}")
