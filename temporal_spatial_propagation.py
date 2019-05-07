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

from typing import Dict, Set

from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, ActivationValue, ItemLabel, \
    ItemActivatedEvent, ItemIdx


class TemporalSpatialPropagation:

    def __init__(self,
                 underlying_graph: Graph,
                 point_labelling_dictionary: Dict,
                 buffer_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 node_decay_function: callable):
        """
        Right now, this is just a shim for a underlying TemporalSpreadingActivation

        :param underlying_graph:
            The underlying graph of neighbours within a fixed radius.
        :param point_labelling_dictionary:
        :param buffer_pruning_threshold:
        :param activation_cap:
            If None is supplied, no cap is used.
        :param node_decay_function:
            If None is supplied, a constant function is used by default (i.e. no decay).
        """

        self._tsa: TemporalSpreadingActivation = TemporalSpreadingActivation(
            graph=underlying_graph,
            item_labelling_dictionary=point_labelling_dictionary,
            node_decay_function=node_decay_function,
            # Once pruning has been done, we don't need to decay activation in edges, as target items should receive the
            # full activations of their source items at the time they were last activated.
            # The maximal sphere radius is achieved by the initial graph pruning.
            edge_decay_function=None,
            # Points can't reactivate as long as they are still in the buffer.
            # For now this just means that they have a non-zero activation.
            # ...in fact we use the impulse-pruning threshold instead of 0, as no node will ever receive activation less
            # than this, by definition, and using 0 causes problems with counting supra-threshold nodes.
            firing_threshold=buffer_pruning_threshold, impulse_pruning_threshold=buffer_pruning_threshold,
            activation_cap=activation_cap,
        )

    def activate_item_with_label(self, label: ItemLabel, activation: ActivationValue) -> bool:
        return self._tsa.activate_item_with_label(label, activation)

    def tick(self) -> Set[ItemActivatedEvent]:
        return self._tsa.tick()

    def items_in_buffer(self) -> Set[ItemIdx]:
        return self._tsa.suprathreshold_items()

    @property
    def label2idx(self) -> Dict:
        return self._tsa.label2idx

    @property
    def idx2label(self) -> Dict:
        return self._tsa.idx2label

    @property
    def clock(self) -> int:
        return self._tsa.clock
