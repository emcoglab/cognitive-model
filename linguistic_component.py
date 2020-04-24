"""
===========================
The linguistic component of the model.
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

from typing import Set, Dict

from model.basic_types import ActivationValue, ItemIdx
from model.components import ModelComponent
from model.graph_propagator import Guard
from model.linguistic_propagator import LinguisticPropagator


class LinguisticComponent(ModelComponent):
    """
    The linguistic component of the model.
    Uses an exponential decay on nodes and a gaussian decay on edges.
    """

    def __init__(self,
                 propagator: LinguisticPropagator,
                 activation_cap: ActivationValue,
                 firing_threshold: ActivationValue,
                 ):
        """
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        """

        # Thresholds
        # Use >= and < to test for above/below
        assert (activation_cap >= firing_threshold)
        self.activation_cap: ActivationValue = activation_cap
        self.firing_threshold: ActivationValue = firing_threshold

        super().__init__(propagator)
        assert isinstance(self.propagator, LinguisticPropagator)

        self.propagator.presynaptic_guards.extend([
            # If this node is currently suprathreshold, it acts as activation sink.
            # It doesn't accumulate new activation and cannot fire.
            self._under_firing_threshold(self.firing_threshold)
        ])
        # No pre-synaptic modulation
        self.propagator.postsynaptic_modulations.extend([
            # Cap on a node's total activation after receiving incoming
            self._apply_activation_cap(activation_cap)
        ])
        self.propagator.postsynaptic_guards.extend([
            # Activation must exceed a firing threshold to cause further propagation.
            self._exceeds_firing_threshold(self.firing_threshold)
        ])

    @staticmethod
    def _exceeds_firing_threshold(firing_threshold: ActivationValue) -> Guard:
        def guard(idx: ItemIdx, activation: ActivationValue) -> bool:
            return activation >= firing_threshold
        return guard

    @staticmethod
    def _under_firing_threshold(firing_threshold: ActivationValue) -> Guard:
        def guard(idx: ItemIdx, activation: ActivationValue) -> bool:
            return activation < firing_threshold
        return guard

    def suprathreshold_items(self) -> Set[ItemIdx]:
        """
        Items which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return set(
            n
            for n in self.propagator.graph.nodes
            if self.propagator.activation_of_item_with_idx(n) >= self.firing_threshold
        )
