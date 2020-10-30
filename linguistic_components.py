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
from __future__ import annotations

from typing import Optional

from .basic_types import ActivationValue, ItemIdx
from .components import ModelComponentWithAccessibleSet
from .graph_propagator import Guard
from .linguistic_propagator import LinguisticPropagator


class LinguisticComponent(ModelComponentWithAccessibleSet):
    """
    The linguistic component of the model.
    Uses an exponential decay on nodes and a gaussian decay on edges.
    """

    def __init__(self,
                 propagator: LinguisticPropagator,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 firing_threshold: ActivationValue,
                 ):
        """
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        """

        # Thresholds
        # Use >= and < to test for above/below
        self.firing_threshold: ActivationValue = firing_threshold

        super().__init__(propagator, accessible_set_threshold, accessible_set_capacity)
        assert isinstance(self.propagator, LinguisticPropagator)

        self.propagator.presynaptic_guards.extend([
            # If this node is currently suprathreshold, it acts as activation sink.
            # It doesn't accumulate new activation and cannot fire.
            self._under_firing_threshold(self.firing_threshold)
        ])
        # No pre-synaptic modulation
        # No post-synaptic modulation
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
