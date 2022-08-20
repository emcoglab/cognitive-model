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

from .basic_types import ActivationValue
from .components import ModelComponentWithAccessibleSet
from .guards import make_under_firing_threshold_guard_for, make_exceeds_firing_threshold_guard_for
from .modulations import make_apply_activation_cap_modulation_for
from .propagator_linguistic import LinguisticPropagator


class LinguisticComponent(ModelComponentWithAccessibleSet):
    """
    The linguistic component of the model.
    Uses an exponential decay on nodes and a gaussian decay on edges.
    """

    def __init__(self,
                 propagator: LinguisticPropagator,
                 activation_cap: Optional[ActivationValue],
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 firing_threshold: ActivationValue,
                 ):
        """
        :param firing_threshold:
            Firing threshold.
            A node will fire on receiving activation if its activation crosses this threshold.
        """

        super().__init__(propagator, accessible_set_threshold, accessible_set_capacity)
        assert isinstance(self.propagator, LinguisticPropagator)

        if activation_cap is not None:
            assert (activation_cap
                    # If activation_cap == accessible_set_threshold, items will only enter the accessible set when fully
                    # activated.
                    >= self.accessible_set.threshold)

        # Thresholds
        # Use >= and < to test for above/below
        self.firing_threshold: ActivationValue = firing_threshold

        self.propagator.presynaptic_guards.extend([
            # If this node is currently suprathreshold, it acts as activation sink.
            # It doesn't accumulate new activation and cannot fire.
            make_under_firing_threshold_guard_for(self.firing_threshold)
        ])
        # No pre-synaptic modulation
        if activation_cap is not None:
            self.propagator.postsynaptic_modulations.extend([
                # Cap on a node's total activation after receiving incoming activations
                make_apply_activation_cap_modulation_for(activation_cap)
            ])
        self.propagator.postsynaptic_guards.extend([
            # Activation must exceed a firing threshold to cause further propagation.
            make_exceeds_firing_threshold_guard_for(self.firing_threshold)
        ])

