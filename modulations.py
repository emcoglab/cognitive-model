"""
===========================
Modulations
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2022
---------------------------
"""

from typing import Dict, Callable

from framework.cognitive_model.basic_types import ActivationValue, ItemIdx

# Maps an item and its activation to a new, modulated activation
Modulation = Callable[[ItemIdx, ActivationValue], ActivationValue]


# Functions to create modulations

def apply_activation_cap_modulation_for(activation_cap: ActivationValue) -> Modulation:
    def modulation(idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """If accumulated activation is over the cap, apply the cap."""
        return activation if activation <= activation_cap else activation_cap
    return modulation


def attenuate_by_statistic_modulation_for(statistic: Dict[ItemIdx, float]) -> Modulation:
    def modulation(idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """Scale an item's activation by a statistic for the item."""
        return activation * statistic[idx]
    return modulation
