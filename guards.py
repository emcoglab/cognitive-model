"""
===========================
Guards
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

from typing import Callable

from framework.cognitive_model.basic_types import ItemIdx, ActivationValue

# Maps an item and its activation to whether something is permitted to happen,
# i.e. whether the guard is passed
# (True => it is allowed to happen; False => it is not allowed to happen)
Guard = Callable[[ItemIdx, ActivationValue], bool]


# Actual guards

def just_no_guard(idx: ItemIdx, activation: ActivationValue) -> bool:
    """Slot this guard into place to deny whatever is being guarded against."""
    return False


# Functions to create guards


def under_firing_threshold_guard_for(firing_threshold: ActivationValue) -> Guard:
    def guard(idx: ItemIdx, activation: ActivationValue) -> bool:
        return activation < firing_threshold
    return guard


def exceeds_firing_threshold_guard_for(firing_threshold: ActivationValue) -> Guard:
    def guard(idx: ItemIdx, activation: ActivationValue) -> bool:
        return activation >= firing_threshold
    return guard