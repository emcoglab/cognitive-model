"""
===========================
Basic type aliases.
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

from dataclasses import dataclass
from enum import Enum, auto

ActivationValue = float
Node = int
ItemIdx = Node
ItemLabel = str
Length = int

Size = int


class Component(Enum):
    linguistic = 1
    sensorimotor = 2

    # Doesn't actually matter what the order is, it just needs to be defined and consistent to make sorting lists of
    # items a stable process.
    # This does mean that changing the definition of this order has the potential to change the model output.
    def __lt__(self, other: Component):
        return self.value < other.value


@dataclass(eq=True, frozen=True)
class Item:
    idx: ItemIdx
    component: Component


@dataclass(eq=True, frozen=True)
class SizedItem(Item):
    size: Size
