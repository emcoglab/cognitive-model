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
UNIT_SIZE = Size(1)


@dataclass(eq=True, frozen=True)
class Item:
    idx: ItemIdx


@dataclass(eq=True, frozen=True)
class SizedItem(Item):
    component: SizedItem.Component
    size: Size

    class Component(Enum):
        linguistic = auto()
        sensorimotor = auto()
