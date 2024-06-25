"""
===========================
Events associated with model function.
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

from abc import ABC
from dataclasses import dataclass
from typing import Set

from .basic_types import ActivationValue, Item


@dataclass
class ModelEvent(ABC):
    """An event associated with model activity."""
    # The time at which the event occurred.
    time: int
    # Events are always Truthy, so we can use None as the Falsy alternative.
    def __bool__(self) -> bool: return True


@dataclass
class ItemEvent(ModelEvent, ABC):
    """An event involving an item."""
    # The item being activated.
    item: Item


@dataclass
class BufferEvent(ModelEvent, ABC):
    """Events involving the working memory buffer."""
    pass


@dataclass
class SubstitutionEvent(BufferEvent):
    """Represents a substitution of an item in the buffer."""
    # New item entering the buffer as part of the substitution
    new_item: Item
    # Item being displaced as part of the substitution
    displaced_item: Item


@dataclass
class ItemActivatedEvent(ItemEvent):
    """An item is activated."""
    activation: ActivationValue
    fired:      bool


@dataclass
class ItemEnteredBufferEvent(BufferEvent, ItemActivatedEvent):
    """An item was activated and entered the working memory buffer."""
    pass


@dataclass
class ItemLeftBufferEvent(BufferEvent, ItemEvent):
    """An item left the buffer."""
    pass


@dataclass
class ItemDecayedOutEvent(ItemLeftBufferEvent):
    """An item left the buffer by decaying."""
    pass


@dataclass
class ItemDisplacedEvent(ItemLeftBufferEvent):
    """An item left the buffer by being displaced."""
    pass


@dataclass
class BufferFloodEvent(BufferEvent):
    """The buffer becomes full after having each member replaced."""
    pass


@dataclass
class BailoutEvent(ModelEvent):
    """Model running ended with a bailout"""
    concurrent_activations: int
