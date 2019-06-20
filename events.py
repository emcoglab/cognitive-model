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
from abc import ABCMeta
from dataclasses import dataclass

from model.basic_types import ActivationValue, ItemIdx


@dataclass
class ModelEvent(metaclass=ABCMeta):
    """An event associated with model activity."""
    # The time at which the event occurred.
    time: int
    # Events are always Truthy, so we can use None as the Falsy alternative.
    def __bool__(self) -> bool: return True


@dataclass
class ItemEvent(ModelEvent, metaclass=ABCMeta):
    """An event involving an item."""
    # The item being activated.
    item: ItemIdx


@dataclass
class BufferEvent(ModelEvent, metaclass=ABCMeta):
    """Events involving the working memory buffer."""
    pass


@dataclass
class ItemActivatedEvent(ItemEvent):
    """An item is activated."""
    activation: ActivationValue
    fired:      bool


@dataclass
class ItemEnteredBufferEvent(BufferEvent, ItemActivatedEvent):
    """An item was activated and entered the working memory buffer."""
    @classmethod
    def from_activation_event(cls, event: ItemActivatedEvent):
        """Convert from ItemActivatedEvent."""
        return cls(time=event.time, item=event.item, activation=event.activation, fired=event.fired)


@dataclass
class ItemLeftBufferEvent(BufferEvent, ItemEvent):
    """An item left the buffer by decaying or being displaced."""
    pass


@dataclass
class BufferFloodEvent(BufferEvent):
    """The buffer becomes full after having each member replaced."""
    pass


@dataclass
class BailoutEvent(ModelEvent):
    """Model running ended with a bailout"""
    concurrent_activations: int
