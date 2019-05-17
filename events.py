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
    def __repr__(self) -> str: return f"{type(self).__name__}( time={self.time!r} )"


@dataclass
class ItemEvent(ModelEvent, metaclass=ABCMeta):
    """An event involving an item."""
    # The item being activated.
    item: ItemIdx
    def __repr__(self) -> str: return f"{type(self).__name__}( time={self.time!r}, item={self.item!r} )"


@dataclass
class BufferEvent(ItemEvent, metaclass=ABCMeta):
    """Events involving the working memory buffer."""
    pass


@dataclass
class ItemActivatedEvent(ItemEvent):
    """An item is activated."""
    activation: ActivationValue
    def __repr__(self) -> str: return f"{type(self).__name__}( time={self.time!r}, item={self.item!r}, activation={self.activation!r} )"


@dataclass
class ItemFiredEvent(ItemActivatedEvent):
    """An item is activated enough to repropagate its activation."""
    @classmethod
    def from_activation_event(cls, event: ItemActivatedEvent):
        """Convert from ItemActivatedEvent."""
        return cls(time=event.time, item=event.item, activation=event.activation)


@dataclass
# An item enters the buffer only if it fired.  It is impossible that an item enters a buffer without firing.
# Therefore ItemEnteredBufferEvent is a ItemFiredEvent.
class ItemEnteredBufferEvent(BufferEvent, ItemFiredEvent):
    """An item was activated and entered the working memory buffer."""
    pass


@dataclass
class BailoutEvent(ModelEvent):
    """Model running ended with a bailout"""
    concurrent_activations: int
    def __repr__(self) -> str: return f"{type(self).__name__}( time={self.time!r}, concurrent_activations={self.concurrent_activations!r} )"
