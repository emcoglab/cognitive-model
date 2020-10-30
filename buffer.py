"""
===========================
Working memory buffer and accessible set.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, FrozenSet, Iterable

from .ldm.utils.maths import clamp01
from .basic_types import ActivationValue, Size, SizedItem, Item
from .events import ItemLeftBufferEvent, ItemActivatedEvent, ModelEvent, ItemEnteredBufferEvent, BufferFloodEvent, \
    BufferEvent


class OverCapacityError(Exception):
    pass


class LimitedCapacityItemSet(ABC):
    def __init__(self,
                 threshold: ActivationValue,
                 capacity: Optional[Size],
                 enforce_capacity: bool,
                 items: FrozenSet[Item] = None):
        """

        :param threshold:
        :param capacity:
            If None, unlimited capacity.
        :param enforce_capacity:
            If True, will raise an OverCapacityError if .items is ever too large.
        :param items:
        :raises OverCapacityError
        """

        # Use >= and < to test for above/below
        self.threshold: ActivationValue = threshold

        self.capacity: Optional[Size] = capacity
        self._enforce_capacity: bool = enforce_capacity
        self.items: FrozenSet[Item] = frozenset() if items is None else items

        assert self.threshold >= 0
        # zero-size limit is degenerate: the set is always empty.
        assert (self.capacity is None) or (self.capacity > 0)

    @property
    def items(self) -> FrozenSet[Item]:
        # Since self.__items is a frozenset, we don't need to check for capacity after getting (in a finally block)
        # because the item isn't mutated, even by -=, it's just replaced.
        return self.__items

    @items.setter
    def items(self, value: FrozenSet[Item]):
        self.__items = value
        if self._enforce_capacity and self.over_capacity:
            # TODO: raise error, or just prune?
            raise OverCapacityError()

    @property
    def total_size(self) -> Size:
        return self._aggregate_size(self.items)

    @staticmethod
    def _aggregate_size(items: Iterable[Item]) -> Size:
        return Size(sum(
            (i.size if isinstance(i, SizedItem) else 1)
            for i in items
        ))

    @property
    def over_capacity(self) -> bool:
        if self.capacity is None:
            return False
        return self.total_size > self.capacity

    def clear(self):
        """Empties the set of items."""
        self.items = frozenset()

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.items

    @abstractmethod
    def prune_decayed_items(self,
                            activation_lookup: Callable[[Item], ActivationValue],
                            time: int):
        """
        Removes items from the distinguished set which have dropped below threshold.
        :param activation_lookup:
            Function mapping items to their current activation.
        :param time:
            The current time on the clock. Will be used in events.
        """
        raise NotImplementedError()

    @abstractmethod
    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: Callable[[Item], ActivationValue],
                      time: int):
        """
        Present a list of item activations to the set, and upgrades those which entered.
        :param activation_events:
            All activation events.
        :param activation_lookup:
            Function mapping items to their current activation.
            We need this as well as the activation_events because we need to know the activation of items currently in
            the buffer, and that is not stored anywhere (because it would be out of date).
        :param time:
            The current time on the clock. Will be used in events.
        """
        raise NotImplementedError()


class WorkingMemoryBuffer(LimitedCapacityItemSet):

    def __init__(self,
                 threshold: ActivationValue,
                 capacity: Optional[Size],
                 items: FrozenSet[SizedItem] = None):
        super().__init__(threshold=threshold,
                         capacity=capacity,
                         enforce_capacity=True,
                         items=items)

    def prune_decayed_items(self,
                            activation_lookup: Callable[[Item], ActivationValue],
                            time: int) -> List[ItemLeftBufferEvent]:
        """
        Removes items from the buffer which have dropped below threshold.
        :return:
            Events for items which left the buffer by decaying out.
        """
        new_buffer_items = frozenset(
            item
            for item in self.items
            if activation_lookup(item) >= self.threshold
        )
        decayed_out = self.items - new_buffer_items
        self.items = new_buffer_items
        return [
            ItemLeftBufferEvent(time=time, item=item)
            for item in decayed_out
        ]

    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: Callable[[Item], ActivationValue],
                      time: int) -> List[ModelEvent]:
        """
        Present a list of item activations to the buffer, and upgrades those which entered the buffer.
        :return:
            The same events, with some upgraded to buffer entry events.
            Plus events for items which left the buffer through displacement.
            May also return a BufferFlood event if that is detected.
        """

        if len(activation_events) == 0:
            return []

        # At this point self.working_memory_buffer is still the old buffer (after decayed items have been removed)
        presented_items = frozenset(e.item for e in activation_events)

        # Don't present items already in the buffer
        items_already_in_buffer = self.items & presented_items
        presented_items -= items_already_in_buffer

        # region New buffer items

        # First build a new buffer out of everything which *could* end up in the buffer, then cut out things which don't
        # belong there

        # We will sort items in the buffer based on various bits of data.
        # The new buffer is everything in the current working_memory_buffer...
        new_buffer_items: Dict[Item: WorkingMemoryBuffer._SortingData] = {
            item: WorkingMemoryBuffer._SortingData(activation=activation_lookup(item),
                                                   # These items already in the buffer were not presented
                                                   being_presented=False)
            for item in self.items
        }
        # ...plus everything above threshold.
        # We use a dictionary with .update() here to overwrite the activation of anything already in the buffer.
        new_buffer_items.update({
            event.item: WorkingMemoryBuffer._SortingData(activation=event.activation,
                                                         # We've already worked out whether items are potentially
                                                         # entering the buffer
                                                         being_presented=event.item in presented_items)
            for event in activation_events
            if event.activation >= self.threshold
        })

        # Convert to a list of key-value pairs, sorted by activation, descending.
        # We want the order to be by activation, but with ties broken by recency, i.e. items being presented to the
        # buffer precede those already in the buffer.  Because Python's sorting is stable, meaning if we sort by
        # recency first, and then by activation, we get what we want [0].
        #
        # So first we sort by recency (i.e. whether they were presented), descending
        # (i.e. .presented==1 comes before .presented==0)
        #
        #     [0]: https://wiki.python.org/moin/HowTo/Sorting#Sort_Stability_and_Complex_Sorts
        new_buffer_items = sorted(new_buffer_items.items(), key=lambda kv: kv[1].being_presented, reverse=True)
        # Then we sort by activation, descending (larger activation first)
        # Also new_buffer_items is now a list of kv pairs, not a dictionary, so we don't need to use .items()
        new_buffer_items = sorted(new_buffer_items, key=lambda kv: kv[1].activation, reverse=True)

        # Trim down to size if necessary
        if self.capacity is not None:
            while self._aggregate_size([i for i, _ in new_buffer_items]) > self.capacity:
                new_buffer_items.pop()

        # endregion

        new_buffer = frozenset(item for item, _ in new_buffer_items)

        # For returning additional BufferEvents
        displaced_items = self.items - new_buffer
        whole_buffer_replaced = displaced_items == self.items

        # Update buffer
        self.items = new_buffer
        fresh_items = self.items - items_already_in_buffer

        # Upgrade events
        upgraded_events = [
            # Upgrade only those events which newly entered the buffer
            (
                ItemEnteredBufferEvent(time=e.time, item=e.item, activation=e.activation, fired=e.fired)
                if e.item in fresh_items
                else e
            )
            for e in activation_events
        ]

        # Add extra events if necessary
        buffer_events: List[BufferEvent] = [ItemLeftBufferEvent(time=time, item=i) for i in displaced_items]
        if whole_buffer_replaced:
            buffer_events.append(BufferFloodEvent(time=time))

        return upgraded_events + buffer_events

    @dataclass
    class _SortingData:
        """
        For sorting items before entry to the buffer.
        """
        activation: ActivationValue
        being_presented: bool


class AccessibleSet(LimitedCapacityItemSet):

    def __init__(self,
                 threshold: ActivationValue,
                 capacity: Optional[int],
                 items: FrozenSet[Item] = None):
        super().__init__(
            threshold=threshold,
            capacity=capacity,
            enforce_capacity=False,
            items=items)

    # Don't care about sizes
    @staticmethod
    def _aggregate_size(items: Iterable[Item]) -> Size:
        return Size(len(items))

    @property
    def items(self) -> FrozenSet[Item]:
        return self.__items

    @items.setter
    def items(self, value):
        # Update memory pressure whenever we alter the accessible set
        self.__items = value
        if self.capacity is not None:
            self.__pressure = clamp01(len(self.__items) / self.capacity)
        else:
            self.__pressure = 0

    @property
    def pressure(self) -> float:
        """
        Bounded between 0 and 1 inclusive
        0 when accessible set is empty, 1 when full.
        0 when there is no bounded capacity.
        """
        return self.__pressure

    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: Callable[[Item], ActivationValue],
                      time: int):
        if len(activation_events) == 0:
            return

        # Unlike the buffer, we're not returning any events, and the size limit is not enforced.  So we don't need to be
        # so careful about confirming what's already in there and what's getting replaced, etc.

        self.items = self.items.union({
            e.item
            for e in activation_events
            if e.activation >= self.threshold
        })

    def prune_decayed_items(self,
                            activation_lookup: Callable[[Item], ActivationValue],
                            time: int):
        self.items -= {
            item
            for item in self.items
            if activation_lookup(item) < self.threshold
        }
