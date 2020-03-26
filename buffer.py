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
from typing import Optional, Set, Dict, List, Callable

from ldm.utils.maths import clamp01
from model.basic_types import ActivationValue, ItemIdx
from model.events import ItemLeftBufferEvent, ItemActivatedEvent, ModelEvent, ItemEnteredBufferEvent, BufferFloodEvent


class LimitedCapacityItemSet(ABC):
    def __init__(self, threshold: ActivationValue, capacity: Optional[int], items: Set[ItemIdx] = None):
        # Use >= and < to test for above/below
        self.threshold: ActivationValue = threshold
        assert self.threshold >= 0

        self.capacity: Optional[int] = capacity
        self.items: Set[ItemIdx] = set() if items is None else items

        # zero-size limit is degenerate: the set is always empty.
        assert (self.capacity is None) or (self.capacity > 0)

        if self.capacity is not None:
            assert len(self.items) <= self.capacity

    def replace_contents(self, new_items: Set[ItemIdx]):
        """Replaces the items with a new set."""
        assert len(new_items) <= self.capacity
        self.items = new_items

    def clear(self):
        """Empties the set."""
        self.replace_contents(set())

    def __len__(self):
        return len(self.items)

    @abstractmethod
    def prune_decayed_items(self,
                            activation_lookup: Callable[[ItemIdx], ActivationValue],
                            time: int):
        """
        Removes items from the distinguished set which have dropped below threshold.
        :param activation_lookup:
            Function mapping items to their current activation.
        :param time:
            The current time on the clock. Will be used in events.
        """
        pass

    @abstractmethod
    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: Callable[[ItemIdx], ActivationValue],
                      time: int):
        """
        Present a list of item activations to the set, and upgrades those which entered.
        :param activation_events:
            All activation events.
        :param activation_lookup:
            Function mapping items to their current activation.
        :param time:
            The current time on the clock. Will be used in events.
        """
        pass


class WorkingMemoryBuffer(LimitedCapacityItemSet):

    def prune_decayed_items(self,
                            activation_lookup: Callable[[ItemIdx], ActivationValue],
                            time: int) -> List[ItemLeftBufferEvent]:
        """
        Removes items from the buffer which have dropped below threshold.
        :return:
            Events for items which left the buffer by decaying out.
        """
        new_buffer_items = {
            item
            for item in self.items
            if activation_lookup(item) >= self.threshold
        }
        decayed_out = self.items - new_buffer_items
        self.replace_contents(new_buffer_items)
        return [
            ItemLeftBufferEvent(time=time, item=item)
            for item in decayed_out
        ]

    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: Callable[[ItemIdx], ActivationValue],
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
        presented_items = set(e.item for e in activation_events)

        # Don't present items already in the buffer
        items_already_in_buffer = self.items & presented_items
        presented_items -= items_already_in_buffer

        # region New buffer items list of (item, activation)s

        # First build a new buffer out of everything which *could* end up in the buffer, then cut out things which don't
        # belong there

        # We will sort items in the buffer based on various bits of data.
        # The new buffer is everything in the current working_memory_buffer...
        new_buffer_items: Dict[ItemIdx: WorkingMemoryBuffer._SortingData] = {
            item: WorkingMemoryBuffer._SortingData(activation=activation_lookup(item),
                                                   # These items already in the buffer were not presented
                                                   being_presented=False)
            for item in self.items
        }
        # ...plus everything above threshold.
        # We use a dictionary with .update() here to overwrite the activation of anything already in the buffer.
        new_buffer_items.update({
            event.item: WorkingMemoryBuffer._SortingData(activation=event.activation,
                                                         # We've already worked out whether items are potentially entering the buffer
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
            new_buffer_items = new_buffer_items[:self.capacity]

        # endregion

        new_buffer = set(kv[0] for kv in new_buffer_items)

        # For returning additional BufferEvents
        whole_buffer_replaced = len(new_buffer - self.items) == self.capacity
        displaced_items = self.items - new_buffer

        # Update buffer: Get the keys (i.e. item idxs) from the sorted list
        self.replace_contents(new_buffer)

        # Upgrade events
        upgraded_events = [
            # Upgrade only those events which newly entered the buffer
            (
                ItemEnteredBufferEvent.from_activation_event(e)
                if e.item in self.items - items_already_in_buffer
                else e
            )
            for e in activation_events
        ]

        # Add extra events if necessary
        if whole_buffer_replaced:
            upgraded_events.append(BufferFloodEvent(time=time))
        upgraded_events.extend([ItemLeftBufferEvent(time=time, item=i) for i in displaced_items])

        return upgraded_events

    @dataclass
    class _SortingData:
        """
        For sorting items before entry to the buffer.
        """
        activation: ActivationValue
        being_presented: bool


class AccessibleSet(LimitedCapacityItemSet):

    @property
    def items(self):
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
                      activation_lookup: Callable[[ItemIdx], ActivationValue],
                      time: int):
        if len(activation_events) == 0:
            return

        # unlike the buffer, we're not returning any events, and there is no size limit, so we don't need to be so
        # careful about confirming what's already in there and what's getting replaced, etc.

        self.items |= {
            e.item
            for e in activation_events
            if e.activation >= self.threshold
        }

    def prune_decayed_items(self,
                            activation_lookup: Callable[[ItemIdx], ActivationValue],
                            time: int):
        self.items -= {
            item
            for item in self.items
            if activation_lookup(item) < self.threshold
        }
