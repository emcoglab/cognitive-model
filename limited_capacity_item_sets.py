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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, FrozenSet, Tuple, Collection, Dict, Set

from .ldm.utils.maths import clamp01
from .basic_types import ActivationValue, Size, SizedItem, Item
from .events import ItemActivatedEvent, ItemDecayedOutEvent, BufferEvent, ItemDisplacedEvent, BufferFloodEvent, \
    ItemEnteredBufferEvent


class OverCapacityError(Exception):
    pass


@dataclass(eq=True, order=False)  # Sorting logic not stored with data
class ItemSortingData:
    """
    For sorting items before entry to the buffer.
    """
    activation: ActivationValue
    freshly_activated: bool
    tiebreaker: float


SortableItems = List[Tuple[Item, ItemSortingData]]
ItemActivationLookup = Callable[[Item], ActivationValue]
ItemValueLookup = Callable[[Item], float]
ItemListMutator = Callable[[SortableItems], None]


def strip_sorting_data(sortable_items: SortableItems) -> List[Item]:
    return [i for i, _ in sortable_items]


def kick_item_from_sortable_list(sortable_list: SortableItems, item_to_kick: Item) -> bool:
    """
    Removes all (Item, ItemSortingData) pairs from a SortableItems list where
    the Item matches the supplied one.

    If the item isn't matched, nothing happens.

    Mutates input list.

    Returns True if the item was actually kicked at least once, else False
    """
    item_was_kicked: bool = False
    while True:
        for item, sorting_data in sortable_list:
            if item == item_to_kick:
                pair_to_kick = (item, sorting_data)
                break
        else:
            # Made it through the whole list and the item is not there any more/ever
            return item_was_kicked
        sortable_list.remove(pair_to_kick)
        item_was_kicked = True


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
        # Since self.__items is a frozenset, we don't need to check for capacity
        # after getting (in a finally block) because the item isn't mutated,
        # even by -=, it's just replaced.
        return self.__items

    @items.setter
    def items(self, value: FrozenSet[Item]):
        self.__items = value
        if self._enforce_capacity and self._over_capacity:
            raise OverCapacityError()

    @property
    def _over_capacity(self) -> bool:
        return not self.items_would_fit(self.items)

    @classmethod
    @abstractmethod
    def aggregate_size(cls, items: Collection[Item]) -> Size:
        pass

    def items_would_fit(self, items: Collection[Item]) -> bool:
        """Returns True iff the list of items would fit within the buffer."""
        if self.capacity is None:
            return True
        return self.aggregate_size(items) <= self.capacity

    def truncate_items_list_to_fit(self, items: List[Item]) -> List[Item]:
        """
        Trims a list to fit within the buffer.
        
        Assumes that the list is sorted in descending order of importance for
        remaining in the set.
        
        Use for example on a list of candidate items to ensure that it will fit 
        within the set.
        """
        # One might think to have a simple trimming loop here like:
        #   while not self.items_would_fit(items: items.pop()
        # However it will often be the case that `items` is much longer than would fit, meaning that .items_would_fit()
        # will get called many times, which can be costly in terms of performance.
        # Therefore we do a single check to short-circuit...
        if self.items_would_fit(items):
            return items
        # ...and then instead count up through the items that will fit.
        # While this will be a little slower in cases where `items` is less than twice the size of `self.capacity`,
        # it will tend to be faster in most real-world cases.
        # If it turns out to be a problem in general, we could check for that threshold and then branch the logic to do
        # an upward or downward aggregation.
        # TODO: This could be optimised by just adding up the sizes with a counter and truncating there rather than
        #  repeatedly calculating the sizes
        items_to_return: List[Item] = []
        for item in items:
            if self.items_would_fit(items_to_return + [item]):
                items_to_return.append(item)
            else:
                break
        return items_to_return

    def clear(self):
        """Empties the set of items."""
        self.items = frozenset()

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.items

    @abstractmethod
    def prune_decayed_items(self,
                            activation_lookup: ItemActivationLookup,
                            time: int):
        """
        Removes items from the distinguished set that have dropped below threshold.
        :param activation_lookup:
            Function mapping items to their current activation.
        :param time:
            The current time on the clock. Will be used in events.
        """
        raise NotImplementedError()

    @abstractmethod
    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: ItemActivationLookup,
                      time: int):
        """
        Present a list of item activations to the set.
        :param activation_events:
            All activation events.
        :param activation_lookup:
            Function mapping items to their current activation.
            We need this as well as the activation_events because we need to know the activation of EXTANT items
            currently in the buffer, and that is not stored anywhere (because it would be out of date).
        :param time:
            The current time on the clock. Will be used in events.
        """
        raise NotImplementedError()


class WorkingMemoryBuffer(LimitedCapacityItemSet):

    def __init__(self,
                 threshold: ActivationValue,
                 capacity: Optional[Size],
                 items: FrozenSet[SizedItem] = None,
                 tiebreaker_lookup: Optional[ItemValueLookup] = None,
                 ):
        """
        :param tiebreaker_lookup:
            Optional function that gives items a value used to break ties for
            purposes of determining entry. Larger values result in increased
            precedence for entry.

            A value of None results in no additional precedence being applied
            when sorting.
        """
        super().__init__(threshold=threshold,
                         capacity=capacity,
                         enforce_capacity=True,
                         items=items)

        self._tiebreaker_lookup: ItemValueLookup = (
            tiebreaker_lookup
            if tiebreaker_lookup is not None
            # If None supplied, don't do any extra tie-breaking
            else lambda item: 0
        )

    @classmethod
    def aggregate_size(cls, items: Collection[SizedItem]) -> Size:
        return Size(sum(i.size for i in items))

    def prune_decayed_items(self,
                            activation_lookup: ItemActivationLookup,
                            time: int) -> List[ItemDecayedOutEvent]:
        """
        Removes items from the buffer that have dropped below threshold.
        :return:
            Events for items that left the buffer by decaying out.
        """
        new_buffer_items = frozenset(
            item
            for item in self.items
            if activation_lookup(item) >= self.threshold
        )
        decayed_out = self.items - new_buffer_items
        self.items = new_buffer_items
        return [
            ItemDecayedOutEvent(time=time, item=item)
            for item in decayed_out
        ]

    def present_items(self,
                      activation_events: List[ItemActivatedEvent],
                      activation_lookup: ItemActivationLookup,
                      time: int,
                      eligible_items_list_mutator: Optional[ItemListMutator] = None,
                      ) -> List[BufferEvent]:
        """
        Present a list of item activations to the buffer.

        :param eligible_items_list_mutator:
            Optional function to mutate the list of items just prior to trimming
            and committing.
            None => do nothing.
        :return:
            Events for items which left the buffer through displacement.
            May also return a BufferFlood event if that is detected.
        """

        # First build a new staged buffer out of everything which *could* end up
        # in the buffer, then cut out things which don't belong there.

        eligible_items_sortable = self._collect_eligible_items(
            from_new_activations=activation_events,
            activation_lookup=activation_lookup)
        eligible_items_sortable = self._sort_eligible_items(
            sortable_items=eligible_items_sortable)
        if eligible_items_list_mutator is not None:
            eligible_items_list_mutator(eligible_items_sortable)
        eligible_items = self.truncate_items_list_to_fit(strip_sorting_data(eligible_items_sortable))
        buffer_events = self._commit_buffer_items(
            items=eligible_items,
            time=time)
        return buffer_events

    def _collect_eligible_items(self,
                                from_new_activations: List[ItemActivatedEvent],
                                activation_lookup: ItemActivationLookup,
                                ) -> SortableItems:
        """
        Given a list of items activated this tick, returns a list of items which
        could be eligible for buffer entry.

        Returns a dict which maps items to some sorting data.
        """

        if len(from_new_activations) == 0:
            # If nothing's been activated this turn, we just sort the items by
            # their activations
            return [
                (item, ItemSortingData(
                    activation=activation_lookup(item),
                    freshly_activated=False,
                    tiebreaker=self._tiebreaker_lookup(item)))
                for item in self.items
            ]

        # We will sort items in the buffer based on various bits of data.
        # We build the list of potential new items in two steps.

        # First everything which is in the current buffer (decayed items have
        # already been removed at this point).
        potential_new_buffer_items: Dict[Item, ItemSortingData] = {
            item: ItemSortingData(
                activation=activation_lookup(item),
                # These items already in teh buffer were not presented
                freshly_activated=False,
                tiebreaker=self._tiebreaker_lookup(item))
            for item in self.items
        }

        # To this we add everything currently above threshold.
        # We use a dictionary with .update() here to overwrite the activation of
        # anything already in the buffer.
        potential_new_buffer_items.update({
            event.item: ItemSortingData(
                activation=event.activation,
                # We've already worked out whether items are potentially
                # entering the buffer
                freshly_activated=event.item not in self.items,
                tiebreaker=self._tiebreaker_lookup(event.item))
            for event in from_new_activations
            if event.activation >= self.threshold
        })

        return [(i, sd) for i, sd in potential_new_buffer_items.items()]

    def _commit_buffer_items(self,
                             items: List[Item],
                             time: int,
                             ) -> List[BufferEvent]:
        """
        Takes a set of eligible buffer items, such as that returned by
        ._collect_eligible_items(), and commits them to the buffer.

        Must be of the appropriate size.
        """

        new_buffer_items = frozenset(items)

        # For returning additional BufferEvents
        displaced_items = self.items - new_buffer_items
        whole_buffer_replaced = displaced_items == self.items

        # Update buffer
        self.items = new_buffer_items

        buffer_events: List[BufferEvent] = [ItemDisplacedEvent(time=time, item=i) for i in displaced_items]
        if whole_buffer_replaced:
            buffer_events.append(BufferFloodEvent(time=time))

        return buffer_events

    @classmethod
    def upgrade_events(cls,
                       old_items: Set[Item],
                       new_items: Set[Item],
                       activation_events: List[ItemActivatedEvent],
                       ) -> List[ItemActivatedEvent]:
        fresh_items = new_items - old_items
        upgraded_events = [
            # Upgrade only those which newly entered the buffer
            (
                ItemEnteredBufferEvent(time=e.time, item=e.item,
                                       activation=e.activation, fired=e.fired)
                if e.item in fresh_items  # TODO: this is currently quite fragile, as the items in the buffer and the items in the events may or not be SizedItems or Items, and x in y doesn't work with different types unless both types override __hash__
                else e
            )
            for e in activation_events
        ]
        return upgraded_events

    @classmethod
    def _sort_eligible_items(cls, sortable_items: SortableItems) -> SortableItems:
        # Convert to a list of key-value pairs, sorted by activation, descending.

        # We want a cascading set of tie-breakers in case of equal activation.
        # Python's sorting is stable, meaning we successively sort by the
        # cascading list of tie-breakers *in reverse order* to get what we want
        # [0].
        #
        # The order of sorting (in forward order) is:
        #
        #  1. Sort by activation, descending.
        #  2. In case of ties, sort by recency — i.e. items being presented to
        #     the buffer precede those already in the buffer, descending (i.e.
        #     .presented==1 comes before .presented==0).
        #  3. An externally provided tiebreaker lookup.
        #
        # In case we STILL have remaining ties (i.e. equal activation, both
        # being presented or not), ties are broken by existing orders of events:
        #
        #  4. Emission order.
        #
        #  5. After this, the order is *too difficult to determine*, as the list
        #     may be mutated using the linguistic placeholder and contingently
        #     re-sorted multiple times. If none of this happens, the order may
        #     be alphabetically by alphabetically-earliest edge endpoint. But
        #     being realistic, the order is not tractably determinable.
        #
        # [0]: https://wiki.python.org/moin/HowTo/Sorting#Sort_Stability_and_Complex_Sorts
        # [1]: Brysbaert et al., 2019.

        # So sort in reverse order of criteria:

        # Final tiebreaker first, descending
        sorted_buffer_items: SortableItems = sorted(sortable_items,
                                                    key=lambda i_s: i_s[1].tiebreaker,
                                                    reverse=True)
        # Then recency (i.e. whether they were presented for the first time just
        # now), descending
        sorted_buffer_items = sorted(sorted_buffer_items,
                                     key=lambda i_s: i_s[1].freshly_activated,
                                     reverse=True)
        # Then we sort by activation, descending (larger activation first)
        sorted_buffer_items = sorted(sorted_buffer_items,
                                     key=lambda i_s: i_s[1].activation,
                                     reverse=True)

        return sorted_buffer_items


class UnsizedWorkingMemoryBuffer(WorkingMemoryBuffer):
    @classmethod
    def aggregate_size(cls, items: Collection[Item]) -> Size:
        return Size(len(items))


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

    @classmethod
    def aggregate_size(cls, items: Collection[Item]) -> Size:
        """
        Here we don't care about sizes, so the combined size of a list of items
        is just the number of items.
        """
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
                      activation_lookup: ItemActivationLookup,
                      time: int):
        if len(activation_events) == 0:
            return

        # Unlike the buffer, we're not returning any events, and the size limit
        # is not enforced.  So we don't need to be so careful about confirming
        # what's already in there and what's getting replaced, etc.

        self.items = self.items.union({
            e.item
            for e in activation_events
            if e.activation >= self.threshold
        })

    def prune_decayed_items(self,
                            activation_lookup: ItemActivationLookup,
                            time: int):
        self.items -= {
            item
            for item in self.items
            if activation_lookup(item) < self.threshold
        }
