"""
===========================
Working memory buffer.
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
from dataclasses import dataclass
from typing import Optional, Set, Dict, List

from model.basic_types import ActivationValue, ItemIdx
from model.events import ItemLeftBufferEvent


class WorkingMemoryBuffer:

    def __init__(self, threshold: ActivationValue, capacity: Optional[int], items: Set[ItemIdx] = None):
        # Use >= and < to test for above/below
        self.threshold: ActivationValue = threshold
        assert self.threshold >= 0
        self.capacity: Optional[int] = capacity
        self.items: Set[ItemIdx] = set() if items is None else items
        if self.capacity is not None:
            assert len(self.items) <= self.capacity

    def replace_contents(self, new_items: Set[ItemIdx]):
        """Replaces the items in the buffer with a new set of items."""
        assert len(new_items) <= self.capacity
        self.items = new_items

    def clear(self):
        """Empties the buffer."""
        self.replace_contents(set())

    def prune_decayed_items(self, activation_lookup: Dict[ItemIdx, ActivationValue], time: int) -> List[ItemLeftBufferEvent]:
        """
        Removes items from the buffer which have dropped below threshold.
        :return:
            Events for items which left the buffer by decaying out.
        """
        new_buffer_items = {
            item
            for item in self.items
            if activation_lookup[item] >= self.threshold
        }
        decayed_out = self.items - new_buffer_items
        self.replace_contents(new_buffer_items)
        return [
            ItemLeftBufferEvent(time=time, item=item)
            for item in decayed_out
        ]

    @dataclass
    class _SortingData:
        """
        For sorting items before entry to the buffer.
        """
        activation: ActivationValue
        being_presented: bool