"""
===========================
Base classes for model components
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
from typing import Set, List, Optional

from .basic_types import ActivationValue, ItemIdx, ItemLabel
from .buffer import AccessibleSet
from .events import ModelEvent, ItemActivatedEvent
from .propagator import GraphPropagator
from .utils.iterable import partition

FULL_ACTIVATION = ActivationValue(1.0)


class ModelComponent(ABC):

    def __init__(self, propagator: GraphPropagator):
        self.propagator: GraphPropagator = propagator

        # This won't change so we set it once
        self._available_labels: Set[ItemLabel] = set(w for i, w in self.propagator.idx2label.items())

    @property
    def available_labels(self) -> Set[ItemLabel]:
        """Labels of concepts in the model component."""
        return self._available_labels

    def reset(self):
        self.propagator.reset()

    def tick(self) -> List[ModelEvent]:
        """
        Evolve the model.
        If you would be tempted to override this, instead try to override self._pre_tick() and self._post_tick().
        """
        time_at_start_of_tick = self.propagator.clock
        pre_tick_events = self._pre_tick()
        propagator_events = self.propagator.tick()
        tick_events = self._post_tick(pre_tick_events=pre_tick_events,
                                      propagator_events=propagator_events,
                                      time_at_start_of_tick=time_at_start_of_tick)
        return tick_events

    @abstractmethod
    def _pre_tick(self) -> List[ModelEvent]:
        """
        Action to perform before the propagator tick().
        Any events produced here will NOT be directly returned as part of self.tick().
        Instead they will be passed to self._post_tick().
        They will be stamped with the propagator-clock time at the start of the tick().
        """
        raise NotImplementedError()

    @abstractmethod
    def _post_tick(self,
                   pre_tick_events: List[ModelEvent],
                   propagator_events: List[ModelEvent],
                   time_at_start_of_tick: int,
                   ) -> List[ModelEvent]:
        """
        Action to perform after the propagator tick().
        It is the responsibility of this function to return all events it is passed, though some may be modified.
        Any events which occur here will also be returned.
        The output of this function is exactly what is returned by self.tick().
        :param pre_tick_events:
            events which occurred as part of self._pre_tick().
        :param propagator_events:
            events which occurred as part of the propagator.tick().
        :param time_at_start_of_tick:
            When this function runs, propagator.clock will have advanced. In order to make sure all events which take
            place during self.tick() have the same timestamp, you can rely on `time_at_start_of_tick` to have the
            correct time.
        :return:
            all events which occurred during the self.tick().
        """
        raise NotImplementedError()


class ModelComponentWithAccessibleSet(ModelComponent, ABC):

    def __init__(self,
                 propagator: GraphPropagator,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 ):

        super().__init__(propagator)

        # The set of items which are "accessible to conscious awareness"
        self.accessible_set: AccessibleSet = AccessibleSet(threshold=accessible_set_threshold, capacity=accessible_set_capacity)

        self.propagator.presynaptic_modulations.extend([
            self._apply_memory_pressure,
        ])
        self.propagator.postsynaptic_guards.extend([
            # Only allow firing until the item is sufficiently activated to be in the accessible set.
            # Items are presented to the accessible set after tick(), so this will only apply if the item was already in
            # the accessible set at the start of this tick.
            self._not_in_accessible_set
        ])

    def _apply_memory_pressure(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # When AS is empty, MP is 0, and activation is unaffected.
        # When AS is full,  MP is 1, and activation is killed.
        return activation * (1 - self.accessible_set.pressure)

    def _not_in_accessible_set(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node will only fire if it's not in the accessible set
        return idx not in self.accessible_set

    def reset(self):
        super().reset()
        self.accessible_set.clear()

    def _pre_tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case accessible set membership is used to modulate or guard anything)
        self.accessible_set.prune_decayed_items(activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
                                                time=self.propagator.clock)
        return []

    def _post_tick(self,
                   pre_tick_events: List[ModelEvent],
                   propagator_events: List[ModelEvent],
                   time_at_start_of_tick: int,
                   ) -> List[ModelEvent]:
        # Activation and firing may be affected by the size of or membership to the accessible set and the buffer, but
        # nothing will ENTER it until later, and everything that will LEAVE this tick already has done so.
        activation_events, other_events = partition(propagator_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update accessible set
        self.accessible_set.present_items(
            activation_events=activation_events,
            activation_lookup=lambda item: self.propagator.activation_of_item_with_idx_at_time(item.idx, time=time_at_start_of_tick),
            time=time_at_start_of_tick)

        return pre_tick_events + activation_events + other_events
