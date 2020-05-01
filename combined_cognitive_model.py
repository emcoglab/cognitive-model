"""
===========================
A cognitive model with combined linguistic and sensorimotor components.
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

from typing import List, Optional

from model.basic_types import ActivationValue, ItemLabel, Size, Item, Component
from model.buffer import WorkingMemoryBuffer
from model.events import ItemActivatedEvent
from model.linguistic_components import LinguisticComponent
from model.sensorimotor_components import SensorimotorComponent
from model.utils.iterable import partition


class InteractiveCombinedCognitiveModel:
    def __init__(self,
                 linguistic_component: LinguisticComponent,
                 sensorimotor_component: SensorimotorComponent,
                 lc_to_smc_delay: int,
                 smc_to_lc_delay: int,
                 lc_item_size: Size,
                 smc_item_size: Size,
                 buffer_threshold: ActivationValue,
                 buffer_capacity: Optional[Size],
                 ):

        # Inter-component delays
        self._lc_to_smc_delay: int = lc_to_smc_delay
        self._smc_to_lc_delay: int = smc_to_lc_delay

        # Relative component item sizes in the shared buffer
        self._lc_item_size: Size = lc_item_size
        self._sm_item_size: Size = smc_item_size

        self.buffer = WorkingMemoryBuffer(threshold=buffer_threshold, capacity=buffer_capacity)

        self.linguistic_component: LinguisticComponent = linguistic_component
        self.sensorimotor_component: SensorimotorComponent = sensorimotor_component

        assert (self.buffer.threshold >= self.sensorimotor_component.accessible_set.threshold >= 0)
        assert (self.buffer.threshold >= self.linguistic_component.accessible_set.threshold >= 0)

    @property
    def clock(self) -> int:
        assert self.linguistic_component.propagator.clock == self.sensorimotor_component.propagator.clock
        return self.linguistic_component.propagator.clock

    def activate_item(self, label: ItemLabel, activation: ActivationValue):
        self.linguistic_component.propagator.activate_item_with_label(label, activation)
        self.sensorimotor_component.propagator.activate_item_with_label(label, activation)

    def activate_items(self, labels: List[ItemLabel], activation: ActivationValue):
        self.linguistic_component.propagator.activate_items_with_labels(labels, activation)
        self.sensorimotor_component.propagator.activate_items_with_labels(labels, activation)

    def reset(self):
        self.linguistic_component.reset()
        self.sensorimotor_component.reset()
        self.buffer.clear()

    def _activation_of_item(self, item: Item) -> ActivationValue:
        if item.component == Component.sensorimotor:
            return self.sensorimotor_component.propagator.activation_of_item_with_idx(item.idx)
        elif item.component == Component.linguistic:
            return self.linguistic_component.propagator.activation_of_item_with_idx(item.idx)

    def tick(self):

        decay_events = self.buffer.prune_decayed_items(
            activation_lookup=self._activation_of_item,
            time=self.clock)

        # Advance each component
        # Increments clock
        lc_tick_events = self.linguistic_component.tick()
        smc_tick_events = self.sensorimotor_component.tick()

        lc_buffer_events = self.buffer.present_items(
            activation_events=[e for e in lc_tick_events if isinstance(e, ItemActivatedEvent)],
            activation_lookup=self._activation_of_item,
            time=self.clock)
        smc_buffer_events = self.buffer.present_items(
            activation_events=[e for e in smc_tick_events if isinstance(e, ItemActivatedEvent)],
            activation_lookup=self._activation_of_item,
            time=self.clock)

        lc_activation_events, lc_other_events = partition(lc_buffer_events, lambda e: isinstance(e, ItemActivatedEvent))
        smc_activation_events, smc_other_events = partition(smc_buffer_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Schedule inter-component activations
        for event in lc_activation_events:
            self.sensorimotor_component.propagator.schedule_activation_of_item_with_idx(
                event.item.idx, event.activation,
                event.time + self._lc_to_smc_delay)
        for event in smc_activation_events:
            self.linguistic_component.propagator.schedule_activation_of_item_with_idx(
                event.item.idx, event.activation,
                event.time + self._smc_to_lc_delay)

        return (
                decay_events
                + lc_activation_events + smc_activation_events
                + lc_other_events + smc_other_events
        )
