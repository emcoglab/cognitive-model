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

from numpy import lcm

from breng_ameng.dialect_dictionary import ameng_to_breng, breng_to_ameng
from .basic_types import ActivationValue, Component, Size, Item, SizedItem
from .buffer import WorkingMemoryBuffer
from .events import ItemActivatedEvent, ItemEvent, ModelEvent
from .linguistic_components import LinguisticComponent
from .sensorimotor_components import SensorimotorComponent
from .utils.exceptions import ItemNotFoundError
from .utils.iterable import partition


class InteractiveCombinedCognitiveModel:
    def __init__(self,
                 linguistic_component: LinguisticComponent,
                 sensorimotor_component: SensorimotorComponent,
                 lc_to_smc_delay: int,
                 smc_to_lc_delay: int,
                 inter_component_attenuation: float,
                 buffer_threshold: ActivationValue,
                 buffer_capacity_linguistic_items: Optional[int],
                 buffer_capacity_sensorimotor_items: Optional[int],
                 ):

        # Inter-component delays and dampening
        self._lc_to_smc_delay: int = lc_to_smc_delay
        self._smc_to_lc_delay: int = smc_to_lc_delay
        self._inter_component_attenuation: float = inter_component_attenuation

        assert (0 <= self._inter_component_attenuation <= 1)

        # Relative component item sizes in the shared buffer
        total_capacity: Size = Size(lcm(buffer_capacity_linguistic_items, buffer_capacity_sensorimotor_items))
        self._lc_item_size: Size = Size(total_capacity // buffer_capacity_linguistic_items)
        self._smc_item_size: Size = Size(total_capacity // buffer_capacity_sensorimotor_items)

        # Make sure things divide evenly
        assert total_capacity / buffer_capacity_linguistic_items == total_capacity // buffer_capacity_linguistic_items
        assert total_capacity / buffer_capacity_sensorimotor_items == total_capacity // buffer_capacity_sensorimotor_items

        self.buffer = WorkingMemoryBuffer(threshold=buffer_threshold, capacity=total_capacity)

        self.linguistic_component: LinguisticComponent = linguistic_component
        self.sensorimotor_component: SensorimotorComponent = sensorimotor_component

        assert (self.buffer.threshold >= self.sensorimotor_component.accessible_set.threshold >= 0)
        assert (self.buffer.threshold >= self.linguistic_component.accessible_set.threshold >= 0)

        # The shared buffer does not affect the activity within either component or between them.
        self.linguistic_component.propagator.postsynaptic_guards.extend([])
        self.sensorimotor_component.propagator.postsynaptic_guards.extend([])

    @property
    def clock(self) -> int:
        assert self.linguistic_component.propagator.clock == self.sensorimotor_component.propagator.clock
        return self.linguistic_component.propagator.clock

    def reset(self):
        self.linguistic_component.reset()
        self.sensorimotor_component.reset()
        self.buffer.clear()

    def _activation_of_item(self, item: Item) -> ActivationValue:
        if item.component == Component.sensorimotor:
            return self.sensorimotor_component.propagator.activation_of_item_with_idx(item.idx)
        elif item.component == Component.linguistic:
            return self.linguistic_component.propagator.activation_of_item_with_idx(item.idx)

    def _apply_item_sizes(self, events: List[ModelEvent]) -> None:
        """
        Converts Items in events to have SizedItems withe the appropriate size.
        :param events:
            List of events.
            Gets mutated.
        """
        for e in events:
            if isinstance(e, ItemEvent):
                if e.item.component == Component.linguistic:
                    e.item = SizedItem(idx=e.item.idx, component=e.item.component,
                                       size=self._lc_item_size)
                elif e.item.component == Component.sensorimotor:
                    e.item = SizedItem(idx=e.item.idx, component=e.item.component,
                                       size=self._smc_item_size)

    def tick(self):

        decay_events = self.buffer.prune_decayed_items(
            activation_lookup=self._activation_of_item,
            time=self.clock)

        # Advance each component
        # Increments clock
        lc_events = self.linguistic_component.tick()
        smc_events = self.sensorimotor_component.tick()

        self._apply_item_sizes(lc_events)
        self._apply_item_sizes(smc_events)

        lc_events = self.buffer.present_items(
            activation_events=[e for e in lc_events if isinstance(e, ItemActivatedEvent)],
            activation_lookup=self._activation_of_item,
            time=self.clock)
        smc_events = self.buffer.present_items(
            activation_events=[e for e in smc_events if isinstance(e, ItemActivatedEvent)],
            activation_lookup=self._activation_of_item,
            time=self.clock)

        lc_activation_events: List[ItemActivatedEvent]
        smc_activation_events: List[ItemActivatedEvent]
        lc_activation_events, lc_other_events = partition(lc_events, lambda e: isinstance(e, ItemActivatedEvent))
        smc_activation_events, smc_other_events = partition(smc_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Schedule inter-component activations
        for event in lc_activation_events:
            # Only transmit to other component if it fired.
            if event.fired:
                # Use label lookup from source component
                linguistic_label = self.linguistic_component.propagator.idx2label[event.item.idx]
                try:
                    self.sensorimotor_component.propagator.schedule_activation_of_item_with_label(
                        label=(
                            # Use the linguistic label if it's available
                            linguistic_label
                            if linguistic_label in self.sensorimotor_component.available_labels
                            # Otherwise try the Americanized version
                            else breng_to_ameng.best_translation_for(linguistic_label)
                        ),
                        activation=event.activation * self._inter_component_attenuation,
                        arrival_time=event.time + self._lc_to_smc_delay)
                except ItemNotFoundError:
                    # Linguistic item was not found in Sensorimotor component
                    pass
        for event in smc_activation_events:
            # Only transmit to other component if it fired.
            if event.fired:
                # Use label lookup from source component
                sensorimotor_label = self.sensorimotor_component.propagator.idx2label[event.item.idx]
                try:
                    self.linguistic_component.propagator.schedule_activation_of_item_with_label(
                        label=(
                            sensorimotor_label
                            # Use the linguistic label if it's available
                            if sensorimotor_label in self.linguistic_component.available_labels
                            # Otherwise try the Britishised version
                            else ameng_to_breng.best_translation_for(sensorimotor_label)
                        ),
                        activation=event.activation * self._inter_component_attenuation,
                        arrival_time=event.time + self._smc_to_lc_delay)
                except ItemNotFoundError:
                    # Sensorimotor item was not found in Linguistic component
                    pass

        return (
                decay_events
                + lc_activation_events + lc_other_events
                + smc_activation_events + smc_other_events
        )
