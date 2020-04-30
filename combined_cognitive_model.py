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

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import DefaultDict, List

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.events import ItemActivatedEvent, ModelEvent
from model.linguistic_components import LinguisticComponent
from model.sensorimotor_components import SensorimotorComponent


@dataclass
class ComponentEvent:
    event: ModelEvent
    component: ComponentEvent.Type

    class Type(Enum):
        Linguistic = auto()
        Sensorimotor = auto()


class InteractiveCombinedCognitiveModel:
    def __init__(self,
                 linguistic_component: LinguisticComponent,
                 sensorimotor_component: SensorimotorComponent,
                 lc_to_smc_delay: int,
                 smc_to_lc_delay: int):

        self.linguistic_component: LinguisticComponent = linguistic_component
        self.sensorimotor_component: SensorimotorComponent = sensorimotor_component
        self.lc_to_smc_delay: int = lc_to_smc_delay
        self.smc_to_lc_delay: int = smc_to_lc_delay

        # A clock-keyed dictionary of label-keyed dictionaries of _activation_records
        self.future_smc_activations: DefaultDict[int, DefaultDict[ItemIdx, ActivationValue]] = defaultdict(lambda: defaultdict(ActivationValue))

        self.event_log: List[ComponentEvent] = []

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

    def tick(self):

        # Advance each component
        # Increments clock
        lc_events = self.linguistic_component.tick()
        smc_events = self.sensorimotor_component.tick()

        self.event_log.extend([ComponentEvent(event=e, component=ComponentEvent.Type.Linguistic) for e in lc_events])
        self.event_log.extend([ComponentEvent(event=e, component=ComponentEvent.Type.Sensorimotor) for e in smc_events])

        lc_activation_events = [e for e in lc_events if isinstance(e, ItemActivatedEvent)]
        smc_activation_events = [e for e in smc_events if isinstance(e, ItemActivatedEvent)]

        # Set up future _activation_records from smc to lc
        for event in smc_activation_events:
            self.linguistic_component.propagator.schedule_activation_of_item_with_idx(
                event.item, event.activation,
                event.time + self.smc_to_lc_delay)

        for event in lc_activation_events:
            self.sensorimotor_component.propagator.schedule_activation_of_item_with_idx(
                event.item, event.activation,
                event.time + self.lc_to_smc_delay)
