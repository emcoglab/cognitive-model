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
2018
---------------------------
"""
from collections import defaultdict
from typing import Set, DefaultDict, List

from model.basic_types import ActivationValue, ItemIdx
from model.events import ItemActivatedEvent
from model.linguistic_component import LinguisticComponent
from model.sensorimotor_component import SensorimotorComponent


class TandemCognitiveModel:
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

        self.clock: int = int(0)

        # TODO: This.
        self.event_log: List = []

    def activate_lc_word(self, word: str, activation: float):
        self.linguistic_component.activate_item_with_label(word, activation)

    def activate_smc_concept(self, word: str, activation: float):
        self.sensorimotor_component.activate_item_with_label(word, activation)

    def tick(self):
        assert self.clock == self.linguistic_component.clock
        assert self.clock == self.sensorimotor_component.clock

        self.clock += 1
        lc_events = self.linguistic_component.tick()
        smc_events = self.sensorimotor_component.tick()

        lc_activation_events: Set[ItemActivatedEvent] = set(e for e in lc_events if isinstance(e, ItemActivatedEvent))
        smc_activation_events: Set[ItemActivatedEvent] = set(e for e in smc_events if isinstance(e, ItemActivatedEvent))

        # Set up future _activation_records from smc to lc
        for event in smc_activation_events:
            self.linguistic_component.schedule_activation_of_item_with_idx(
                event.item,
                event.activation,
                event.time + self.smc_to_lc_delay)

        # TODO: This should be refactored into model_component.schedule_activation(l, a, t) methods
        # Set up future _activation_records from lc to smc
        for event in lc_activation_events:
            self.future_smc_activations[event.time + self.smc_to_lc_delay][event.item] += event.activation

        # Apply predestined _activation_records
        if self.clock in self.future_smc_activations.keys():
            smc_activations = self.future_smc_activations.pop(self.clock)
            for idx, activation in smc_activations.items():
                self.sensorimotor_component.activate_item_with_idx(idx, activation)
