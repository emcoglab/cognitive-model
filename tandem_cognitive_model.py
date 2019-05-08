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

from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.component import ActivationValue, ItemLabel, ItemActivatedEvent


class TandemCognitiveModel:
    def __init__(self,
                 linguistic_component: TemporalSpreadingActivation,
                 sensorimotor_component: TemporalSpatialPropagation,
                 lc_to_smc_delay: int,
                 smc_to_lc_delay: int):

        self.linguistic_component: TemporalSpreadingActivation = linguistic_component
        self.sensorimotor_component: TemporalSpatialPropagation = sensorimotor_component
        self.lc_to_smc_delay: int = lc_to_smc_delay
        self.smc_to_lc_delay: int = smc_to_lc_delay

        # A clock-keyed dictionary of label-keyed dictionaries of _activation_records
        self.future_smc_activations: DefaultDict[int, DefaultDict[ItemLabel, ActivationValue]] = defaultdict(lambda: defaultdict(ActivationValue))

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
        lc_activated_words: Set[ItemActivatedEvent] = self.linguistic_component.tick()
        smc_activated_concepts: Set[ItemActivatedEvent] = self.sensorimotor_component.tick()

        # Set up future _activation_records from smc to lc
        for concept in smc_activated_concepts:
            self.linguistic_component.schedule_activation_of_item_with_idx(
                self.linguistic_component.label2idx(concept.label),
                concept.activation,
                concept.time_activated + self.smc_to_lc_delay)

        # TODO: This should be refactored into model_component.schedule_activation(l, a, t) methods
        # Set up future _activation_records from lc to smc
        for word in lc_activated_words:
            self.future_smc_activations[word.time_activated + self.smc_to_lc_delay][word.label] += word.activation

        # Apply predestined _activation_records
        if self.clock in self.future_smc_activations.keys():
            smc_activations = self.future_smc_activations.pop(self.clock)
            for concept_label, activation in smc_activations.items():
                self.sensorimotor_component.activate_item_with_label(concept_label, activation)
