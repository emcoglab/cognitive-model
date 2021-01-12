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
from typing import List, Optional, Set, Dict, DefaultDict
from logging import getLogger

from numpy import lcm
from nltk.stem import WordNetLemmatizer

from .sensorimotor_norms.breng_translation.dictionary.dialect_dictionary import ameng_to_breng, breng_to_ameng
from .basic_types import ActivationValue, Component, Size, Item, SizedItem
from .buffer import WorkingMemoryBuffer
from .events import ItemActivatedEvent, ItemEvent, ModelEvent
from .linguistic_components import LinguisticComponent
from .sensorimotor_components import SensorimotorComponent
from .utils.exceptions import ItemNotFoundError
from .utils.iterable import partition

_logger = getLogger(__name__)


class InterComponentMapping:
    def __init__(self,
                 linguistic_vocab: Set[str],
                 sensorimotor_vocab: Set[str],
                 ignore_identity_mapping: bool = True,
                 ):

        _logger.info("Setting up inter-component mapping")

        lemmatiser = WordNetLemmatizer()

        # linguistic --> sensorimotor direction
        # This is the easier direction, as we only ever map one -> one
        linguistic_to_sensorimotor: DefaultDict[str, Set[str]] = defaultdict(set)
        for linguistic_term in linguistic_vocab:

            if linguistic_term in sensorimotor_vocab:
                # Term exists in both places: just map directly
                linguistic_to_sensorimotor[linguistic_term].add(linguistic_term)

            else:
                # Can't map directly
                # Attempt translation:
                breng_linguistic_terms = ameng_to_breng.best_translations_for(linguistic_term)
                if len(breng_linguistic_terms) > 0:
                    # If there is at least one option, pick the best one by BrEng preference.
                    for breng_linguistic_term in breng_linguistic_terms:
                        if breng_linguistic_term in sensorimotor_vocab:
                            linguistic_to_sensorimotor[linguistic_term].add(breng_linguistic_term)
                            break
                    # If unset, try to lemmatise
                    # TODO: 1. this really seems like it could be cleaned up
                    # TODO: 2. some tests would be nice, for purposes of cleaning it up without breaking anything...
                    #  how about the tests from the example??
                    if linguistic_term not in linguistic_to_sensorimotor:
                        for breng_linguistic_term in breng_linguistic_terms:
                            lemma = lemmatiser.lemmatize(breng_linguistic_term)
                            if lemma in sensorimotor_vocab:
                                linguistic_to_sensorimotor[linguistic_term].add(lemma)
                                break
                    # If we're unset here, there's nothing we can do
                    pass
                else:
                    # No translations, so as a last-ditch attempt we try for a lemmatisation:
                    lemma = lemmatiser.lemmatize(linguistic_term)
                    if lemma in sensorimotor_vocab:
                        linguistic_to_sensorimotor[linguistic_term].add(lemma)
                    else:
                        # There's nothing we can do
                        pass

        # sensorimotor --> linguistic direction
        sensorimotor_to_linguistic: DefaultDict[str, Set[str]] = defaultdict(set)
        for sensorimotor_term in sensorimotor_vocab:

            possible_targets = {
                # Cases where there are multiple possible linguistic options (e.g. judgement and judgment)
                linguistic_source
                for linguistic_source, sensorimotor_targets in linguistic_to_sensorimotor.items()
                if sensorimotor_term in sensorimotor_targets
            } | set(
                # Translations of sensorimotor terms (e.g. zucchini to courgette) in case both exist.
                # In order to pick up examples like ANESTHETISE -> anaesthetise (where anesthetise isn't AmEng), we need
                # to find all co-sourced translations; i.e. target-dialogue words who share a source-dialogue
                # translation.
                cosourced_target  # e.g. anaesthetise
                for source in breng_to_ameng.translations_for(sensorimotor_term)  # e.g. anaesthetise
                for cosourced_target in ameng_to_breng.translations_for(source)  # e.g. anesthetize

            )
            # But we only want valid targets
            possible_targets &= linguistic_vocab

            if len(possible_targets) == 0:
                # No hits, so there's nothing more we can do
                _logger.warning(f"Failed to find map for {sensorimotor_term}")
                continue
            elif len(possible_targets) == 1:
                # Just one option, so we pick it
                sensorimotor_to_linguistic[sensorimotor_term] = possible_targets
            else:
                # Pick the best equivalence class of targets
                sensorimotor_to_linguistic[sensorimotor_term] = set(ameng_to_breng.zipf_winners_among(possible_targets))

        # No need to remember entries with an empty set
        linguistic_null = [
            s
            for s, ts in linguistic_to_sensorimotor.items()
            if len(ts) == 0
        ]
        sensorimotor_null = [
            s
            for s, ts in sensorimotor_to_linguistic.items()
            if len(ts) == 0
        ]
        for n in linguistic_null:
            del linguistic_to_sensorimotor[n]
        for n in sensorimotor_null:
            del sensorimotor_to_linguistic[n]
        # Almost every item will be mapped to itself, so we don't need to explicitly remember that. Let's save memory!
        if ignore_identity_mapping:
            linguistic_identity = [
                s
                for s, ts in linguistic_to_sensorimotor.items()
                if ts == {s}
            ]
            sensorimotor_identity = [
                s
                for s, ts in sensorimotor_to_linguistic.items()
                if ts == {s}
            ]
            for i in linguistic_identity:
                del linguistic_to_sensorimotor[i]
            for i in sensorimotor_identity:
                del sensorimotor_to_linguistic[i]

        # Freeze and set
        self.linguistic_to_sensorimotor: Dict[str, Set[str]] = dict(linguistic_to_sensorimotor)
        self.sensorimotor_to_linguistic: Dict[str, Set[str]] = dict(sensorimotor_to_linguistic)

    @staticmethod
    def linguistic_equivalents(linguistic_term: str) -> Set[str]:
        """Words which are equi-preferred to the supplied word."""
        # Check if it has any different translations
        translations = set(ameng_to_breng.best_translations_for(linguistic_term))
        if linguistic_term in translations:
            # Group of equivalents
            return translations
        else:
            # translations, if any, are not equivalent
            return {linguistic_term}
    # No one-many mappings in the other direction; each sensorimotor equivalence class is of size 1


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

        # Inter-component mapping logic
        self._mapping = InterComponentMapping(
            linguistic_vocab=self.linguistic_component.available_labels,
            sensorimotor_vocab=self.sensorimotor_component.available_labels,
        )

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
                # If there are mappings, use them
                if linguistic_label in self._mapping.linguistic_to_sensorimotor:
                    for sensorimotor_label in self._mapping.linguistic_to_sensorimotor[linguistic_label]:
                        self.sensorimotor_component.propagator.schedule_activation_of_item_with_label(
                            label=sensorimotor_label,
                            activation=event.activation * self._inter_component_attenuation,
                            arrival_time=event.time + self._lc_to_smc_delay)
                # Otherwise just try the fallback of direct mapping
                else:
                    try:
                        self.sensorimotor_component.propagator.schedule_activation_of_item_with_label(
                            label=linguistic_label,
                            activation=event.activation * self._inter_component_attenuation,
                            arrival_time=event.time + self._lc_to_smc_delay)
                    # If the direct mapping is not possible, just forget it
                    except ItemNotFoundError:
                        continue
        for event in smc_activation_events:
            # Only transmit to other component if it fired.
            if event.fired:
                # Use label lookup from source component
                sensorimotor_label = self.sensorimotor_component.propagator.idx2label[event.item.idx]
                # If there are mappings, use them
                if sensorimotor_label in self._mapping.sensorimotor_to_linguistic:
                    for linguistic_label in self._mapping.sensorimotor_to_linguistic[sensorimotor_label]:
                        self.linguistic_component.propagator.schedule_activation_of_item_with_label(
                            label=linguistic_label,
                            activation=event.activation * self._inter_component_attenuation,
                            arrival_time=event.time + self._smc_to_lc_delay)
                # Otherwise just try the fallback of direct mapping
                else:
                    try:
                        self.linguistic_component.propagator.schedule_activation_of_item_with_label(
                            label=sensorimotor_label,
                            activation=event.activation * self._inter_component_attenuation,
                            arrival_time=event.time + self._smc_to_lc_delay)
                    # If the direct mapping is not possible, just forget it
                    except ItemNotFoundError:
                        continue

        return (
            decay_events
            + lc_activation_events + lc_other_events
            + smc_activation_events + smc_other_events
        )
