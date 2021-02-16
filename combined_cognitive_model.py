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
from pathlib import Path
from typing import List, Optional, Set, Dict, DefaultDict
from logging import getLogger

from numpy import lcm
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN as POS_NOUN, VERB as POS_VERB

from .ldm.corpus.tokenising import modified_word_tokenize
from .sensorimotor_norms.breng_translation.dictionary.dialect_dictionary import ameng_to_breng, breng_to_ameng
from .basic_types import ActivationValue, Component, Size, Item, SizedItem
from .buffer import WorkingMemoryBuffer
from .events import ItemActivatedEvent, ItemEvent, ModelEvent
from .linguistic_components import LinguisticComponent
from .sensorimotor_components import SensorimotorComponent
from .utils.dictionary import forget_keys_for_values_satisfying
from .utils.iterable import partition

_logger = getLogger(__name__)


class InterComponentMapping:

    _ignored_words = {
        # articles
        "a", "the",
        # prepositions
        "of",
        # other
        "and", "'s", "in", "on"
        # punctuation
        ".",
    }

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
                    # If we're unset here, there's nothing we can do
                    continue
                else:
                    # Nothing we can do
                    continue

        def find_linguistic_matches(sensorimotor_term: str) -> Set[str]:
            """For a given sensorimotor term, finds all linguistic terms it should match"""
            possible_targets = {
                # Cases where there are multiple possible linguistic options (e.g. judgement and judgment)
                linguistic_source
                for linguistic_source, sensorimotor_targets in linguistic_to_sensorimotor.items()
                if sensorimotor_term in sensorimotor_targets
            } | {
                # Translations of sensorimotor terms (e.g. zucchini to courgette) in case both exist.
                # In order to pick up examples like ANESTHETISE -> anaesthetise (where anesthetise isn't AmEng), we need
                # to find all co-sourced translations; i.e. target-dialogue words who share a source-dialogue
                # translation.
                cosourced_target  # e.g. anaesthetise
                for source in breng_to_ameng.translations_for(sensorimotor_term)  # e.g. anaesthetise
                for cosourced_target in ameng_to_breng.translations_for(source)  # e.g. anesthetize
            }
            # Don't forget to include the original term, if it's already a match
            if sensorimotor_term in linguistic_vocab:
                possible_targets.add(sensorimotor_term)
            # But we only want valid targets
            return possible_targets & linguistic_vocab

        # sensorimotor --> linguistic direction
        sensorimotor_to_linguistic: DefaultDict[str, Set[str]] = defaultdict(set)
        for sensorimotor_term in sensorimotor_vocab:

            # We map from single-word and multi-word sensorimotor terms separately
            if " " not in sensorimotor_term:
                # Single-word term

                possible_targets = find_linguistic_matches(sensorimotor_term)

                if len(possible_targets) == 0:
                    # No hits, so there's nothing more we can do
                    # _logger.warning(f"Failed to find map for {sensorimotor_term}")
                    continue
                elif len(possible_targets) == 1:
                    # Just one option, so we pick it
                    sensorimotor_to_linguistic[sensorimotor_term] = possible_targets
                else:
                    # Pick the best equivalence class of targets
                    sensorimotor_to_linguistic[sensorimotor_term] = set(ameng_to_breng.zipf_winners_among(possible_targets))

            else:
                # Multi-word term

                # First check for the same thing without spaces
                compound_sensorimotor_term = "".join(sensorimotor_term.split(" "))
                compound_matches = find_linguistic_matches(compound_sensorimotor_term)
                if len(compound_matches) >= 1:
                    # That worked! Use the one(s) we found
                    sensorimotor_to_linguistic[sensorimotor_term] = compound_matches
                    continue
                else:
                    # Didn't work, we'll fall through to the next strategy below
                    pass

                # Tokenise and ignore stopwords
                sensorimotor_term_tokens = {
                    token
                    for token in modified_word_tokenize(sensorimotor_term)
                    if token not in InterComponentMapping._ignored_words
                }

                matched_components: Set[str] = set()
                for token in sensorimotor_term_tokens:
                    # For each component word, apply above mapping logic
                    matches = find_linguistic_matches(token)
                    if len(matches) == 0:
                        continue
                    elif len(matches) == 1:
                        matched_components |= matches
                    else:
                        # Include the zipf-preferred term where there are multiple matches found
                        matched_components |= set(ameng_to_breng.zipf_winners_among(matches))

                if len(matched_components) == 0:
                    # No hits at all, so there's nothing more we can do
                    continue
                else:
                    sensorimotor_to_linguistic[sensorimotor_term] = matched_components

        # Add lemmatised forms to linguistic -> sensorimotor direction
        for linguistic_term in linguistic_vocab:
            if linguistic_term not in linguistic_to_sensorimotor:
                # No mapping already exists, so we try for a lemmatisation of the base form:
                lemma_noun = lemmatiser.lemmatize(linguistic_term, pos=POS_NOUN)
                lemma_verb = lemmatiser.lemmatize(linguistic_term, pos=POS_VERB)
                if lemma_noun in sensorimotor_vocab:
                    linguistic_to_sensorimotor[linguistic_term].add(lemma_noun)
                    continue
                elif lemma_verb in sensorimotor_vocab:
                    linguistic_to_sensorimotor[linguistic_term].add(lemma_verb)
                    continue
                else:
                    # If it hasn't worked this time, we can try a last-ditch attempt to translate it first
                    # First try a translation
                    breng_linguistic_terms = ameng_to_breng.best_translations_for(linguistic_term)
                    if len(breng_linguistic_terms) > 0:
                        # We've already tried this above, and it won't have worked,
                        # but we can try to lemmatise AFTER translation
                        for breng_linguistic_term in breng_linguistic_terms:
                            lemma_noun = lemmatiser.lemmatize(breng_linguistic_term, pos=POS_NOUN)
                            lemma_verb = lemmatiser.lemmatize(breng_linguistic_term, pos=POS_VERB)
                            if lemma_noun in sensorimotor_vocab:
                                linguistic_to_sensorimotor[linguistic_term].add(lemma_noun)
                                break
                            elif lemma_verb in sensorimotor_vocab:
                                linguistic_to_sensorimotor[linguistic_term].add(lemma_verb)
                                break
                    # If it's still unset here, there's nothing we can do
                    continue

        # No need to remember entries with an empty set
        forget_keys_for_values_satisfying(linguistic_to_sensorimotor, lambda _, targets: len(targets) == 0)
        forget_keys_for_values_satisfying(sensorimotor_to_linguistic, lambda _, targets: len(targets) == 0)

        # Almost every item will be mapped to itself, so we don't need to explicitly remember that. Let's save memory!
        forget_keys_for_values_satisfying(linguistic_to_sensorimotor, lambda source, targets: targets == {source})
        forget_keys_for_values_satisfying(sensorimotor_to_linguistic, lambda source, targets: targets == {source})

        # Freeze and set
        self.linguistic_to_sensorimotor: Dict[str, Set[str]] = dict(linguistic_to_sensorimotor)
        self.sensorimotor_to_linguistic: Dict[str, Set[str]] = dict(sensorimotor_to_linguistic)

    def save_to(self, directory: Path):
        _logger.info(f"Saving mapping to {directory}...")
        import yaml
        lts_filename = " mapping_linguistic_to_sensorimotor.yaml"
        stl_filename = " mapping_sensorimotor_to_linguistic.yaml"
        with Path(directory, lts_filename).open(mode="w", encoding="utf-8") as lts_file:
            yaml.dump(
                {
                    source: sorted(targets)
                    for source, targets in self.linguistic_to_sensorimotor.items()
                },
                lts_file)
        with Path(directory, stl_filename).open(mode="w", encoding="utf-8") as stl_file:
            yaml.dump(
                {
                    source: sorted(targets)
                    for source, targets in self.sensorimotor_to_linguistic.items()
                },
                stl_file)


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
        self.mapping = InterComponentMapping(
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
                if linguistic_label in self.mapping.linguistic_to_sensorimotor:
                    sensorimotor_targets = self.mapping.linguistic_to_sensorimotor[linguistic_label]
                # Otherwise just try the fallback of direct mapping
                elif linguistic_label in self.sensorimotor_component.available_labels:
                    sensorimotor_targets = {linguistic_label}
                # If the mapping is not possible, just forget it
                else:
                    continue
                for sensorimotor_target in sensorimotor_targets:
                    # All of the sensorimotor targets are now guaranteed to be in the sensorimotor component
                    self.sensorimotor_component.propagator.schedule_activation_of_item_with_label(
                        label=sensorimotor_target,
                        activation=event.activation * self._inter_component_attenuation / len(sensorimotor_targets),
                        arrival_time=event.time + self._lc_to_smc_delay)

        for event in smc_activation_events:
            # Only transmit to other component if it fired.
            if event.fired:
                # Use label lookup from source component
                sensorimotor_label = self.sensorimotor_component.propagator.idx2label[event.item.idx]
                # If there are mappings, use them
                if sensorimotor_label in self.mapping.sensorimotor_to_linguistic:
                    linguistic_targets = self.mapping.sensorimotor_to_linguistic[sensorimotor_label]
                # Otherwise just try the fallback of direct mapping
                elif sensorimotor_label in self.linguistic_component.available_labels:
                    linguistic_targets = {sensorimotor_label}
                # If the direct mapping is not possible, just forget it
                else:
                    continue
                for linguistic_target in linguistic_targets:
                    self.linguistic_component.propagator.schedule_activation_of_item_with_label(
                        label=linguistic_target,
                        activation=event.activation * self._inter_component_attenuation / len(linguistic_targets),
                        arrival_time=event.time + self._smc_to_lc_delay)

        return (
            decay_events
            + lc_activation_events + lc_other_events
            + smc_activation_events + smc_other_events
        )
