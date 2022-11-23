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
2020, 2022
---------------------------
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import List, Optional, Set, Dict, DefaultDict, Tuple, Sequence, Container
from logging import getLogger

from numpy import lcm, inf
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN as POS_NOUN, VERB as POS_VERB

from .basic_types import ActivationValue, Component, Size, Item, SizedItem, ItemLabel
from .components import ModelComponent
from .components_linguistic import LinguisticComponent
from .components_sensorimotor import SensorimotorComponent
from .events import ItemActivatedEvent, ItemEvent, ModelEvent, ItemDisplacedEvent, SubstitutionEvent
from .ldm.corpus.tokenising import modified_word_tokenize
from .limited_capacity_item_sets import SortableItems, kick_item_from_sortable_list, WorkingMemoryBuffer, \
    strip_sorting_data, replace_in_sortable_list
from .prevalence.brysbaert_prevalence import BrysbaertPrevalence
from .sensorimotor_norms.breng_translation.dictionary.dialect_dictionary import ameng_to_breng, breng_to_ameng
from .utils.maths import prevalence_from_fraction_known
from .utils.dictionary import forget_keys_for_values_satisfying
from .utils.iterable import partition
from .version import VERSION

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

    # Don't reference these externally, use InterComponentMapping.load_from(directory=) on the parent directory
    _lts_filename = " mapping_linguistic_to_sensorimotor.yaml"
    _stl_filename = " mapping_sensorimotor_to_linguistic.yaml"

    _Mapping = Dict[ItemLabel, Set[ItemLabel]]

    def __init__(self,
                 linguistic_to_sensorimotor: _Mapping,
                 sensorimotor_to_linguistic: _Mapping):
        self.linguistic_to_sensorimotor: InterComponentMapping._Mapping = linguistic_to_sensorimotor
        self.sensorimotor_to_linguistic: InterComponentMapping._Mapping = sensorimotor_to_linguistic

    # Alternative constructor
    @classmethod
    def from_vocabs(cls,
                    linguistic_vocab: Set[str],
                    sensorimotor_vocab: Set[str],
                    ignore_identity_mapping: bool = True,
                    ):

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
        if ignore_identity_mapping:
            forget_keys_for_values_satisfying(linguistic_to_sensorimotor, lambda source, targets: targets == {source})
            forget_keys_for_values_satisfying(sensorimotor_to_linguistic, lambda source, targets: targets == {source})

        # Freeze and set
        return InterComponentMapping(
            linguistic_to_sensorimotor=dict(linguistic_to_sensorimotor),
            sensorimotor_to_linguistic=dict(sensorimotor_to_linguistic),
        )

    def save_to(self, directory: Path):
        import yaml
        _logger.info(f"Saving mapping to {directory}...")
        with Path(directory, InterComponentMapping._lts_filename).open(mode="w", encoding="utf-8") as lts_file:
            yaml.dump(
                {
                    source: sorted(targets)
                    for source, targets in self.linguistic_to_sensorimotor.items()
                },
                lts_file,
                yaml.SafeDumper,
                indent=2,
            )
        with Path(directory, InterComponentMapping._stl_filename).open(mode="w", encoding="utf-8") as stl_file:
            yaml.dump(
                {
                    source: sorted(targets)
                    for source, targets in self.sensorimotor_to_linguistic.items()
                },
                stl_file,
                yaml.SafeDumper,
                indent=2,
            )

    @classmethod
    def load_from(cls, directory: Path) -> InterComponentMapping:
        """
        :raises: FileNotFoundError
        """
        import yaml
        _logger.info(f"Loading mapping from {directory.name}")
        with Path(directory, InterComponentMapping._lts_filename).open("r", encoding="utf-8") as lts_file:
            lts: cls._Mapping = yaml.load(lts_file, yaml.SafeLoader)
        with Path(directory, InterComponentMapping._stl_filename).open("r", encoding="utf-8") as stl_file:
            stl: cls._Mapping = yaml.load(stl_file, yaml.SafeLoader)
        return InterComponentMapping(
            linguistic_to_sensorimotor={
                k: set(v)
                for k, v in lts.items()
            },
            sensorimotor_to_linguistic={
                k: set(v)
                for k, v in stl.items()
            }
        )


class InteractiveCombinedCognitiveModel:

    version: str = VERSION

    def __init__(self,
                 linguistic_component: LinguisticComponent,
                 sensorimotor_component: SensorimotorComponent,
                 lc_to_smc_delay: int,
                 smc_to_lc_delay: int,
                 lc_to_smc_threshold: ActivationValue,
                 smc_to_lc_threshold: ActivationValue,
                 buffer_threshold: ActivationValue,
                 use_linguistic_placeholder: bool,
                 cross_component_attenuation: float,
                 buffer_capacity_linguistic_items: Optional[int],
                 buffer_capacity_sensorimotor_items: Optional[int],
                 with_mapping: Optional[InterComponentMapping] = None
                 ):
        """
        :param: with_mapping:
            Allows the mapping to be supplied (e.g. if it has been loaded).
            If missing or None, it will be recreated from scratch, which takes
            some time.
        """

        # Inter-component delays, thresholds and dampening
        self._lc_to_smc_delay: int = lc_to_smc_delay
        self._smc_to_lc_delay: int = smc_to_lc_delay
        self._lc_to_smc_threshold: ActivationValue = lc_to_smc_threshold
        self._smc_to_lc_threshold: ActivationValue = smc_to_lc_threshold
        self._cross_component_attenuation: float = cross_component_attenuation

        assert self._lc_to_smc_threshold >= 0
        assert self._smc_to_lc_threshold >= 0

        assert 0 <= self._cross_component_attenuation <= 1

        # Relative component item sizes in the shared buffer
        total_capacity: Size = Size(lcm(buffer_capacity_linguistic_items, buffer_capacity_sensorimotor_items))
        self._lc_item_size: Size = Size(total_capacity // buffer_capacity_linguistic_items)
        self._smc_item_size: Size = Size(total_capacity // buffer_capacity_sensorimotor_items)

        # Make sure things divide evenly
        assert total_capacity / buffer_capacity_linguistic_items == total_capacity // buffer_capacity_linguistic_items
        assert total_capacity / buffer_capacity_sensorimotor_items == total_capacity // buffer_capacity_sensorimotor_items

        self.linguistic_component: LinguisticComponent = linguistic_component
        self.sensorimotor_component: SensorimotorComponent = sensorimotor_component

        self._prevalence = BrysbaertPrevalence()

        self.buffer = WorkingMemoryBuffer(threshold=buffer_threshold,
                                          capacity=total_capacity,
                                          tiebreaker_lookup=self.__prevalence_lookup)

        assert self.buffer.threshold >= self.sensorimotor_component.accessible_set.threshold >= 0
        assert self.buffer.threshold >= self.linguistic_component.accessible_set.threshold >= 0

        self._use_linguistic_placeholder: bool = use_linguistic_placeholder

        # The shared buffer does not affect the activity within either component or between them.
        self.linguistic_component.propagator.firing_guards.extend([])
        self.sensorimotor_component.propagator.firing_guards.extend([])

        # Inter-component mapping logic
        _logger.info("Setting up inter-component mapping")
        # Note that this mapping is NOT assumed to contain identity mappings
        self.mapping: InterComponentMapping
        if with_mapping is not None:
            self.mapping = with_mapping
        else:
            self.mapping = InterComponentMapping.from_vocabs(
                linguistic_vocab=self.linguistic_component.available_labels,
                sensorimotor_vocab=self.sensorimotor_component.available_labels,
                ignore_identity_mapping=True,
            )

        # To prevent cat -> CAT producing an automatic CAT -> cat activation a few ticks down the line, we suppress that
        # here. When cat -> CAT is scheduled, we remember to ignore the activation coming back from CAT on the tick it
        # was scheduled for.
        # tick -> [items]
        self._suppress_linguistic_items_on_tick: DefaultDict[int, List[ItemLabel]] = defaultdict(list)
        self._suppress_sensorimotor_items_on_tick: DefaultDict[int, List[ItemLabel]] = defaultdict(list)

    @property
    def clock(self) -> int:
        try:
            assert self.linguistic_component.propagator.clock == self.sensorimotor_component.propagator.clock
        except AssertionError as e:
            e.args += (f"linguistic clock: {self.linguistic_component.propagator.clock}",
                       f"sensorimotor clock: {self.sensorimotor_component.propagator.clock}")
            raise
        return self.linguistic_component.propagator.clock

    def reset(self):
        self.linguistic_component.reset()
        self.sensorimotor_component.reset()
        self.buffer.clear()

    def _activation_of_item(self, item: Item) -> ActivationValue:
        return self._activation_of_item_at_time(item=item, time=self.clock)

    def _activation_of_item_at_time(self, item: Item, time: int) -> ActivationValue:
        if item.component == Component.sensorimotor:
            return self.sensorimotor_component.propagator.activation_of_item_with_idx_at_time(item.idx, time=time)
        elif item.component == Component.linguistic:
            return self.linguistic_component.propagator.activation_of_item_with_idx_at_time(item.idx, time=time)

    def label_lookup(self, item: Item) -> str:
        """:raises: KeyError when the item is not found"""
        mc: ModelComponent
        if item.component == Component.sensorimotor:
            mc = self.sensorimotor_component
        elif item.component == Component.linguistic:
            mc = self.linguistic_component
        else:
            raise NotImplementedError()
        return mc.propagator.idx2label[item.idx]

    def _apply_item_sizes_in_events(self, events: List[ModelEvent]) -> None:
        """
        Converts Items in events to have SizedItems with the appropriate size.

        Mutates the input list.
        """
        for e in events:
            if isinstance(e, ItemEvent):
                e.item = self._apply_item_size(e.item)

    def _apply_item_size(self, item: Item) -> SizedItem:
        """Converts items to their appropriate sizes."""
        if item.component == Component.linguistic:
            return SizedItem(idx=item.idx, component=item.component, size=self._lc_item_size)
        elif item.component == Component.sensorimotor:
            return SizedItem(idx=item.idx, component=item.component, size=self._smc_item_size)
        else:
            raise NotImplementedError()

    def tick(self):
        time_at_start_of_tick = self.clock

        pre_tick_events = self._pre_tick()

        # Advance each component
        # Increments clock
        lc_events = self.linguistic_component.tick()
        smc_events = self.sensorimotor_component.tick()

        tick_events = self._post_tick(pre_tick_events=pre_tick_events,
                                      model_events=lc_events + smc_events,
                                      time_at_start_of_tick=time_at_start_of_tick)

        return tick_events

    def _pre_tick(self) -> List[ModelEvent]:
        decay_events = self.buffer.prune_decayed_items(
            activation_lookup=self._activation_of_item,
            time=self.clock)
        return decay_events

    def _post_tick(self,
                   pre_tick_events: List[ModelEvent],
                   model_events: List[ModelEvent],
                   time_at_start_of_tick: int):

        activation_events, model_other_events = partition(model_events, lambda e: isinstance(e, ItemActivatedEvent))

        buffer_events = self._present_items_to_buffer(
            activation_events=activation_events,
            time_at_start_of_tick=time_at_start_of_tick)

        # Repartition events
        buffer_activation_events, buffer_other_events = partition(buffer_events, lambda e: isinstance(e, ItemActivatedEvent))

        self._handle_inter_component_activity(
            activation_events=buffer_activation_events,
            time_at_start_of_tick=time_at_start_of_tick)

        return (
            pre_tick_events
            + buffer_activation_events
            + model_other_events + buffer_other_events
        )

    def __get_linguistic_placeholders(self, sm_item: Item) -> Tuple[SizedItem | None, Set[SizedItem]]:
        """
        Given a sensorimotor item, returns the linguistic placeholder.

        There can be multiple linguistic placeholders (think multi-word terms), so
        returns first the preferred linguistic placeholder item and then a set of
        other items which also matched.

        Returns None in the first case where there is no substitution available.
        """

        sm_label: ItemLabel = self.sensorimotor_component.propagator.idx2label[sm_item.idx]

        # Check first to see if the sensorimotor item has a simple available counterpart in the linguistic component
        if sm_label in self.linguistic_component.available_labels:
            ling_item = self._apply_item_size(Item(idx=self.linguistic_component.propagator.label2idx[sm_label],
                                                   component=Component.linguistic))
            return ling_item, set()

        # If there's no direct counterpart, use the mapping
        try:
            ling_labels = self.mapping.sensorimotor_to_linguistic[sm_label]
        except KeyError:
            # No substitution available
            return None, set()
        ling_items: List[Item] = [
            Item(
                idx=self.linguistic_component.propagator.label2idx[ling_label],
                component=Component.linguistic)
            for ling_label in ling_labels
        ]

        # Return the most prevalent and then the rest
        ling_items.sort(key=lambda item: self.__prevalence_lookup(item), reverse=True)
        preferred_ling_item: SizedItem = self._apply_item_size(ling_items[0])
        other_ling_items: Set[SizedItem] = set(self._apply_item_size(i) for i in ling_items[1:])

        return preferred_ling_item, other_ling_items

    def _present_items_to_buffer(self,
                                 activation_events: List[ItemActivatedEvent],
                                 time_at_start_of_tick: int,
                                 ) -> List[ModelEvent]:
        """
        Present activation events to buffer and upgrade as necessary.
        """

        self._apply_item_sizes_in_events(activation_events)

        def activation_lookup(item: Item) -> ActivationValue:
            return self._activation_of_item_at_time(
                item=item, time=time_at_start_of_tick)

        previous_buffer = self.buffer.items

        if self._use_linguistic_placeholder:

            # We'll pass these into the mutator as a closure, so we get back the substitutions which were made therein

            # Dictionary of mapping substituted sensorimotor items to their linguistic placeholders
            substitutions: Dict[Item, Item] = dict()
            # Items which need to be activated as the result of substitutions
            # Stores the item together with the activation to give it
            placeholders_for_activation: Set[Tuple[Item, ActivationValue]] = set()
            # Items which need to be deactivated as the result of substitutions
            substituted_items_to_deactivate: Set[Item] = set()
            # Items which will be kicked from the buffer
            kicked_from_buffer: Set[Item] = set()

            def linguistic_placeholder_substitution_mutator(eligible_sortable_items: SortableItems) -> None:
                """
                Apply the linguistic-placeholder substitution to the list of items
                eligible for buffer entry.

                Preserves the order of items in the list (i.e. items are substituted in-place).

                ## The linguistic-placeholder substitution:

                When items are competing for entry to the buffer (i.e. now), if the
                buffer would become over-full, at that point we take every sensorimotor
                item *which would enter the buffer* and replace it (if possible) with its
                linguistic counterpart, to try and make more room for new items to enter.
                """

                # Items which should be substituted but for which no substitution is available
                no_substitutions_available: Set[Item] = set()

                # Recursively apply substitutions ot the least-activated item remaining in the buffer until as much
                # space is freed as required

                while not self.buffer.items_would_fit(strip_sorting_data(eligible_sortable_items)):

                    # Apply the substitution to the least-activated sensorimotor item that would end up the buffer
                    least_sm: Item = self.__get_least_sm_item(
                        provisional_buffer_items=self.buffer.truncate_items_list_to_fit(strip_sorting_data(eligible_sortable_items)),
                        ignoring=(
                            # We ignore all items where we know there are no substitutions to be made...
                            no_substitutions_available
                            # ...and substitutions we have already made
                            | substitutions.keys()))
                    if least_sm is None:
                        # No sensorimotor items left to substitute
                        break

                    ling_preferred_placeholder_for_buffer, other_ling_placeholders = self.__get_linguistic_placeholders(least_sm)
                    if ling_preferred_placeholder_for_buffer is None:
                        # No substitution available
                        no_substitutions_available.add(least_sm)
                        continue

                    # At this point a substitution will be made

                    activation_of_sensorimotor_item = activation_lookup(least_sm)
                    # All placeholders get activated by the appropriate amount
                    placeholders_for_activation.add((
                        # The blessed linguistic item will get the sensorimotor item's full activation
                        ling_preferred_placeholder_for_buffer,
                        activation_of_sensorimotor_item))
                    placeholders_for_activation.update({
                        (
                            ling_item,
                            # The other linguistic items get the sensorimotor item's activation distributed between them
                            activation_of_sensorimotor_item / len(other_ling_placeholders)
                        )
                        for ling_item in other_ling_placeholders
                    })

                    # Then we can deactivate the substituted item
                    substituted_items_to_deactivate.add(least_sm)

                    # region: Now do the actual mutation of the list.

                    # Make substitution
                    replace_in_sortable_list(eligible_sortable_items, item_out=least_sm, item_in=ling_preferred_placeholder_for_buffer)
                    # Record it
                    substitutions[least_sm] = ling_preferred_placeholder_for_buffer

                    # Substituted items not presented to the buffer also get kicked if they're already there
                    for item in other_ling_placeholders:
                        if kick_item_from_sortable_list(eligible_sortable_items, item_to_kick=item):
                            # Record the kicking
                            kicked_from_buffer.add(item)

                    # endregion

            buffer_events = self.buffer.present_items(
                activation_events=activation_events,
                activation_lookup=activation_lookup,
                time=time_at_start_of_tick,
                eligible_items_list_mutator=linguistic_placeholder_substitution_mutator,
            )

            # region: Substitution cleanup

            # Validate the closure-captured collections
            assert all(i.component == Component.sensorimotor for i in substituted_items_to_deactivate)
            assert all(i.component == Component.linguistic for i, _a in placeholders_for_activation)

            # New events for items which were kicked
            for item in kicked_from_buffer:
                buffer_events.append(ItemDisplacedEvent(
                    item=item,
                    time=time_at_start_of_tick))

            # Omit activation events relating to items which have just been
            # deactivated
            edited_activation_events = []
            for event in activation_events:
                if event.item in substituted_items_to_deactivate:
                    continue
                edited_activation_events.append(event)
            activation_events = edited_activation_events

            # Now activate and deactivated the items which were substituted
            for item in substituted_items_to_deactivate:
                self.__set_activation_of_item(item, ActivationValue(0), time_at_start_of_tick=time_at_start_of_tick)
            for item, activation in placeholders_for_activation:
                new_activation_events = self.__set_activation_of_item(item, activation, time_at_start_of_tick=time_at_start_of_tick)
                # New activation events for items which were activated
                activation_events.extend(new_activation_events)

            # Ensure that edited events also have sized items
            self._apply_item_sizes_in_events(activation_events)

            # Add the buffer substitution events
            for sm_item, ling_item in substitutions.items():
                buffer_events.append(SubstitutionEvent(new_item=ling_item, displaced_item=sm_item, time=time_at_start_of_tick))

            # endregion

        else:
            # Just present all items together
            buffer_events = self.buffer.present_items(
                activation_events=activation_events,
                activation_lookup=activation_lookup,
                time=time_at_start_of_tick,
            )

        activation_events = self.buffer.upgrade_events(
            old_items=set(previous_buffer), new_items=set(self.buffer.items),
            activation_events=activation_events)

        # noinspection PyTypeChecker
        return activation_events + buffer_events

    def __set_activation_of_item(self, item: Item, activation: ActivationValue, time_at_start_of_tick: int) -> List[ItemActivatedEvent]:
        from framework.cognitive_model.propagator import ActivationRecord  # TODO: move this logic into the propagators?
        component: ModelComponent
        if item.component == Component.sensorimotor:
            component = self.sensorimotor_component
        else:
            component = self.linguistic_component
        if activation == 0:
            # Just deactivate
            component.propagator._activation_records[item.idx] = ActivationRecord(activation=0, time_activated=time_at_start_of_tick)
            activation_events = []
        else:
            # Schedule and process activation
            component.propagator._schedule_activation_of_item_with_idx(item.idx, activation=activation, arrival_time=time_at_start_of_tick)
            # We will have advanced the individual propagator as part of this .tick(), so we do a hair-raising rollback
            # of the clock and re-propagate the newly added activations
            clock_should_be = component.propagator.clock
            component.propagator.clock = time_at_start_of_tick
            activation_events = component.propagator._evolve_propagator()
            component.propagator.clock = clock_should_be
        return activation_events

    def _handle_inter_component_activity(self,
                                         activation_events: List[ItemActivatedEvent],
                                         time_at_start_of_tick: int):

        linguistic_activation_events, sensorimotor_activation_events = partition(activation_events, lambda e: e.item.component == Component.linguistic)
        assert all(e.item.component == Component.sensorimotor for e in sensorimotor_activation_events)

        # Check for suppressed items
        try:
            suppressed_linguistic_items = self._suppress_linguistic_items_on_tick.pop(time_at_start_of_tick)
        except KeyError:
            # No suppressed linguistic items for this tick
            suppressed_linguistic_items = []
        try:
            suppressed_sensorimotor_items = self._suppress_sensorimotor_items_on_tick.pop(time_at_start_of_tick)
        except KeyError:
            # No suppressed sensorimotor items for this tick
            suppressed_sensorimotor_items = []

        self._schedule_inter_component_activations(
            source_component=self.linguistic_component,
            target_component=self.sensorimotor_component,
            source_component_activation_events=linguistic_activation_events,
            label_mapping=self.mapping.linguistic_to_sensorimotor,
            currently_suppressed_source_items=suppressed_linguistic_items,
            suppressed_target_items_dict=self._suppress_sensorimotor_items_on_tick,
            activation_threshold=self._lc_to_smc_threshold,
            cross_component_attenuation=self._cross_component_attenuation,
            delay=self._lc_to_smc_delay,
        )
        self._schedule_inter_component_activations(
            source_component=self.sensorimotor_component,
            target_component=self.linguistic_component,
            source_component_activation_events=sensorimotor_activation_events,
            label_mapping=self.mapping.sensorimotor_to_linguistic,
            currently_suppressed_source_items=suppressed_sensorimotor_items,
            suppressed_target_items_dict=self._suppress_linguistic_items_on_tick,
            activation_threshold=self._smc_to_lc_threshold,
            cross_component_attenuation=self._cross_component_attenuation,
            delay=self._smc_to_lc_delay,
        )

    @classmethod
    def _schedule_inter_component_activations(cls,
                                              source_component: ModelComponent,
                                              target_component: ModelComponent,
                                              source_component_activation_events: List[ItemActivatedEvent],
                                              activation_threshold: ActivationValue,
                                              cross_component_attenuation: float,
                                              label_mapping: Dict[ItemLabel, Set[ItemLabel]],
                                              currently_suppressed_source_items: List[ItemLabel],
                                              suppressed_target_items_dict: Dict[int, List[ItemLabel]],
                                              delay: int):
        """
        Mutates `suppressed_target_items_dict`.

        :param source_component:
        :param target_component:
        :param source_component_activation_events:
        :param activation_threshold:
            Only activations which would meet-or-exceed this on arrival will actually be sent.
        :param cross_component_attenuation:
            Linearly scales cross-component activations (i.e. 0 => squash, 1 => identity)
        :param label_mapping:
        :param currently_suppressed_source_items:
        :param suppressed_target_items_dict:
        :param delay:
        :return:
        """
        for event in source_component_activation_events:
            # Only transmit to other component if it fired.
            if not event.fired:
                continue
            # Use label lookup from source component
            source_label = source_component.propagator.idx2label[event.item.idx]
            # Check if it's currently suppressed
            if source_label in currently_suppressed_source_items:
                continue
            # If there are mappings, use them
            if source_label in label_mapping:
                targets = label_mapping[source_label]
            # Otherwise just try the fallback of direct mapping
            elif source_label in target_component.available_labels:
                targets = {source_label}
            # If the mapping is not possible, just forget it
            else:
                continue
            # All of the target labels are now guaranteed to be in the target component
            for target in targets:
                arrival_activation = event.activation
                if arrival_activation < activation_threshold:
                    continue
                if arrival_activation < source_component.propagator.impulse_pruning_threshold:
                    continue
                arrival_activation /= len(targets)  # Divide activation between components
                arrival_activation *= cross_component_attenuation
                arrival_time = event.time + delay

                target_component.propagator.schedule_activation_of_item_with_label(
                    label=target,
                    activation=arrival_activation,
                    arrival_time=arrival_time)
                # Remember to suppress the bounce-back, if any
                suppressed_target_items_dict[arrival_time].append(target)

    @classmethod
    def __get_least_sm_item(cls, provisional_buffer_items: Sequence[Item], *, ignoring: Container[Item]) -> Item | None:
        """
        Gets the least-activated sensorimotor item which would enter the buffer.

        Optional list of items to ignore (of we know we won't find a substitution
        for them.

        Assumes items are already sorted in descending order of precedence
        """
        item: Item
        # items are already sorted in reverse
        for item in reversed(provisional_buffer_items):
            if item in ignoring:
                continue
            if item.component == Component.linguistic:
                # Not interested in linguistic items
                continue
            return item
        # No items left to look at
        return None

    @cache
    def __prevalence_lookup(self, item: Item) -> float:
        try:
            if item.component == Component.sensorimotor:
                label = self.sensorimotor_component.propagator.idx2label[
                    item.idx]
                return prevalence_from_fraction_known(
                    self.sensorimotor_component.sensorimotor_norms.fraction_known(
                        label))
            elif item.component == Component.linguistic:
                label = self.linguistic_component.propagator.idx2label[item.idx]
                return self._prevalence.prevalence_for(label)
            else:
                raise NotImplementedError()
        except LookupError:
            # Item missing
            return -inf
