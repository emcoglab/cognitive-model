"""
===========================
The sensorimotor component of the model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from os import path
from typing import Set, List, Dict

from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, ItemIdx, ItemLabel, Node
from model.events import ModelEvent, ItemActivatedEvent, ItemEnteredBufferEvent, BufferFloodEvent
from model.graph import Graph
from model.graph_propagation import _load_labels
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.iterable import partition
from model.utils.maths import make_decay_function_lognormal, prevalence_from_fraction_known, scale01
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class NormAttenuationStatistic(Enum):
    """The statistic to use for attenuating activation of norms labels."""
    FractionKnown = auto()
    Prevalence = auto()

    @property
    def name(self) -> str:
        """The name of the NormAttenuationStatistic"""
        if self is NormAttenuationStatistic.FractionKnown:
            return "Fraction known"
        if self is NormAttenuationStatistic.Prevalence:
            return "Prevalence"
        else:
            raise NotImplementedError()


@dataclass
class _BufferSortingData:
    """
    For sorting items before entry to the buffer.
    """
    activation: ActivationValue
    presented: bool


class SensorimotorComponent(TemporalSpatialPropagation):
    """
    The sensorimotor component of the model.
    Uses a lognormal decay on nodes.
    """

    # region __init__

    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 max_sphere_radius: int,
                 lognormal_median: float,
                 lognormal_sigma: float,
                 buffer_size_limit: int,
                 buffer_threshold: ActivationValue,
                 activation_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 norm_attenuation_statistic: NormAttenuationStatistic,
                 use_prepruned: bool = False,
                 ):
        """
        :param distance_type:
            The metric used to determine distances between points.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param max_sphere_radius:
            What is the maximum radius of a sphere
        :param lognormal_median:
            The median of the lognormal decay.
        :param lognormal_sigma:
            The sigma parameter for the lognormal decay.
        :param buffer_size_limit:
            The maximum size of the buffer. After this, qualifying items will displace existing items rather than just
            being added.
        :param buffer_threshold:
            The minimum activation required for a concept to enter the working_memory_buffer.
        :param activation_threshold:
            Used to determine what counts as "activated" and in the accessible set.
        :param activation_cap:
            If None is supplied, no cap is used.
        :param use_prepruned:
            Whether to use the prepruned graphs or do pruning on load.
            Only to be used for testing purposes.
        """

        # region Validation

        # max_sphere_radius == 0 would be degenerate: no item can ever activate any other item.
        assert (max_sphere_radius > 0)
        # lognormal_sigma or lognormal_median == 0 will probably cause a division-by-zero error, and anyway causes
        # everything to decay to 0 activation in a single tick
        assert (lognormal_median > 0)
        assert (lognormal_sigma > 0)
        # zero-size buffer size limit is degenerate: the buffer is always empty
        assert (buffer_size_limit > 0)
        assert (activation_cap
                # If activation_cap == buffer_threshold, items will only enter the buffer when fully activated.
                >= buffer_threshold
                # If buffer_pruning_threshold == activation_threshold then the only things in the accessible set with be
                # those items which were displaced from the buffer before being pruned. We probably won't use this but
                # it's not invalid or degenerate.
                >= activation_threshold
                # activation_threshold must be strictly positive, else no item can ever be reactivated (since membership
                # to the accessible set is a guard to reactivation).
                > 0)

        # endregion

        # Load graph
        idx2label = load_labels_from_sensorimotor()
        super(SensorimotorComponent, self).__init__(

            underlying_graph=_load_graph(distance_type, length_factor, max_sphere_radius,
                                         use_prepruned, idx2label),
            idx2label=idx2label,
            # Sigma for the log-normal decay gets multiplied by the length factor, so that if we change the length
            # factor, sigma doesn't also  have to change for the behaviour of the model to be approximately equivalent.
            node_decay_function=make_decay_function_lognormal(median=length_factor * lognormal_median, sigma=lognormal_sigma),
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Thresholds

        # Use >= and < to test for above/below
        self.buffer_threshold: ActivationValue = buffer_threshold
        self.activation_threshold: ActivationValue = activation_threshold
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        # Data

        self.buffer_size_limit = buffer_size_limit

        # A local copy of the sensorimotor norms data
        self._sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()

        self.norm_attenuation_statistic: NormAttenuationStatistic = norm_attenuation_statistic
        self._attenuation_statistic: Dict[ItemIdx, float] = {
            idx: self.__get_statistic_for_item(idx)
            for idx in self.graph.nodes
        }

        # endregion

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are currently being consciously considered.
        #
        # A fixed size (self.buffer_size_limit).  Items may enter the buffer when they are activated and leave when they
        # decay sufficiently (self.buffer_pruning_threshold) or are displaced.
        #
        # This is updated each .tick() based on items which fired (a prerequisite for entering the buffer)
        self.working_memory_buffer: Set[ItemIdx] = set()

        # endregion

    def __get_statistic_for_item(self, idx: ItemIdx):
        """Gets the correct statistic for an item."""
        if self.norm_attenuation_statistic is NormAttenuationStatistic.FractionKnown:
            # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
            return self._sensorimotor_norms.fraction_known(self.idx2label[idx])
        elif self.norm_attenuation_statistic is NormAttenuationStatistic.Prevalence:
            # Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the
            # purposes of attenuating the activation
            return scale_prevalence_01(prevalence_from_fraction_known(self._sensorimotor_norms.fraction_known(self.idx2label[idx])))
        else:
            raise NotImplementedError()

    # endregion

    def reset(self):
        super(SensorimotorComponent, self).reset()
        self.working_memory_buffer = set()

    # region tick()

    def _evolve_model(self) -> List[ModelEvent]:

        # Proceed with ._evolve_model() and record what became activated
        model_events = super(SensorimotorComponent, self)._evolve_model()

        activation_events, other_events = partition(model_events, lambda e: isinstance(e, ItemActivatedEvent))

        # There will be at most one event for each item which has an event
        assert len(activation_events) == len(set(e.item for e in activation_events))

        # Update buffer

        self.__prune_decayed_items_in_buffer()
        # Some events will get updated commensurately
        activation_events = self.__present_items_to_buffer(activation_events)

        return activation_events + other_events

    def __prune_decayed_items_in_buffer(self):
        """Removes items from the buffer which have dropped below threshold."""
        self.working_memory_buffer = {
            item
            for item in self.working_memory_buffer
            if self.activation_of_item_with_idx(item) >= self.buffer_threshold
        }

    def __present_items_to_buffer(self, activation_events: List[ItemActivatedEvent]) -> List[ModelEvent]:
        """
        Present a list of item activations to the buffer, and upgrades those which entered the buffer.
        :param activation_events:
            All activation events.
        :return:
            The same events, with some upgraded to buffer entry events.
            May also return a BufferFlood event if that is detected.
        :side effects:
            Mutates self.working_memory_buffer.
        """

        # Sometimes we can leave early. If nothing was activated, nothing will enter the buffer, and there are no events
        # to upgrade.
        if len(activation_events) == 0:
            return []

        # At this point self.working_memory_buffer is still the old buffer (after decayed items have been removed)
        presented_items = set(e.item for e in activation_events)
        items_already_in_buffer = self.working_memory_buffer & presented_items
        # I have a feeling that we'll never present things to the buffer which are already in there, but just in case...
        if len(items_already_in_buffer) > 0:
            logger.warning("Tried to present items to the buffer which were already in there.")
        eligible_items = presented_items - items_already_in_buffer

        # region New buffer items list of (item, activation)s

        # We will sort items in the buffer based on various bits of data. Temporarily make a sorting-data namedtuple.
        # The new buffer is everything in the current working_memory_buffer...
        new_buffer_items: Dict[ItemIdx: _BufferSortingData] = {
            item: _BufferSortingData(activation=self.activation_of_item_with_idx(item),
                                     # These items already in the buffer were not presented
                                     presented=False)
            for item in self.working_memory_buffer
        }
        # ...plus everything above threshold.
        # We use a dictionary with .update() here to overwrite the activation of anything already in the buffer.
        new_buffer_items.update({
            e.item: _BufferSortingData(activation=e.activation,
                                       # We've already worked out whether items are potentially entering the buffer
                                       presented=e.item in eligible_items)
            for e in activation_events
            if e.activation >= self.buffer_threshold
        })
        # Convert to a list of key-value pairs, sorted by activation, descending.
        # We want the order to be by activation, but with ties broken by recency, i.e. items being presented to the
        # buffer precede those already in the buffer.  Because Python's sorting is stable [0], meaning if we sort by
        # recency first, and then by activation, we get what we want [1].
        #     [0]: http://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        #     [1]: https://wiki.python.org/moin/HowTo/Sorting#Sort_Stability_and_Complex_Sorts
        # So first we sort by recency (i.e. whether they were presented), descending
        # (i.e. .presented==1 comes before .presented==0)
        new_buffer_items = sorted(new_buffer_items.items(), key=lambda kv: kv[1].presented, reverse=True)
        # Then we sort by activation, descending (larger activation first)
        # Also new_buffer_items is now a list of kv pairs, not a dictionary, so we don't need to use .items()
        new_buffer_items = sorted(new_buffer_items, key=lambda kv: kv[1].activation, reverse=True)

        # Trim down to size if necessary
        new_buffer_items = new_buffer_items[:self.buffer_size_limit]

        # endregion

        new_buffer = set(kv[0] for kv in new_buffer_items)

        # Detect if whole buffer will be replaced
        whole_buffer_replaced = len(new_buffer - self.working_memory_buffer) == self.buffer_size_limit

        # Update buffer: Get the keys (i.e. item idxs) from the sorted list
        self.working_memory_buffer = new_buffer

        # Upgrade events
        upgraded_events = [
            # Upgrade only those events which newly entered the buffer
            (ItemEnteredBufferEvent.from_activation_event(e)
             if e.item in self.working_memory_buffer - items_already_in_buffer
             else e)
            for e in activation_events
        ]

        if whole_buffer_replaced:
            upgraded_events.append(BufferFloodEvent(time=self.clock))

        return upgraded_events

    # endregion

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        """Labels of concepts"""
        return set(w for i, w in self.idx2label.items())

    def accessible_set(self) -> Set[ItemIdx]:
        """
        The items in the accessible set.
        May take a long time to produce: for quick internal checks use self._is_in_accessible_set(item)
        """
        return set(n
                   for n in self.graph.nodes
                   if self._is_in_accessible_set(n))

    def _is_in_accessible_set(self, item: ItemIdx) -> bool:
        """
        Use this rather than in self.accessible_set() for quick internal checks, to avoid having to run through the set
        generator.
        """
        return self.activation_of_item_with_idx(item) > self.activation_threshold

    def _presynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return activation * self._attenuation_statistic[idx]

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__,
        # so applying the cap does not effect whether the node will fire or not)
        return activation if activation <= self.activation_cap else self.activation_cap

    def _presynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node can only be activated if not in the working_memory_buffer (i.e. activation below pruning threshold)
        return not self._is_in_accessible_set(idx)


def scale_prevalence_01(prevalence: float) -> float:
    """
    Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the purposes of
    attenuating the activation.
    """
    return scale01((-2.575829303548901, 2.5758293035489004), prevalence)


def load_labels_from_sensorimotor() -> Dict[ItemIdx, ItemLabel]:
    return _load_labels(path.join(Preferences.graphs_dir, "sensorimotor words.nodelabels"))


def _load_graph(distance_type, length_factor, max_sphere_radius, use_prepruned, node_labelling_dictionary) -> Graph:
    if use_prepruned:
        logger.warning("Using pre-pruned graph. THIS SHOULD BE USED FOR TESTING PURPOSES ONLY!")

        edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {max_sphere_radius}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path, with_feedback=True)

        # nodes which got removed from the edgelist because all their edges got pruned
        for i, w in node_labelling_dictionary.items():
            sensorimotor_graph.add_node(Node(i))

    else:

        edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path,
                                                      ignore_edges_longer_than=max_sphere_radius)
    return sensorimotor_graph
