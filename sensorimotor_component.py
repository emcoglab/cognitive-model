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
from typing import Set, List, Dict, Optional

from ldm.utils.maths import DistanceType, clamp01

from model.basic_types import ActivationValue, ItemIdx, ItemLabel, Node
from model.events import ModelEvent, ItemActivatedEvent, ItemEnteredBufferEvent, BufferFloodEvent, ItemLeftBufferEvent
from model.graph import Graph
from model.graph_propagation import _load_labels
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.iterable import partition
from model.utils.maths import make_decay_function_lognormal, prevalence_from_fraction_known, scale_prevalence_01
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


class WorkingMemoryBuffer:

    def __init__(self, capacity: Optional[int], items: Set[ItemIdx] = None):
        self._capacity: Optional[int] = capacity
        self.items: Set[ItemIdx] = set() if items is None else items
        if self.capacity is not None:
            assert len(self.items) <= self.capacity

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    def replace_contents(self, new_items: Set[ItemIdx]):
        assert len(new_items) <= self.capacity
        self.items = new_items

    def clear(self):
        self.replace_contents(set())

    @dataclass
    class _SortingData:
        """
        For sorting items before entry to the buffer.
        """
        activation: ActivationValue
        being_presented: bool


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
                 node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float,
                 buffer_capacity: Optional[int],
                 accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue,
                 accessible_set_threshold: ActivationValue,
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
        :param node_decay_lognormal_median:
            The node_decay_median of the lognormal decay.
        :param node_decay_lognormal_sigma:
            The node_decay_sigma parameter for the lognormal decay.
        :param buffer_capacity:
            The maximum size of the buffer. After this, qualifying items will displace existing items rather than just
            being added.
        :param buffer_threshold:
            The minimum activation required for a concept to enter the working_memory_buffer.
        :param accessible_set_threshold:
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
        # node_decay_lognormal_sigma or node_decay_lognormal_median == 0 will probably cause a division-by-zero error, and anyway is
        # degenerate: it causes everything to decay to 0 activation in a single tick.
        assert (node_decay_lognormal_median > 0)
        assert (node_decay_lognormal_sigma > 0)
        # zero-size buffer size limit is degenerate: the buffer is always empty.
        assert (buffer_capacity is None) or (buffer_capacity > 0)
        # zero-size accessible set size limit is degenerate: the set is always empty.
        assert (accessible_set_capacity is None) or (accessible_set_capacity > 0)
        assert (activation_cap
                # If activation_cap == buffer_threshold, items will only enter the buffer when fully activated.
                >= buffer_threshold
                # If buffer_pruning_threshold == accessible_set_threshold then the only things in the accessible set
                # will be those items which were displaced from the buffer before being pruned. We probably won't use
                # this but it's not invalid or degenerate.
                >= accessible_set_threshold
                # accessible_set_threshold must be strictly positive, else no item can ever be reactivated (since
                # membership to the accessible set is a guard to reactivation).
                > 0)

        # endregion

        # Load graph
        idx2label = load_labels_from_sensorimotor()
        super(SensorimotorComponent, self).__init__(

            underlying_graph=_load_graph(distance_type, length_factor, max_sphere_radius,
                                         use_prepruned, idx2label),
            idx2label=idx2label,
            node_decay_function=make_decay_function_lognormal(median=node_decay_lognormal_median, sigma=node_decay_lognormal_sigma),
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        self._model_spec.update({
            "Distance type": distance_type.name,
            "Length factor": length_factor,
            "Max sphere radius": max_sphere_radius,
            "Log-normal median": node_decay_lognormal_median,
            "Log-normal sigma": node_decay_lognormal_sigma,
            "Buffer capacity": buffer_capacity,
            "Buffer threshold": buffer_threshold,
            "Norm attenuation statistic": norm_attenuation_statistic.name,
            "Activation cap": activation_cap,
            "Activation threshold": accessible_set_threshold,
            "Accessible set capacity": accessible_set_capacity,
        })

        # Thresholds

        # Use >= and < to test for above/below
        self.buffer_threshold: ActivationValue = buffer_threshold
        self.accessible_set_threshold: ActivationValue = accessible_set_threshold
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        # Data

        self.accessible_set_capacity: Optional[int] = accessible_set_capacity

        # A local copy of the sensorimotor norms data
        self._sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()

        self.norm_attenuation_statistic: NormAttenuationStatistic = norm_attenuation_statistic
        self._attenuation_statistic: Dict[ItemIdx, float] = {
            idx: self._get_statistic_for_item(idx)
            for idx in self.graph.nodes
        }

        # endregion

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are currently being consciously considered.
        #
        # A fixed size (self.buffer_capacity).  Items may enter the buffer when they are activated and leave when they
        # decay sufficiently (self.buffer_pruning_threshold) or are displaced.
        #
        # This is updated each .tick() based on items which fired (a prerequisite for entering the buffer)
        self.working_memory_buffer: WorkingMemoryBuffer = WorkingMemoryBuffer(buffer_capacity)

        # Bounded between 0 and 1 inclusive
        # 0 when accessible set is empty, 1 when full.
        # 0 when there is no bounded capacity.
        self.__memory_pressure: float = 0

        # The set of items which are "accessible to conscious awareness" even if they are not in the working memory
        # buffer
        self.accessible_set: Set[ItemIdx] = set()

        # endregion

    @property
    def accessible_set(self):
        return self.__accessible_set

    @accessible_set.setter
    def accessible_set(self, value):
        # Update memory pressure whenever we alter the accessible set
        self.__accessible_set = value
        if self.accessible_set_capacity is not None:
            self.__memory_pressure = clamp01(len(self.__accessible_set) / self.accessible_set_capacity)
        else:
            self.__memory_pressure = 0

    @property
    def memory_pressure(self) -> float:
        return self.__memory_pressure

    def _get_statistic_for_item(self, idx: ItemIdx):
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
        self.working_memory_buffer.clear()
        self.accessible_set = set()

    # region tick()

    def _evolve_model(self) -> List[ModelEvent]:

        # Decay events before activating anything new
        # (in case buffer or accessible set membership is used to modulate or guard anything)
        decay_events = self.__prune_decayed_items_in_buffer()
        self.__prune_decayed_items_in_accessible_set()

        logger.info(f"\tAS: {len(self.accessible_set)}/{self.accessible_set_capacity if self.accessible_set_capacity is not None else '∞'} "
                    f"(MP: {self.memory_pressure})")

        # Proceed with ._evolve_model() and record what became activated
        # Activation and firing may be affected by the size of or membership to the accessible set and the buffer, but
        # nothing will ENTER it until later, and everything that will LEAVE this tick already has done so.
        model_events = super(SensorimotorComponent, self)._evolve_model()
        activation_events, other_events = partition(model_events, lambda e: isinstance(e, ItemActivatedEvent))
        # There will be at most one event for each item which has an event
        assert len(activation_events) == len(set(e.item for e in activation_events))

        # Update accessible set
        self.__present_items_to_accessible_set(activation_events)
        # Update buffer
        # Some events will get updated commensurately.
        # `activation_events` may now contain some non-activation events.
        activation_events = self.__present_items_to_buffer(activation_events)

        return decay_events + activation_events + other_events

    def _presynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # If accumulated activation is over the cap, apply the cap
        activation = min(activation, self.activation_cap)
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        activation *= self._attenuation_statistic[idx]
        # When AS is full, MP is 1, and activation is killed.
        # When AS is empty, MP is 0, and activation is unaffected.
        activation *= 1 - self.memory_pressure
        return activation

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__, so
        # applying the cap does not effect whether the node will fire or not)
        return min(activation, self.activation_cap)

    def _postsynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node will only fire if it's not in the accessible set
        return idx not in self.accessible_set

    def __prune_decayed_items_in_accessible_set(self):
        """
        Removes items from the accessible set which have dropped below threshold.

        Cardinality of accessible set is used to dampen activation, but does not affect the accessible set threshold, so
        we can safely prune things here without worrying about that.

        :side effects:
            Mutates self.accessible_set.
            Mutates self.memory_pressure.
        """
        self.accessible_set -= {
            item
            for item in self.accessible_set
            if self.activation_of_item_with_idx(item) < self.accessible_set_threshold
        }

    def __prune_decayed_items_in_buffer(self) -> List[ItemLeftBufferEvent]:
        """
        Removes items from the buffer which have dropped below threshold.
        :return:
            Events for items which left the buffer by decaying out.
        :side effects:
            Mutates self.working_memory_buffer.
        """

        new_buffer_items = {
            item
            for item in self.working_memory_buffer.items
            if self.activation_of_item_with_idx(item) >= self.buffer_threshold
        }
        decayed_out = self.working_memory_buffer.items - new_buffer_items
        self.working_memory_buffer.replace_contents(new_buffer_items)
        return [
            ItemLeftBufferEvent(time=self.clock, item=item)
            for item in decayed_out
        ]

    def __present_items_to_accessible_set(self, activation_events: List[ItemActivatedEvent]):
        """
        Presents a list of item actiavtions to the accessible set.
        :param activation_events:
            All activatino events
        :side effects:
            Mutates self.accessible_set.
            Mutates self.memory_pressure.
        """
        if len(activation_events) == 0:
            return

        # unlike the buffer, we're not returning any events, and there is no size limit, so we don't need to be so
        # careful about confirming what's already in there and what's getting replaced, etc.

        self.accessible_set |= {
            e.item
            for e in activation_events
            if e.activation >= self.accessible_set_threshold
        }

    def __present_items_to_buffer(self, activation_events: List[ItemActivatedEvent]) -> List[ModelEvent]:
        """
        Present a list of item activations to the buffer, and upgrades those which entered the buffer.
        :param activation_events:
            All activation events.
        :return:
            The same events, with some upgraded to buffer entry events.
            Plus events for items which left the buffer through displacement.
            May also return a BufferFlood event if that is detected.
        :side effects:
            Mutates self.working_memory_buffer.
        """

        if len(activation_events) == 0:
            return []

        # At this point self.working_memory_buffer is still the old buffer (after decayed items have been removed)
        presented_items = set(e.item for e in activation_events)

        # Don't present items already in the buffer
        items_already_in_buffer = self.working_memory_buffer.items & presented_items
        presented_items -= items_already_in_buffer

        # region New buffer items list of (item, activation)s

        # First build a new buffer out of everything which *could* end up in the buffer, then cut out things which don't
        # belong there

        # We will sort items in the buffer based on various bits of data.
        # The new buffer is everything in the current working_memory_buffer...
        new_buffer_items: Dict[ItemIdx: WorkingMemoryBuffer._SortingData] = {
            item: WorkingMemoryBuffer._SortingData(activation=self.activation_of_item_with_idx(item),
                                     # These items already in the buffer were not presented
                                     being_presented=False)
            for item in self.working_memory_buffer.items
        }
        # ...plus everything above threshold.
        # We use a dictionary with .update() here to overwrite the activation of anything already in the buffer.
        new_buffer_items.update({
            event.item: WorkingMemoryBuffer._SortingData(activation=event.activation,
                                           # We've already worked out whether items are potentially entering the buffer
                                           being_presented=event.item in presented_items)
            for event in activation_events
            if event.activation >= self.buffer_threshold
        })

        # Convert to a list of key-value pairs, sorted by activation, descending.
        # We want the order to be by activation, but with ties broken by recency, i.e. items being presented to the
        # buffer precede those already in the buffer.  Because Python's sorting is stable, meaning if we sort by
        # recency first, and then by activation, we get what we want [0].
        #
        # So first we sort by recency (i.e. whether they were presented), descending
        # (i.e. .presented==1 comes before .presented==0)
        #
        #     [0]: https://wiki.python.org/moin/HowTo/Sorting#Sort_Stability_and_Complex_Sorts
        new_buffer_items = sorted(new_buffer_items.items(), key=lambda kv: kv[1].being_presented, reverse=True)
        # Then we sort by activation, descending (larger activation first)
        # Also new_buffer_items is now a list of kv pairs, not a dictionary, so we don't need to use .items()
        new_buffer_items = sorted(new_buffer_items, key=lambda kv: kv[1].activation, reverse=True)

        # Trim down to size if necessary
        if self.working_memory_buffer.capacity is not None:
            new_buffer_items = new_buffer_items[:self.working_memory_buffer.capacity]

        # endregion

        new_buffer = set(kv[0] for kv in new_buffer_items)

        # For returning additional BufferEvents
        whole_buffer_replaced = len(new_buffer - self.working_memory_buffer.items) == self.working_memory_buffer.capacity
        displaced_items = self.working_memory_buffer.items - new_buffer

        # Update buffer: Get the keys (i.e. item idxs) from the sorted list
        self.working_memory_buffer.replace_contents(new_buffer)

        # Upgrade events
        upgraded_events = [
            # Upgrade only those events which newly entered the buffer
            (
                ItemEnteredBufferEvent.from_activation_event(e)
                if e.item in self.working_memory_buffer.items - items_already_in_buffer
                else e
            )
            for e in activation_events
        ]

        # Add extra events if necessary
        if whole_buffer_replaced:
            upgraded_events.append(BufferFloodEvent(time=self.clock))
        upgraded_events.extend([ItemLeftBufferEvent(time=self.clock, item=i) for i in displaced_items])

        return upgraded_events

    # endregion

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        """Labels of concepts"""
        return set(w for i, w in self.idx2label.items())


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
