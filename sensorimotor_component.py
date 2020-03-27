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
from enum import Enum, auto
from os import path
from typing import Set, List, Dict, Optional

from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, ItemIdx, ItemLabel, Node
from model.buffer import WorkingMemoryBuffer, AccessibleSet
from model.events import ModelEvent, ItemActivatedEvent
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
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        # Data

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
        self.working_memory_buffer: WorkingMemoryBuffer = WorkingMemoryBuffer(buffer_threshold, buffer_capacity)

        # The set of items which are "accessible to conscious awareness" even if they are not in the working memory
        # buffer
        self.accessible_set: AccessibleSet = AccessibleSet(threshold=accessible_set_threshold, capacity=accessible_set_capacity)

        # endregion

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
        self.accessible_set.clear()

    # region tick()

    def _evolve_model(self) -> List[ModelEvent]:

        # Decay events before activating anything new
        # (in case buffer or accessible set membership is used to modulate or guard anything)
        decay_events = self.working_memory_buffer.prune_decayed_items(
            activation_lookup=self.activation_of_item_with_idx,
            time=self.clock)
        self.accessible_set.prune_decayed_items(activation_lookup=self.activation_of_item_with_idx,
                                                time=self.clock)

        logger.info(f"\tAS: {len(self.accessible_set)}/{self.accessible_set.capacity if self.accessible_set.capacity is not None else '∞'} "
                    f"(MP: {self.accessible_set.pressure})")

        # Proceed with ._evolve_model() and record what became activated
        # Activation and firing may be affected by the size of or membership to the accessible set and the buffer, but
        # nothing will ENTER it until later, and everything that will LEAVE this tick already has done so.
        model_events = super(SensorimotorComponent, self)._evolve_model()
        activation_events, other_events = partition(model_events, lambda e: isinstance(e, ItemActivatedEvent))
        # There will be at most one event for each item which has an event
        assert len(activation_events) == len(set(e.item for e in activation_events))

        # Update accessible set
        self.accessible_set.present_items(activation_events=activation_events,
                                          activation_lookup=self.activation_of_item_with_idx,
                                          time=self.clock)
        # Update buffer
        # Some events will get updated commensurately.
        # `activation_events` may now contain some non-activation events.
        activation_events = self.working_memory_buffer.present_items(activation_events, activation_lookup=self.activation_of_item_with_idx, time=self.clock)

        return decay_events + activation_events + other_events

    def _presynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # If accumulated activation is over the cap, apply the cap
        activation = min(activation, self.activation_cap)
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        activation *= self._attenuation_statistic[idx]
        # When AS is full, MP is 1, and activation is killed.
        # When AS is empty, MP is 0, and activation is unaffected.
        activation *= 1 - self.accessible_set.pressure
        return activation

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__, so
        # applying the cap does not effect whether the node will fire or not)
        return min(activation, self.activation_cap)

    def _postsynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node will only fire if it's not in the accessible set
        return idx not in self.accessible_set

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
