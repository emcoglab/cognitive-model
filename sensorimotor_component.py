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
from os import path
from typing import Set, Optional, List

import yaml

from ldm.utils.maths import DistanceType
from model.graph_propagation import _load_labels
from model.basic_types import ActivationValue, ItemIdx, ItemLabel, Node
from model.events import ModelEvent, ItemActivatedEvent, ItemEnteredBufferEvent
from model.graph import Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_lognormal, prevalence_from_fraction_known, scale01
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class SensorimotorComponent(TemporalSpatialPropagation):
    """
    The sensorimotor component of the model.
    Uses a lognormal decay on nodes.
    """

    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 max_sphere_radius: int,
                 lognormal_sigma: float,
                 buffer_size_limit: int,
                 buffer_entry_threshold: ActivationValue,
                 buffer_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 use_prepruned: bool = False,
                 ):
        """
        :param distance_type:
            The metric used to determine distances between points.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param max_sphere_radius:
            What is the maximum radius of a sphere
        :param lognormal_sigma:
            The sigma parameter for the lognormal decay.
        :param buffer_size_limit:
            The maximum size of the buffer. After this, qualifying items will displace existing items rather than just
            being added.
        :param buffer_entry_threshold:
            The minimum activation required for a concept to enter the working_memory_buffer.
        :param buffer_pruning_threshold:
            The activation threshold at which to remove items from the working_memory_buffer.
        :param activation_cap:
            If None is supplied, no cap is used.
        :param use_prepruned:
            Whether to use the prepruned graphs or do pruning on load.
            Only to be used for testing purposes.
        """

        # Load graph
        idx2label = load_labels_from_sensorimotor()
        super(SensorimotorComponent, self).__init__(

            underlying_graph=_load_graph(distance_type, length_factor, max_sphere_radius,
                                         use_prepruned, idx2label),
            idx2label=idx2label,
            # Sigma for the log-normal decay gets multiplied by the length factor, so that if we change the length
            # factor, sigma doesn't also  have to change for the behaviour of the model to be approximately equivalent.
            node_decay_function=make_decay_function_lognormal(sigma=lognormal_sigma * length_factor),
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Thresholds

        # Use >= and < to test for above/below
        self.buffer_entry_threshold: ActivationValue = buffer_entry_threshold
        self.buffer_pruning_threshold: ActivationValue = buffer_pruning_threshold
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        self.buffer_size_limit = buffer_size_limit

        # A local copy of the sensorimotor norms data
        self.sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()

        # endregion

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are currently being consciously considered.
        #
        # A fixed size (self.buffer_size_limit).  Items may enter the buffer when they are activated and leave when they
        # decay sufficiently (self.buffer_pruning_threshold) or are displaced.
        #
        # Currently this *could* be implemented as a simple property which lists the top-n most activated things, rather
        # than being meticulously maintained during as the model runs.  However it is better to maintain it as we go as
        # it will be easier to extend it into a multi-component buffer in the future.
        self.working_memory_buffer: Set[ItemIdx] = set()

        # endregion

    def reset(self):
        super(SensorimotorComponent, self).reset()
        self.working_memory_buffer = set()

    def tick(self) -> List[ModelEvent]:
        # Clear cruft from the buffer
        self._prune_decayed_items_in_buffer()

        # Proceed with the tick
        return super(SensorimotorComponent, self).tick()

    def activate_item_with_idx(self, idx: ItemIdx, activation: ActivationValue) -> Optional[ItemActivatedEvent]:
        # Activate the item
        activation_event = super(SensorimotorComponent, self).activate_item_with_idx(idx, activation)

        # Present it as available to enter the buffer if it activated
        if activation_event:
            item_did_enter_buffer = self._present_to_working_memory_buffer(idx)
            if item_did_enter_buffer:
                # If it entered the buffer, upgrade the event
                activation_event = ItemEnteredBufferEvent.from_activation_event(activation_event)

        return activation_event

    def _present_to_working_memory_buffer(self, item: ItemIdx) -> bool:
        """
        Try to get an item into the buffer just as it's being activated.
        :param item:
            The candidate item
        :return:
            True if the item got into the buffer, else False.
        """
        activation: ActivationValue = self.activation_of_item_with_idx(item)

        # Check if item can enter buffer
        if activation < self.buffer_entry_threshold:
            return False

        # The item is eligible for adding, but if only if there is room for it, else it may displace something
        # already in the buffer.

        # If there is room, it just goes in
        if len(self.working_memory_buffer) < self.buffer_size_limit:
            self.working_memory_buffer.add(item)
            return True

        # If there wasn't room, we may displace something

        # Item with the lowest activation in the WM buffer
        lowest_item, lowest_activation = min(
            ((n, self.activation_of_item_with_idx(n)) for n in self.working_memory_buffer),
            key=lambda tup: tup[1])

        # If everything already in the buffer is larger than the candidate, it doesn't get let in
        if activation < lowest_activation:
            return False

        # If it's larger than something, it does
        self.working_memory_buffer.remove(lowest_item)
        self.working_memory_buffer.add(item)
        return True

    def _prune_decayed_items_in_buffer(self):
        """Removes items from the buffer which have dropped below threshold."""
        items_to_prune = []
        for item in self.working_memory_buffer:
            if self.activation_of_item_with_idx(item) < self.buffer_pruning_threshold:
                items_to_prune.append(item)

        for item in items_to_prune:
            self.working_memory_buffer.remove(item)

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        """Labels of concepts"""
        return set(w for i, w in self.idx2label.items())

    def items_in_buffer(self) -> Set[ItemIdx]:
        """Items which are above the working_memory_buffer-pruning threshold."""
        return set(
            n
            for n in self.graph.nodes
            if self.activation_of_item_with_idx(n) >= self.buffer_pruning_threshold
        )

    def accessible_set(self) -> Set[ItemIdx]:
        """The items in the accessible set."""
        return set(
            n
            for n in self.graph.nodes
            if self.activation_of_item_with_idx(n) > 0
        )

    def _presynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return self._attenuate_by_fraction_known(idx, activation)

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__,
        # so applying the cap does not effect whether the node will fire or not.
        return activation if activation <= self.activation_cap else self.activation_cap

    def _presynaptic_guard(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node can only fire if not in the working_memory_buffer (i.e. activation below pruning threshold)
        return idx not in self.accessible_set()

    def _attenuate_by_prevalence(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """Attenuates the activation by the prevalence of the item."""
        prevalence = prevalence_from_fraction_known(self.sensorimotor_norms.fraction_known(self.idx2label[item]))
        # Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the
        # purposes of attenuating the activation
        scaled_prevalence = scale01((-2.575829303548901, 2.5758293035489004), prevalence)
        return activation * scaled_prevalence

    def _attenuate_by_fraction_known(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """Attenuates the activation by the fraction of people who know the item."""
        # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
        return activation * self.sensorimotor_norms.fraction_known(self.idx2label[item])


def save_model_spec_sensorimotor(length_factor, max_sphere_radius, sigma, response_dir):
    spec = {
        "Length factor": length_factor,
        "Max sphere radius": max_sphere_radius,
        "Log-normal sigma": sigma,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def load_labels_from_sensorimotor():
    return _load_labels(path.join(Preferences.graphs_dir, "sensorimotor words.nodelabels"))


def _load_graph(distance_type, length_factor, max_sphere_radius, use_prepruned, node_labelling_dictionary):
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
