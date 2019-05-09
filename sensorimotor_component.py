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
from typing import Set

import yaml

from ldm.utils.maths import DistanceType
from model.common import ActivationValue, ItemLabel, _load_labels, ItemIdx
from model.graph import Graph, Node
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_lognormal, prevalence_from_fraction_known, rescale01
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
                 impulse_pruning_threshold: ActivationValue,
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
        :param buffer_pruning_threshold:
            The activation threshold at which to remove items from the buffer.
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
            activation_cap=activation_cap,
            impulse_pruning_threshold=impulse_pruning_threshold,
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        # Thresholds
        # Use >= and < to test for above/below
        self.buffer_pruning_threshold = buffer_pruning_threshold

        # A local copy of the sensorimotor norms data
        self.sensorimotor_norms = SensorimotorNorms()

        # endregion

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        """Labels of concepts"""
        return set(w for i, w in self.idx2label.items())

    def items_in_buffer(self) -> Set[ItemIdx]:
        """Items which are above the buffer-pruning threshold. """
        return set(
            n
            for n in self.graph.nodes
            if self.activation_of_item_with_idx(n) >= self.buffer_pruning_threshold
        )

    def _presynaptic_modulation(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return self._attenuate_activation_by_fraction_known(item, activation)

    def _presynaptic_guard(self, activation: ActivationValue) -> bool:
        # Node can only fire if not in the buffer (i.e. activation below pruning threshold)
        return activation < self.buffer_pruning_threshold

    def _attenuate_activation_by_prevelence(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """Attenuates the activation by the prevalence of the item."""
        prevalence = prevalence_from_fraction_known(self.sensorimotor_norms.fraction_known(self.idx2label[item]))
        # Brysbaert's prevelence has a defined range, so we can rescale it into [0, 1] for the purposes of attenuating the activation
        scaled_prevalence = rescale01(-2.575829303548901, 2.5758293035489004, prevalence)
        # Linearly scale prevalence into [0, 1] for purposes of scaling
        return activation * scaled_prevalence

    def _attenuate_activation_by_fraction_known(self, item: ItemIdx, activation: ActivationValue) -> ActivationValue:
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
