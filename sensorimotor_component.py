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
from model.common import ActivationValue, ItemLabel, _load_labels
from model.graph import Graph, Node
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_lognormal
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class SensorimotorComponent(TemporalSpatialPropagation):
    """
    The sensorimotor component of the model.
    """

    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 pruning_length: int,
                 lognormal_sigma: float,
                 impulse_pruning_threshold: ActivationValue,
                 buffer_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 use_prepruned: bool = False,
                 ):

        # Load graph
        idx2label = load_labels_from_sensorimotor()
        super(SensorimotorComponent, self).__init__(

            underlying_graph=_load_graph(distance_type, length_factor, pruning_length,
                                         use_prepruned, idx2label),
            idx2label=idx2label,
            # Sigma for the log-normal decay gets multiplied by the length factor, so that if we change the length
            # factor, sigma doesn't also  have to change for the behaviour of the model to be approximately equivalent.
            node_decay_function=make_decay_function_lognormal(sigma=lognormal_sigma * length_factor),
            buffer_pruning_threshold=buffer_pruning_threshold,
            activation_cap=activation_cap,
            impulse_pruning_threshold=impulse_pruning_threshold,
        )

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        return set(w for i, w in self.idx2label.items())


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


def _load_graph(distance_type, length_factor, pruning_length, use_prepruned, node_labelling_dictionary):
    if use_prepruned:
        logger.warning("Using pre-pruned graph. THIS SHOULD BE USED FOR TESTING PURPOSES ONLY!")

        edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {pruning_length}.edgelist"
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
                                                      ignore_edges_longer_than=pruning_length)
    return sensorimotor_graph
