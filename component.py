"""
===========================
Common features of components of the cognitive model.
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

import json
import logging
from collections import namedtuple
from os import path
from typing import Dict, Set

import yaml

from ldm.utils.maths import DistanceType
from model.graph import Node, Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_lognormal
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


ActivationValue = float
ItemIdx = Node
ItemLabel = str


class ActivationRecord(namedtuple('ActivationRecord', ['activation',
                                                       'time_activated'])):
    """
    ActivationRecord stores a historical node activation.

    It is immutable, so must be used in conjunction with TSA.node_decay_function in order to determine the
    current activation of a node.

    `activation` stores the total accumulated level of activation at this node when it was activated.
    `time_activated` stores the clock value when the node was last activated, or -1 if it has never been activated.

    Don't thoughtlessly change this class as it probably needs to remain a small namedtuple for performance reasons.
    """
    __slots__ = ()


class ItemActivatedEvent(namedtuple('ItemActivatedEvent', ['activation',
                                                           'time_activated',
                                                           'label'])):
    """
    A node activation event.
    Used to pass out of TSA.tick().
    """
    # TODO: this is basically the same as ActivationRecord, and could probably be removed in favour of it.
    label: ItemLabel
    activation: ActivationValue
    time_activated: int

    def __repr__(self) -> str:
        return f"<'{self.label}' ({self.activation}) @ {self.time_activated}>"


def _load_labels(nodelabel_path: str) -> Dict[ItemIdx, ItemLabel]:
    with open(nodelabel_path, mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    # TODO: this isn't a great way to do this
    node_labelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_labelling_dictionary[ItemIdx(k)] = v
    return node_labelling_dictionary


def blank_node_activation_record() -> ActivationRecord:
    """A record for an unactivated node."""
    return ActivationRecord(activation=0, time_activated=-1)


class SensorimotorComponent(TemporalSpatialPropagation):
    """
    The sensorimotor component of the model.
    """

    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 pruning_length: int,
                 lognormal_sigma: float,
                 buffer_pruning_threshold: ActivationValue,
                 activation_cap: ActivationValue,
                 use_prepruned: bool = False,
                 ):

        # Load node relabelling dictionary
        logger.info(f"Loading node labels")
        node_labelling_dictionary = load_labels_from_sensorimotor()

        # Load graph
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

        super(SensorimotorComponent, self).__init__(
            underlying_graph=sensorimotor_graph,
            point_labelling_dictionary=node_labelling_dictionary,
            # Sigma for the log-normal decay gets multiplied by the length factor, so that if we change the length
            # factor, sigma doesn't also  have to change for the behaviour of the model to be approximately equivalent.
            node_decay_function=make_decay_function_lognormal(sigma=lognormal_sigma * length_factor),
            buffer_pruning_threshold=buffer_pruning_threshold,
            activation_cap=activation_cap,
        )

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        return set(w for i, w in self.idx2label.items())


def load_labels_from_sensorimotor():
    return _load_labels(path.join(Preferences.graphs_dir, "sensorimotor words.nodelabels"))


def save_model_spec_linguistic(edge_decay_sd_factor, firing_threshold, length_factor, model_name, n_words, response_dir):
    spec = {
        "Model name":       model_name,
        "Length factor":    length_factor,
        "SD factor":        edge_decay_sd_factor,
        "Firing threshold": firing_threshold,
        "Words":            n_words,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def save_model_spec_sensorimotor(length_factor, max_sphere_radius, sigma, response_dir):
    spec = {
        "Length factor":     length_factor,
        "Max sphere radius": max_sphere_radius,
        "Log-normal sigma":   sigma,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def load_model_spec(response_dir) -> dict:
    with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
        return yaml.load(spec_file, yaml.SafeLoader)