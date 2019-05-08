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
from enum import Enum, auto
from os import path
from typing import Dict, Set

import yaml
from pandas import DataFrame

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.graph import Node, Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.temporal_spreading_activation import TemporalSpreadingActivation, load_labels_from_corpus
from model.utils.maths import make_decay_function_lognormal, make_decay_function_exponential_with_decay_factor, \
    make_decay_function_gaussian_with_sd
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


def blank_node_activation_record() -> ActivationRecord:
    """A record for an unactivated node."""
    return ActivationRecord(activation=0, time_activated=-1)


class ItemActivatedEvent:
    """
    A node activation event.
    Used to pass out of TSA.tick().
    """

    def __init__(self, label: str, activation: ActivationValue, time_activated: int):
        self.label = label
        # Use an ActivationRecord to store this so we don't have repeated code
        self._record = ActivationRecord(activation=activation, time_activated=time_activated)

    @property
    def activation(self) -> ActivationValue:
        return self._record.activation

    @property
    def time_activated(self) -> int:
        return self._record.time_activated

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


class EdgePruningType(Enum):
    Length     = auto()
    Percent    = auto()
    Importance = auto()


class LinguisticComponent(TemporalSpreadingActivation):
    """
    The linguistic component of the model.
    """

    def __init__(self,
                 n_words: int,
                 distributional_model: DistributionalSemanticModel,
                 freq_dist: FreqDist,
                 length_factor: int,
                 node_decay_factor: float,
                 edge_decay_sd_factor: float,
                 impulse_pruning_threshold: ActivationValue,
                 firing_threshold: ActivationValue,
                 distance_type: DistanceType = None,
                 edge_pruning=None,
                 edge_pruning_type: EdgePruningType = None,
                 ):

        graph = LinguisticComponent._load_graph(n_words, length_factor, distributional_model,
                                                distance_type, edge_pruning_type, edge_pruning)

        node_labelling_dictionary = load_labels_from_corpus(distributional_model.corpus_meta, n_words)

        super(LinguisticComponent, self).__init__(
            graph=graph,
            item_labelling_dictionary=node_labelling_dictionary,
            impulse_pruning_threshold=impulse_pruning_threshold,
            firing_threshold=firing_threshold,
            node_decay_function=make_decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=make_decay_function_gaussian_with_sd(
                sd=edge_decay_sd_factor * length_factor)
        )

        self.available_words: Set[ItemLabel] = set(freq_dist.most_common_tokens(n_words))

    @classmethod
    def _load_graph(cls, n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning) -> Graph:

        # Check if distance_type is needed and get filename
        if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.ngram:
            assert distance_type is None
            graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"
        else:
            assert distance_type is not None
            graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

        # Load graph
        if edge_pruning is None:
            logger.info(f"Loading graph from {graph_file_name}")
            graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name))

        elif edge_pruning_type is EdgePruningType.Length:
            logger.info(f"Loading graph from {graph_file_name}, pruning any edge longer than {edge_pruning}")
            graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                             ignore_edges_longer_than=edge_pruning,
                                             keep_at_least_n_edges=Preferences.min_edges_per_node)

        elif edge_pruning_type is EdgePruningType.Percent:
            quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"
            quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0,
                                               index_col=None)
            pruning_length = quantile_data[
                # Use 1 - so that smallest top quantiles get converted to longest edges
                quantile_data["Top quantile"] == 1 - (edge_pruning / 100)
                ]["Pruning length"].iloc[0]
            logger.info(f"Loading graph from {graph_file_name}, pruning longest {edge_pruning}% of edges (anything over {pruning_length})")
            graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                             ignore_edges_longer_than=edge_pruning,
                                             keep_at_least_n_edges=Preferences.min_edges_per_node)

        elif edge_pruning_type is EdgePruningType.Importance:
            logger.info(
                f"Loading graph from {graph_file_name}, pruning longest {edge_pruning}% of edges")
            graph = Graph.load_from_edgelist_with_importance_pruning(
                file_path=path.join(Preferences.graphs_dir, graph_file_name),
                ignore_edges_with_importance_greater_than=edge_pruning,
                keep_at_least_n_edges=Preferences.min_edges_per_node)

        else:
            raise NotImplementedError()

        return graph

    @property
    def is_connected(self) -> bool:
        return self.graph.is_connected()

    @property
    def has_orphans(self) -> bool:
        return self.graph.has_orphaned_nodes()


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


def save_model_spec_linguistic(edge_decay_sd_factor, firing_threshold, length_factor, model_name, n_words,
                               response_dir):
    spec = {
        "Model name": model_name,
        "Length factor": length_factor,
        "SD factor": edge_decay_sd_factor,
        "Firing threshold": firing_threshold,
        "Words": n_words,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def save_model_spec_sensorimotor(length_factor, max_sphere_radius, sigma, response_dir):
    spec = {
        "Length factor": length_factor,
        "Max sphere radius": max_sphere_radius,
        "Log-normal sigma": sigma,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def load_model_spec(response_dir) -> dict:
    with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
        return yaml.load(spec_file, yaml.SafeLoader)
