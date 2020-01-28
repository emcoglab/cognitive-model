"""
===========================
The linguistic component of the model.
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
from typing import Dict

from numpy import Infinity
from pandas import DataFrame

from ldm.corpus.corpus import CorpusMetadata
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.graph import Graph, EdgePruningType
from model.graph_propagation import _load_labels
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.maths import make_decay_function_exponential_with_decay_factor, make_decay_function_gaussian_with_sd
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class LinguisticComponent(TemporalSpreadingActivation):
    """
    The linguistic component of the model.
    Uses an exponential decay on nodes and a gaussian decay on edges.
    """

    def __init__(self,
                 n_words: int,
                 distributional_model: DistributionalSemanticModel,
                 length_factor: int,
                 node_decay_factor: float,
                 edge_decay_sd_factor: float,
                 impulse_pruning_threshold: ActivationValue,
                 firing_threshold: ActivationValue,
                 activation_cap: ActivationValue = Infinity,
                 distance_type: DistanceType = None,
                 edge_pruning=None,
                 edge_pruning_type: EdgePruningType = None,
                 ):
        """
        :param n_words:
            The number of words to use for this model.
        :param distributional_model:
            The form of the linguistic distributional space.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param node_decay_factor:
            The decay factor for the exponential decay on nodes.
        :param edge_decay_sd_factor:
            The SD of the gaussian curve governing edge decay.
        :param firing_threshold:
        :param distance_type:
            The metric used to determine distances between vectors.
        :param edge_pruning:
            The level of edge pruning.
            Only used if `edge_pruning_type` is not None.
        :param edge_pruning_type:
            How edges should be pruned.
            This determines how the `edge_pruning` value is interpreted (e.g. as a percentage, absolute length or
            importance value).
            Use None to not prune, and ignore `edge_pruning`.
        """

        super(LinguisticComponent, self).__init__(
            graph=_load_graph(n_words, length_factor, distributional_model,
                              distance_type, edge_pruning_type, edge_pruning),
            idx2label=load_labels_from_corpus(distributional_model.corpus_meta, n_words),
            impulse_pruning_threshold=impulse_pruning_threshold,
            firing_threshold=firing_threshold,
            node_decay_function=make_decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=make_decay_function_gaussian_with_sd(
                sd=edge_decay_sd_factor * length_factor)
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        self._model_spec.update({
            "Words": n_words,
            "Model name": distributional_model.name,
            "Length factor": length_factor,
            "Impulse pruning threshold": impulse_pruning_threshold,
            "SD factor": edge_decay_sd_factor,
            "Node decay": node_decay_factor,
            "Firing threshold": firing_threshold,
        })

        # Cap on a node's total activation after receiving incoming.
        self.activation_cap = activation_cap
        if self.activation_cap < self.firing_threshold:
            raise ValueError(f"activation cap {self.activation_cap} cannot be less than the firing threshold {self.firing_threshold}")

        # endregion

    def _postsynaptic_modulation(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__,
        # so applying the cap does not effect whether the node will fire or not.
        return activation if activation <= self.activation_cap else self.activation_cap


def _load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning) -> Graph:

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


def load_labels_from_corpus(corpus: CorpusMetadata, n_words: int) -> Dict[ItemIdx, ItemLabel]:
    return _load_labels(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))
