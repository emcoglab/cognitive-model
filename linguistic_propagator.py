from os import path
from typing import Dict, Optional, List

from pandas import DataFrame

from ldm.corpus.corpus import CorpusMetadata
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.basic_types import Length, ItemIdx, ItemLabel, ActivationValue

from model.utils.logging import logger
from model.events import ModelEvent
from model.graph import Graph, EdgePruningType
from model.graph_propagator import GraphPropagator, _load_labels, IMPULSE_PRUNING_THRESHOLD
from model.utils.maths import make_decay_function_exponential_with_decay_factor, make_decay_function_gaussian_with_sd
from preferences import Preferences


class LinguisticPropagator(GraphPropagator):
    """
    Spreading activation on a graph over time.
    Nodes have a firing threshold and an activation cap.
    """

    def __init__(self,
                 length_factor: int,
                 n_words: int,
                 distributional_model: DistributionalSemanticModel,
                 distance_type: Optional[DistanceType],
                 node_decay_factor: float,
                 edge_decay_sd_factor: float,
                 edge_pruning_type: Optional[EdgePruningType],
                 edge_pruning: Optional[Length],
                 ):
        """
        :param n_words:
            The number of words to use for this model.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param distributional_model:
            The form of the linguistic distributional space.
        :param distance_type:
            The metric used to determine distances between vectors.
        :param edge_decay_sd_factor:
            The SD of the gaussian curve governing edge decay.
        :param node_decay_factor:
            The decay factor for the exponential decay on nodes.
        :param edge_pruning_type:
            How edges should be pruned.
            This determines how the `edge_pruning` value is interpreted (e.g. as a percentage, absolute length or
            importance value).
            Use None to not prune, and ignore `edge_pruning`.
        :param edge_pruning:
            The level of edge pruning.
            Only used if `edge_pruning_type` is not None.
        """

        # Load graph
        idx2label = _load_labels_from_corpus(distributional_model.corpus_meta, n_words)
        super(LinguisticPropagator, self).__init__(
            graph=_load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning),
            idx2label=idx2label,
            impulse_pruning_threshold=IMPULSE_PRUNING_THRESHOLD,
            node_decay_function=make_decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=make_decay_function_gaussian_with_sd(
                sd=edge_decay_sd_factor * length_factor),
        )

        self._model_spec_additional_fields = {
            "Words": n_words,
            "Model name": distributional_model.name,
            "Length factor": length_factor,
            "SD factor": edge_decay_sd_factor,
            "Node decay": node_decay_factor,
        }

    # endregion

    # TODO: having this "additional fields" thing is a bit of a mess, but it works for now.
    #  Eventually I need to consolidate this, make it resemble Spec, and decide where it should live.
    @property
    def _model_spec(self) -> Dict:
        spec = super()._model_spec
        spec.update(self._model_spec_additional_fields)
        return spec


def _load_labels_from_corpus(corpus: CorpusMetadata, n_words: int) -> Dict[ItemIdx, ItemLabel]:
    return _load_labels(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))


def _load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning: Length) -> Graph:

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


# TODO: essentially repeated code
class LinguisticOneHopPropagator(LinguisticPropagator):
    """A LinguisticPropagator which allows only hops from the initial nodes."""
    def __init__(self,
                 length_factor: int,
                 n_words: int,
                 distributional_model: DistributionalSemanticModel,
                 distance_type: Optional[DistanceType],
                 node_decay_factor: float,
                 edge_decay_sd_factor: float,
                 edge_pruning_type: Optional[EdgePruningType],
                 edge_pruning: Optional[Length],
                 ):

        super().__init__(
            length_factor=length_factor,
            n_words=n_words,
            distributional_model=distributional_model,
            distance_type=distance_type,
            node_decay_factor=node_decay_factor,
            edge_decay_sd_factor=edge_decay_sd_factor,
            edge_pruning_type=edge_pruning_type,
            edge_pruning=edge_pruning,
            )

        # region Resettable

        # Prevent additional impulses being created
        self._block_firing: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_firing = False

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        if self._block_firing:
            return
        else:
            super().schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_firing = True
        return model_events
