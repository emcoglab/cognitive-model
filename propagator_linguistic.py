from os import path
from typing import Dict, Optional

from pandas import DataFrame

from .ldm.corpus.corpus import CorpusMetadata
from .ldm.model.base import LinguisticDistributionalModel
from .ldm.utils.maths import DistanceType
from .basic_types import Length, ItemIdx, ItemLabel, Component
from .utils.log import logger
from .graph import Graph, EdgePruningType
from .propagator import GraphPropagator, _load_labels, IMPULSE_PRUNING_THRESHOLD
from .decay_functions import make_decay_function_exponential_with_decay_factor, make_decay_function_gaussian_with_sd
from .preferences.preferences import Preferences


class LinguisticPropagator(GraphPropagator):
    """
    Spreading activation on a graph over time.
    Nodes have a firing threshold and an activation cap.
    """

    def __init__(self,
                 length_factor: int,
                 n_words: int,
                 distributional_model: LinguisticDistributionalModel,
                 distance_type: Optional[DistanceType],
                 node_decay_factor: float,
                 edge_decay_sd: float,
                 edge_pruning_type: Optional[EdgePruningType],
                 edge_pruning: Optional[Length],
                 shelf_life: Optional[int] = None,
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
        :param edge_decay_sd:
            The SD of the gaussian curve governing edge decay.
            Scale is the source distance, not the quantised Length.
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
        super().__init__(
            graph=_load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning),
            idx2label=idx2label,
            impulse_pruning_threshold=IMPULSE_PRUNING_THRESHOLD,
            node_decay_function=make_decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=make_decay_function_gaussian_with_sd(
                sd=edge_decay_sd * length_factor),
            component=Component.linguistic,
            shelf_life=shelf_life
        )


def _load_labels_from_corpus(corpus: CorpusMetadata, n_words: int) -> Dict[ItemIdx, ItemLabel]:
    return _load_labels(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))


def _load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning: Length) -> Graph:

    # Check if distance_type is needed and get filename
    if distributional_model.model_type.metatype is LinguisticDistributionalModel.MetaType.ngram:
        assert distance_type is None
        graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"
    else:
        assert distance_type is not None
        graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # First try pickle
    import pickle
    pickle_filename = f"{graph_file_name[:-9]}.pickle"  # swap .edgelist for .pickle
    pickle_path = path.join(Preferences.graphs_dir, pickle_filename)
    try:
        logger.info(f"Attempting to load pickled graph from {pickle_path}")
        return Graph.load_from_pickle(pickle_path)
    except FileNotFoundError:
        logger.info(f"Couldn't find pickle file, falling back to edgelist [{pickle_path}]")
    except pickle.UnpicklingError as e:
        logger.warning(f"Pickled graph appears to be broken. Consider deleting it. Falling back to edgelist [{pickle_path}]: {repr(e)}")

    # Load graph
    if edge_pruning is None:
        logger.info(f"Loading graph from {graph_file_name}")
        graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                         with_feedback=True)

    elif edge_pruning_type is EdgePruningType.Length:
        logger.info(f"Loading graph from {graph_file_name}, pruning any edge longer than {edge_pruning}")
        graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                         ignore_edges_longer_than=edge_pruning,
                                         keep_at_least_n_edges=Preferences.min_edges_per_node,
                                         with_feedback=True)

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
                                         keep_at_least_n_edges=Preferences.min_edges_per_node,
                                         with_feedback=True)

    elif edge_pruning_type is EdgePruningType.Importance:
        logger.info(
            f"Loading graph from {graph_file_name}, pruning longest {edge_pruning}% of edges")
        graph = Graph.load_from_edgelist_with_importance_pruning(
            file_path=path.join(Preferences.graphs_dir, graph_file_name),
            ignore_edges_with_importance_greater_than=edge_pruning,
            keep_at_least_n_edges=Preferences.min_edges_per_node)

    else:
        raise NotImplementedError()

    if not path.isfile(pickle_path):
        logger.info(f"Saving pickled version for faster loading next time")
        graph.save_as_pickle(pickle_path)

    return graph
