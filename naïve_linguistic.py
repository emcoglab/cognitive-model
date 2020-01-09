"""
===========================
Naïve linguistic models (no spreading activation).
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
from abc import ABC, abstractmethod
from typing import Dict, List
from logging import getLogger

from numpy import array, percentile, Infinity
from scipy.sparse import issparse
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel, VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.exceptions import WordNotFoundError
from ldm.utils.lists import chunks
from ldm.utils.maths import DistanceType
from model.basic_types import ItemLabel, ActivationValue, ItemIdx
from model.events import ModelEvent
from model.graph import EdgePruningType
from model.linguistic_component import load_labels_from_corpus, LinguisticComponent
from model.naïve import DistanceOnlyModelComponent
from model.utils.maths import distance_from_similarity

logger = getLogger(__name__)


SPARSE_BATCH_SIZE = 1_000


class LinguisticOneHopComponent(LinguisticComponent):
    """A LinguisticComponent which allows only hops from the initial nodes."""
    def __init__(self, n_words: int, distributional_model: DistributionalSemanticModel, length_factor: int,
                 node_decay_factor: float, edge_decay_sd_factor: float, impulse_pruning_threshold: ActivationValue,
                 firing_threshold: ActivationValue, activation_cap: ActivationValue = Infinity,
                 distance_type: DistanceType = None, edge_pruning=None, edge_pruning_type: EdgePruningType = None):
        super().__init__(n_words, distributional_model, length_factor, node_decay_factor, edge_decay_sd_factor,
                         impulse_pruning_threshold, firing_threshold, activation_cap, distance_type, edge_pruning,
                         edge_pruning_type)

        # region Resettable

        # Prevent additional impulses being created
        self._block_firing: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_firing = False

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        # TODO: this suggests it would simplify things to factor out a firing method, which could be blocked separately
        #  to individual scheduling.
        #  Not worth it unless we start adding more baseline models or tinkering with this one.
        if self._block_firing:
            return
        else:
            super().schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_firing = True
        return model_events

    def scheduled_activation_count(self) -> int:
        return sum([1
                    for tick, schedule_activation in self._scheduled_activations.items()
                    for idx, activation in schedule_activation.items()
                    if activation > 0])


class LinguisticDistanceOnlyModelComponent(DistanceOnlyModelComponent, ABC):

    def __init__(self, quantile: float, n_words: int, distributional_model: DistributionalSemanticModel):
        self._distributional_model: DistributionalSemanticModel = distributional_model

        # cache for quantile distances
        self.__quantile_distances: Dict[ItemLabel, float] = dict()

        super().__init__(
            quantile=quantile,
            words=FreqDist.load(distributional_model.corpus_meta.freq_dist_path).most_common_tokens(n_words),
            idx2label=load_labels_from_corpus(distributional_model.corpus_meta, n_words))

    def quantile_distance_from(self, word: ItemLabel) -> float:
        """:raises WordNotFoundError"""
        if word not in self.words:
            raise WordNotFoundError(word)
        if word not in self.__quantile_distances:
            self.__quantile_distances[word] = self._compute_quantile_distance_from(word)
        return self.__quantile_distances[word]

    @abstractmethod
    def _compute_quantile_distance_from(self, word: ItemLabel) -> float:
        raise NotImplementedError()


class LinguisticVectorDistanceOnlyModel(LinguisticDistanceOnlyModelComponent):

    def __init__(self, quantile: float, n_words: int, distributional_model: VectorSemanticModel,
                 distance_type: DistanceType):
        self.distance_type: DistanceType = distance_type
        super().__init__(quantile=quantile, distributional_model=distributional_model, n_words=n_words)

    def distance_between(self, word_1, word_2) -> float:
        """:raises WordNotFoundError"""
        if word_1 not in self.words:
            raise WordNotFoundError(word_1)
        if word_2 not in self.words:
            raise WordNotFoundError(word_2)
        if not self._distributional_model.is_trained:
            self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, VectorSemanticModel)
        return self._distributional_model.distance_between(word_1, word_2, self.distance_type)

    def _compute_quantile_distance_from(self, word: ItemLabel) -> float:
        """:raises WordNotFoundError"""
        if word not in self.words:
            raise WordNotFoundError(word)
        if not self._distributional_model.is_trained:
            self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, VectorSemanticModel)
        word_vector: array = self._distributional_model.vector_for_word(word)
        word_vector: array = word_vector.reshape(1, len(word_vector))

        distances: array
        if issparse(self._distributional_model._model):
            # can't do pdists for sparse matrices
            # can't convert self.model to dense as it's BIG (up to 50k x 10M)
            # so chunk self.model up and convert each chunk to dense
            ds = []
            # Only need to take chunks up to the number of words being considered
            for chunk in chunks(range(self._n_words), SPARSE_BATCH_SIZE):
                model_chunk = self._distributional_model._model[chunk, :].todense()
                if self.distance_type in [DistanceType.cosine, DistanceType.Euclidean, DistanceType.correlation]:
                    distance_chunk = distance_matrix(word_vector, model_chunk, metric=self.distance_type.name)
                elif self.distance_type == DistanceType.Minkowski3:
                    distance_chunk = minkowski_distance_matrix(word_vector, model_chunk, 3)
                else:
                    raise NotImplementedError()
                ds.extend(distance_chunk.squeeze().tolist())
            distances = array(ds)

        else:
            # Can just do regular pdists
            if self.distance_type in [DistanceType.cosine, DistanceType.Euclidean, DistanceType.correlation]:
                distances = distance_matrix(
                    word_vector,
                    # Only consider matrix up to n_words
                    self._distributional_model._model[:self._n_words, :],
                    metric=self.distance_type.name)
            elif self.distance_type == DistanceType.Minkowski3:
                distances = minkowski_distance_matrix(
                    word_vector,
                    # Only consider matrix up to n_words
                    self._distributional_model._model[:self._n_words, :],
                    3)
            else:
                raise NotImplementedError()

        return percentile(distances, self.quantile * 100)


class LinguisticNgramDistanceOnlyModel(LinguisticDistanceOnlyModelComponent):

    def __init__(self, quantile: float, n_words: int, distributional_model: NgramModel):
        super().__init__(quantile=quantile, distributional_model=distributional_model, n_words=n_words)
        logger.info("Finding minimum and maximum values")
        # Set the max value for later turning associations into distances.
        # We will rarely need the whole model in memory, so we load it once for computing the max, then unload it.
        self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, NgramModel)
        # This is the same calculation as is used in save_edgelist_from_similarity
        # (i.e. without filtering the matrix first).
        self._max_value = self._distributional_model.underlying_count_model.matrix.data.max()
        self._min_value = self._distributional_model.underlying_count_model.matrix.data.min()
        assert self._min_value != 0  # make sure zeros were eliminated
        self._distributional_model.untrain()

    def distance_between(self, word_1, word_2) -> float:
        """:raises WordNotFoundError"""
        if word_1 not in self.words:
            raise WordNotFoundError(word_1)
        if word_2 not in self.words:
            raise WordNotFoundError(word_2)
        if not self._distributional_model.is_trained:
            self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, NgramModel)
        return distance_from_similarity(
            self._distributional_model.association_between(word_1, word_2),
            self._max_value, self._min_value)

    def _compute_quantile_distance_from(self, word: ItemLabel) -> float:
        """:raises WordNotFoundError"""
        if word not in self.words:
            raise WordNotFoundError(word)
        assert isinstance(self._distributional_model, NgramModel)
        similarities: array = self._distributional_model.underlying_count_model.vector_for_word(word)
        distances: array = distance_from_similarity(similarities,
                                                    min_similarity=self._min_value, max_similarity=self._max_value)
        return percentile(distances, self.quantile * 100)
