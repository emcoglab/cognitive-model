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
from typing import Dict
from logging import getLogger

from numpy import array, percentile
from scipy.sparse import issparse
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel, VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.lists import chunks
from ldm.utils.maths import DistanceType
from model.basic_types import ItemLabel
from model.linguistic_component import load_labels_from_corpus
from model.naïve import NaïveModelComponent
from model.utils.maths import distance_from_similarity

logger = getLogger(__name__)


SPARSE_BATCH_SIZE = 1_000


class LinguisticNaïveModelComponent(NaïveModelComponent, ABC):


    def __init__(self, n_words: int, distributional_model: DistributionalSemanticModel):
        self._distributional_model: DistributionalSemanticModel = distributional_model

        # cache for median distances
        self.__median_distances: Dict[ItemLabel, float] = dict()

        super().__init__(
            words=FreqDist.load(distributional_model.corpus_meta.freq_dist_path).most_common_tokens(n_words),
            idx2label=load_labels_from_corpus(distributional_model.corpus_meta, n_words))

    def median_distance_from(self, word: ItemLabel) -> float:
        if word not in self.__median_distances:
            self.__median_distances[word] = self._compute_median_distance_from(word)
        return self.__median_distances[word]

    @abstractmethod
    def _compute_median_distance_from(self, word: ItemLabel) -> float:
        raise NotImplementedError()


class LinguisticVectorNaïveModel(LinguisticNaïveModelComponent):

    def __init__(self, n_words: int, distributional_model: VectorSemanticModel,
                 distance_type: DistanceType):
        self.distance_type: DistanceType = distance_type
        super().__init__(distributional_model=distributional_model, n_words=n_words)

    def distance_between(self, word_1, word_2) -> float:
        self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, VectorSemanticModel)
        return self._distributional_model.distance_between(word_1, word_2, self.distance_type)

    def _compute_median_distance_from(self, word: ItemLabel) -> float:
        assert isinstance(self._distributional_model, VectorSemanticModel)
        self._distributional_model.train(memory_map=True)
        word_vector: array = self._distributional_model.vector_for_word(word)
        word_vector: array = word_vector.reshape(1, len(word_vector))

        distances: array
        if issparse(self._distributional_model._model):
            # can't do pdists for sparse matrices
            # can't convert self.model to dense as it's BIG (up to 50k x 10M)
            # so chunk self.model up and convert each chunk to dense
            ds = []
            for chunk in chunks(range(self._distributional_model._model.shape[0]), SPARSE_BATCH_SIZE):
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
                distances = distance_matrix(word_vector, self._distributional_model._model, metric=self.distance_type.name)
            elif self.distance_type == DistanceType.Minkowski3:
                distances = minkowski_distance_matrix(word_vector, self._distributional_model._model, 3)
            else:
                raise NotImplementedError()

        return percentile(distances, 50)


class LinguisticNgramNaïveModel(LinguisticNaïveModelComponent):

    def __init__(self, n_words: int, distributional_model: NgramModel):
        super().__init__(distributional_model=distributional_model, n_words=n_words)
        logger.info("Finding minimum and maximum values")
        # Set the max value for later turning associations into distances.
        # We will rarely need the whole model in memory, so we load it once for computing the max, then unload it.
        self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, NgramModel)
        # This is the same calculation as is used in save_edgelist_from_similarity
        # (i.e. without filtering the matrix first).
        self._max_value = self._distributional_model.underlying_count_model.matrix.data.max()
        self._min_value = self._distributional_model.underlying_count_model.matrix.data.min()
        assert self._min_value > 0  # make sure zeros were eliminated
        self._distributional_model.untrain()

    def distance_between(self, word_1, word_2) -> float:
        self._distributional_model.train(memory_map=True)
        assert isinstance(self._distributional_model, NgramModel)
        return distance_from_similarity(
            self._distributional_model.association_between(word_1, word_2),
            self._max_value, self._min_value)

    def _compute_median_distance_from(self, word: ItemLabel) -> float:
        assert isinstance(self._distributional_model, NgramModel)
        similarities: array = self._distributional_model.underlying_count_model.vector_for_word(word)
        distances: array = distance_from_similarity(similarities,
                                                    min_similarity=self._min_value, max_similarity=self._max_value)
        return percentile(distances, 50)
