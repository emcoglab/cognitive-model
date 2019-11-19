"""
===========================
Naïve models (no spreading activation.
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
from collections import defaultdict
from os import path
from typing import List, Dict

from numpy import median

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel, VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.maths import DistanceType, distance
from model.basic_types import ItemLabel, Length, ItemIdx
from model.graph import iter_edges_from_edgelist
from model.utils.maths import distance_from_similarity
from model.linguistic_component import load_labels_from_corpus
from model.sensorimotor_component import load_labels_from_sensorimotor
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


class NaïveModel(ABC):
    def __init__(self,
                 length_factor: int,
                 words: List[ItemLabel],
                 idx2label: Dict[ItemIdx, ItemLabel],
                 ):
        self.length_factor: int = length_factor
        self.words: List[ItemLabel] = words
        self._n_words: int = len(words)
        self.idx2label: Dict[ItemIdx, ItemLabel] = idx2label
        self.median_distances: Dict[ItemLabel, Length] = self.__median_distances_from_words(words)

    @property
    @abstractmethod
    def _graph_filename(self) -> str:
        """The name of the file in which the graph is stored. (Not the full path.)"""
        raise NotImplementedError()

    def __median_distances_from_words(self, words: List[ItemLabel]) -> Dict[ItemLabel, Length]:
        distributions: Dict[ItemLabel, List[Length]] = defaultdict(list)
        for edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, self._graph_filename)):
            for i in words:
                if i in edge:
                    distributions[i].append(length)
        return {
            label: Length(median(lengths))
            for label, lengths in distributions.items()
        }

    def is_hit(self, source, target) -> bool:
        """
        Hit when target is < median of distances from source
        :param source:
        :param target:
        :return:
        :raises LookupError
        """
        if source not in self.words:
            raise LookupError(source)
        if target not in self.words:
            raise LookupError(target)
        return self.distance_between(source, target) < self.median_distances[source]

    @abstractmethod
    def distance_between(self, word_1, word_2) -> float:
        """
        Compute distance between two words.
        :param word_1:
        :param word_2:
        :return:
        :raises LookupError
        """
        raise NotImplementedError()


class SensorimotorNaïveModel(NaïveModel):
    def __init__(self, length_factor: int, distance_type: DistanceType):
        self._sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()
        self.distance_type: DistanceType = distance_type
        super().__init__(length_factor=length_factor,
                         words=list(self._sensorimotor_norms.iter_words()), idx2label=load_labels_from_sensorimotor())

    def _graph_filename(self) -> str:
        # Copied from SensorimotorComponent
        return f"sensorimotor for testing only {self.distance_type.name} distance length {self.length_factor}.edgelist"

    def distance_between(self, word_1, word_2) -> float:
        return distance(
            self._sensorimotor_norms.vector_for_word(word_1),
            self._sensorimotor_norms.vector_for_word(word_2),
            self.distance_type)


class LinguisticNaïveModel(NaïveModel, ABC):

    def __init__(self, length_factor: int, n_words: int,
                 distributional_model: DistributionalSemanticModel):
        words: List[ItemLabel] = FreqDist.load(distributional_model.corpus_meta.freq_dist_path)\
            .most_common_tokens(n_words)
        self.distributional_model: DistributionalSemanticModel = distributional_model
        super().__init__(length_factor=length_factor, words=words,
                         idx2label=load_labels_from_corpus(distributional_model.corpus_meta, n_words))


class LinguisticVectorNaïveModel(LinguisticNaïveModel):

    def __init__(self,
                 n_words: int,
                 distance_type: DistanceType,
                 length_factor: int,
                 distributional_model: VectorSemanticModel):
        self.distance_type: DistanceType = distance_type
        super().__init__(length_factor=length_factor, distributional_model=distributional_model, n_words=n_words)

    def distance_between(self, word_1, word_2) -> float:
        self.distributional_model.train(memory_map=True)
        assert isinstance(self.distributional_model, VectorSemanticModel)
        return self.distributional_model.distance_between(word_1, word_2, self.distance_type)

    @property
    def _graph_filename(self) -> str:
        # Copied from LinguisticComponent
        return f"{self.distributional_model.name} {self.distance_type.name} {self._n_words} words length {self.length_factor}.edgelist"


class LinguisticNgramNaïveModel(LinguisticNaïveModel):

    def __init__(self, n_words: int, length_factor: int, distributional_model: NgramModel):
        super().__init__(length_factor=length_factor, distributional_model=distributional_model, n_words=n_words)
        # Set the max value for later turning associations into distances.
        # We will rarely need the whole model in memory, so we load it once for computing the max, then unload it.
        self.distributional_model.train(memory_map=True)
        assert isinstance(self.distributional_model, NgramModel)
        # This is the same calculation as is used in save_edgelist_from_similarity
        # (i.e. without filtering the matrix first).
        self._max_value = self.distributional_model.underlying_count_model.matrix.data.max()
        self._min_value = self.distributional_model.underlying_count_model.matrix.data.min()
        assert self._min_value > 0  # make sure zeros were eliminated
        self.distributional_model.untrain()

    def distance_between(self, word_1, word_2) -> float:
        self.distributional_model.train(memory_map=True)
        assert isinstance(self.distributional_model, NgramModel)
        return distance_from_similarity(
            self.distributional_model.association_between(word_1, word_2),
            self._max_value, self._min_value)

    @property
    def _graph_filename(self) -> str:
        # Copied from LinguisticComponent
        return f"{self.distributional_model.name} {self._n_words} words length {self.length_factor}.edgelist"
