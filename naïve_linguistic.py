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
from collections import defaultdict
from os import path
from pathlib import Path
from typing import List, Dict

from numpy import median

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel, VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.maths import DistanceType
from model.basic_types import ItemLabel
from model.graph import iter_edges_from_edgelist
from model.linguistic_component import load_labels_from_corpus
from model.naïve import NaïveModelComponent, logger
from model.utils.maths import distance_from_similarity
from preferences import Preferences


class LinguisticNaïveModelComponent(NaïveModelComponent, ABC):

    def __init__(self, length_factor: int, n_words: int,
                 distributional_model: DistributionalSemanticModel):
        words: List[ItemLabel] = FreqDist.load(distributional_model.corpus_meta.freq_dist_path)\
            .most_common_tokens(n_words)
        self.distributional_model: DistributionalSemanticModel = distributional_model
        super().__init__(length_factor=length_factor, words=words,
                         idx2label=load_labels_from_corpus(distributional_model.corpus_meta, n_words))

    @property
    @abstractmethod
    def _graph_filename(self) -> str:
        """The name of the file in which the graph is stored. (Not the full path.)"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def _distributions_filename(self) -> str:
        """The name of the file in which to store the edge length distributions. (Not the full path.)"""
        raise NotImplementedError()

    @property
    def _distributions_path(self) -> Path:
        return Path(Preferences.graphs_dir, self._distributions_filename)

    def __load_distance_distributions(self) -> Dict[ItemLabel, List[Length]]:
        distributions: Dict[ItemLabel, List[Length]] = defaultdict(list)
        with self._distributions_path.open(mode="r", encoding="utf-8") as distributions_file:
            for line in distributions_file:
                label, ds = line.split(":")
                distributions[label].extend([int(d) for d in ds.split(",")])
        return distributions

    def __save_distance_distributions(self, distributions: Dict[ItemLabel, List[Length]]):
        with self._distributions_path.open(mode="r", encoding="utf-8") as distributions_file:
            for l, ds in distributions.items():
                line = f"{l}:"
                line += ",".join([str(d) for d in ds])
                line += "\n"

    def __median_distances_from_words(self, words: List[ItemLabel]) -> Dict[ItemLabel, Length]:
        if self._distributions_path.exists():
            logger.info("Loading median distances")
            distributions: Dict[ItemLabel, List[Length]] = self.__load_distance_distributions()
        else:
            logger.info("Computing median distances")
            distributions: Dict[ItemLabel, List[Length]] = defaultdict(list)
            for i, (edge, length) in enumerate(iter_edges_from_edgelist(path.join(Preferences.graphs_dir, self._graph_filename)), start=1):
                for word in words:
                    if self.label2idx[word] in edge:
                        distributions[word].append(length)
                if i % 1000 == 0:
                    logger.info(f"Done {i:,} edges.")
            self.__save_distance_distributions(distributions)
        return {
            label: Length(median(lengths))
            for label, lengths in distributions.items()
        }


class LinguisticVectorNaïveModel(LinguisticNaïveModelComponent):

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

    @property
    def _distributions_filename(self) -> str:
        return f"{self.distributional_model.name} {self.distance_type.name} {self._n_words} words length {self.length_factor}.distributions"


class LinguisticNgramNaïveModel(LinguisticNaïveModelComponent):

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

    @property
    def _distributions_filename(self) -> str:
        return f"{self.distributional_model.name} {self._n_words} words length {self.length_factor}.distributions"
