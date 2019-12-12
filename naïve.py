"""
===========================
Naïve models (no spreading activation).
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
from logging import getLogger
from typing import List, Dict

from model.basic_types import ItemLabel, ItemIdx

logger = getLogger(__name__)


class DistanceOnlyModelComponent(ABC):
    def __init__(self,
                 words: List[ItemLabel],
                 idx2label: Dict[ItemIdx, ItemLabel]):
        self.words: List[ItemLabel] = words
        self._n_words: int = len(words)
        self.idx2label: Dict[ItemIdx, ItemLabel] = idx2label
        self.label2idx: Dict[ItemLabel, ItemIdx] = {v: k for k, v in idx2label.items()}

    def is_hit(self, source, target) -> bool:
        """
        Hit when target is < median of distances from source.

        :raises LookupError
        """
        if source not in self.words:
            raise LookupError(source)
        if target not in self.words:
            raise LookupError(target)
        return self.distance_between(source, target) < self.median_distance_from(source)

    @abstractmethod
    def median_distance_from(self, word: ItemLabel) -> float:
        """
        :raises LookupError
        """
        raise NotImplementedError()

    @abstractmethod
    def distance_between(self, word_1: ItemLabel, word_2: ItemLabel) -> float:
        """
        Compute distance between two words.

        :raises LookupError
        """
        raise NotImplementedError()


