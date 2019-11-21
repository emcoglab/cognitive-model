"""
===========================
Naïve sensorimotor models (no spreading activation).
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
from typing import Dict

from numpy import array, percentile
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.utils.maths import DistanceType, distance
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

from model.basic_types import ItemLabel
from model.naïve import NaïveModelComponent
from model.sensorimotor_component import load_labels_from_sensorimotor


class SensorimotorNaïveModelComponent(NaïveModelComponent):
    def __init__(self, distance_type: DistanceType):
        self._sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()
        self.distance_type: DistanceType = distance_type

        # cache for median distances
        self.__median_distances: Dict[ItemLabel, float] = dict()

        super().__init__(words=list(self._sensorimotor_norms.iter_words()),
                         idx2label=load_labels_from_sensorimotor())

    def distance_between(self, word_1: ItemLabel, word_2: ItemLabel) -> float:
        return distance(
            self._sensorimotor_norms.vector_for_word(word_1),
            self._sensorimotor_norms.vector_for_word(word_2),
            self.distance_type)

    def median_distance_from(self, word: ItemLabel) -> float:
        if word not in self.__median_distances:
            self.__median_distances[word] = self._compute_median_distance_from(word)
        return self.__median_distances[word]

    def _compute_median_distance_from(self, word: ItemLabel) -> float:
        """
        :raises WordNotInNormsError
        """
        word_vector: array = array(self._sensorimotor_norms.vector_for_word(word)).reshape(1, 11)

        distances: array
        if self.distance_type in [DistanceType.cosine, DistanceType.correlation, DistanceType.Euclidean]:
            distances = distance_matrix(word_vector, array(self._sensorimotor_norms.matrix()),
                                        metric=self.distance_type.name)
        elif self.distance_type == DistanceType.Minkowski3:
            distances = minkowski_distance_matrix(word_vector, array(self._sensorimotor_norms.matrix()), 3)
        else:
            raise NotImplementedError()

        return percentile(distances, 50)
