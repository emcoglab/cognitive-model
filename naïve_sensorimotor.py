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
from typing import Dict, List, Optional

from numpy import array, percentile
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.utils.maths import DistanceType, distance
from model.events import ModelEvent
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

from model.basic_types import ItemLabel, ItemIdx, ActivationValue
from model.naïve import DistanceOnlyModelComponent
from model.sensorimotor_component import load_labels_from_sensorimotor, SensorimotorComponent, NormAttenuationStatistic


class SensorimotorOneHopComponent(SensorimotorComponent):
    """A SensorimotorComponent which allows only hops from the initial nodes."""
    def __init__(self, distance_type: DistanceType, length_factor: int, max_sphere_radius: int, lognormal_median: float,
                 lognormal_sigma: float, buffer_capacity: Optional[int], accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue, accessible_set_threshold: ActivationValue,
                 activation_cap: ActivationValue, norm_attenuation_statistic: NormAttenuationStatistic,
                 use_prepruned: bool):

        super().__init__(distance_type, length_factor, max_sphere_radius, lognormal_median, lognormal_sigma,
                         buffer_capacity, accessible_set_capacity, buffer_threshold, accessible_set_threshold,
                         activation_cap, norm_attenuation_statistic, use_prepruned)

        # region Resettable

        # Prevent additional impulses being created
        self._block_new_impulses: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_new_impulses = False

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        if self._block_new_impulses:
            return
        else:
            super().schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def scheduled_activation_count(self) -> int:
        return sum([1
                    for tick, schedule_activation in self._scheduled_activations.items()
                    for idx, activation in schedule_activation.items()
                    if activation > 0])

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_new_impulses = True
        return model_events


class SensorimotorDistanceOnlyModelComponent(DistanceOnlyModelComponent):
    def __init__(self, quantile: float, distance_type: DistanceType):
        self._sensorimotor_norms: SensorimotorNorms = SensorimotorNorms()
        self.distance_type: DistanceType = distance_type

        # cache for quantile distances
        self.__quantile_distances: Dict[ItemLabel, float] = dict()

        super().__init__(quantile=quantile,
                         words=list(self._sensorimotor_norms.iter_words()),
                         idx2label=load_labels_from_sensorimotor())

    def distance_between(self, word_1: ItemLabel, word_2: ItemLabel) -> float:
        return distance(
            self._sensorimotor_norms.vector_for_word(word_1),
            self._sensorimotor_norms.vector_for_word(word_2),
            self.distance_type)

    def quantile_distance_from(self, word: ItemLabel) -> float:
        if word not in self.__quantile_distances:
            self.__quantile_distances[word] = self._compute_quantile_distance_from(word)
        return self.__quantile_distances[word]

    def _compute_quantile_distance_from(self, word: ItemLabel) -> float:
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

        return percentile(distances, self.quantile * 100)
