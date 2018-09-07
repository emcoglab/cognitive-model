"""
===========================
Temporal spatial expansion.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
from typing import List, Dict, Set

from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cdist

from ldm.core.utils.maths import DistanceType
from model.common import ItemActivatedEvent, ActivationValue, ActivationRecord
from model.points_in_space import Point, PointsInSpace
from model.utils.math import decay_function_lognormal_median


class TemporalSpatialExpansion:
    def __init__(self,
                 points_in_space: PointsInSpace,
                 expansion_rate: float,
                 max_radius: float,
                 distance_type: DistanceType,
                 decay_median: float,
                 decay_shape: float,
                 decay_threshold: float):

        self.points: PointsInSpace = points_in_space
        self.expansion_rate: float = expansion_rate
        self.max_radius: float = max_radius
        self.distance_type: DistanceType = distance_type
        self.decay_threshold = decay_threshold

        self._decay_function = decay_function_lognormal_median(decay_median, decay_shape)

        # Dictionary point -> activation
        self.activations: Dict[Point, ActivationRecord] = dict()

        # Dictionary point -> radius
        self.spheres = dict()

        # Zero-indexed tick counter.
        self.clock: int = int(0)

    def _distances_to_point(self, p: Point) -> ndarray:
        if self.distance_type is DistanceType.cosine:
            distances = cdist(self.points.data_matrix, p.vector, metric=self.distance_type.name)
        else:
            raise NotImplementedError()
        return distances

    def _points_newly_within_sphere(self, centre: Point, outer_radius: float, inner_radius) -> List[Point]:
        """
        Points newly captured within a growing sphere.
        :param centre:
        :param outer_radius:
            Includes outer boundary: Points will have from-centre distance <= this radius.
        :param inner_radius:
            Does not include inner boundary: Points will have from-centre distance > this radius.
        :return:
        """
        distances = self._distances_to_point(centre)
        contained_indices = [i for i, d in enumerate(distances) if inner_radius < d <= outer_radius]
        return [self.points.point_with_label(self.points.idx2label[i]) for i in contained_indices]

    def tick(self):

        self.clock += 1

        self._decay_activations()

        self._grow_spheres()

    def _grow_spheres(self):
        for centre, old_radius in self.spheres.items():
            new_radius = old_radius + self.expansion_rate

            if new_radius > self.max_radius:
                self.spheres.pop(centre)
            else:
                self.spheres[centre] = new_radius

            # activate points within sphere
            for reached_point in self._points_newly_within_sphere(centre, outer_radius=new_radius, inner_radius=old_radius):
                # Pass on full activation when reached
                self.activate_point(reached_point, self.activations[centre].activation)

    def _decay_activations(self):

        # Since activations will be computed according to need using the decay function, decaying them just involves
        # incrementing the clock.

        # However we need to remove concepts from the buffer if they have decayed too much
        for activated_point, activation_record in self.activations.items():
            age = self.clock - activation_record.time_activated
            decayed_activation = self._decay_function(age, activation_record.activation)
            if decayed_activation < self.decay_threshold:
                self.activations.pop(activated_point)

    def activate_point(self, point: Point, incoming_activation: Activation):

        # Create sphere if not already activated
        if point not in self.activations.keys():
            new_activation = incoming_activation
            self.spheres[point] = 0

        # Otherwise accumulate incoming_activation, but reset clock
        else:
            new_activation = self.activations[point].activation + incoming_activation

        self.activations[point] = ActivationRecord(new_activation, self.clock)
