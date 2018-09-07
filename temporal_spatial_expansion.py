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
from typing import List, Dict, Set, Tuple

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
                 decay_threshold: ActivationValue,
                 conscious_access_threshold: ActivationValue):

        self.points: PointsInSpace = points_in_space
        self.expansion_rate: float = expansion_rate
        self.max_radius: float = max_radius
        self.distance_type: DistanceType = distance_type

        # Use >= and < to test for above/below
        self.decay_threshold: ActivationValue = decay_threshold
        self.conscious_access_threshold: ActivationValue = conscious_access_threshold

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
        return [self.points.point_with_idx(i) for i in contained_indices]

    def tick(self) -> Set[ItemActivatedEvent]:

        self.clock += 1

        self._decay_activations()

        points_which_became_consciously_active = self._grow_spheres()

        return set(ItemActivatedEvent(point.label, self.activations[point], self.clock) for point in points_which_became_consciously_active)

    def _grow_spheres(self) -> Set[Point]:
        """
        Radiates spheres
        :return:
            Set of points which became consciously active.
        """

        # TODO: This could be made more clever and efficient by working out which points will be reached when at the
        # TODO: time the sphere is created, and just looking that up here.

        points_which_activated = set()
        points_which_crossed_c_a_t = set()

        for centre, old_radius in self.spheres.items():
            new_radius = old_radius + self.expansion_rate

            if new_radius > self.max_radius:
                self.spheres.pop(centre)
            else:
                self.spheres[centre] = new_radius

            # activate points within sphere
            for reached_point in self._points_newly_within_sphere(centre, outer_radius=new_radius, inner_radius=old_radius):
                # Pass on full activation when reached
                point_did_activate = self.activate_point(reached_point, self.activations[centre].activation)
                if point_did_activate:
                    points_which_activated.add(reached_point)

        return points_which_crossed_c_a_t

    def _decay_activations(self):

        # Since activations will be computed according to need using the decay function, decaying them just involves
        # incrementing the clock.

        # However we need to remove concepts from the buffer if they have decayed too much
        for activated_point, activation_record in self.activations.items():
            age = self.clock - activation_record.time_activated
            decayed_activation = self._decay_function(age, activation_record.activation)
            if decayed_activation < self.decay_threshold:
                self.activations.pop(activated_point)

    def activate_point(self, point: Point, incoming_activation: ActivationValue) -> Tuple[bool, bool]:
        """
        Activate a point.
        :param point:
        :param incoming_activation:
        :return:
            Tuple of bools:
            (
                Point became newly activated (True) or just absorbed and accumulated (False),
                Point crossed conscious access threshold (True) or not (False)
            )
        """

        # Create sphere if not already activated
        if point not in self.activations.keys():
            new_activation = incoming_activation
            self.spheres[point] = 0
            did_activate = True
            # Since the point started at zero activation, we crossed the c_a_t iff the new activation is greater than it
            did_cross_conscious_access_threshold = incoming_activation > self.conscious_access_threshold

        # Otherwise accumulate incoming_activation, but reset clock
        else:
            current_activation = self.activations[point].activation
            currently_below_c_a_t = current_activation > self.conscious_access_threshold
            new_activation = current_activation + incoming_activation
            did_activate = False
            did_cross_conscious_access_threshold = currently_below_c_a_t and (new_activation > self.conscious_access_threshold)

        self.activations[point] = ActivationRecord(new_activation, self.clock)

        return did_activate, did_cross_conscious_access_threshold

    def activate_point_with_label(self, label: str, activation: ActivationValue):
        self.activate_point(self.points.point_with_label(label), activation)
