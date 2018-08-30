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

from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cdist

from ldm.core.utils.maths import DistanceType
from model.points_in_space import Point, PointsInSpace
from model.sensorimotor import SensorimotorNorms


class TemporalSpatialExpansion:
    def __init__(self, pis: PointsInSpace):
        self.points: PointsInSpace = pis

    def distances_to_point(self, p: Point, distance_type: DistanceType) -> ndarray:
        if distance_type is DistanceType.cosine:
            distances = cdist(self.points.data_matrix, p.vector, metric=distance_type.name)
        else:
            raise NotImplementedError()
        return distances


if __name__ == '__main__':

    pis = SensorimotorNorms().as_points_in_space()
    tse = TemporalSpatialExpansion(pis)
