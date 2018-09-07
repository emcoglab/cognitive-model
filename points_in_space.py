"""
===========================
Points in space.
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

from typing import Optional, Iterator, List

from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cdist

from ldm.core.utils.maths import DistanceType
from model.component import ItemIdx


class DimensionalityError(Exception):
    pass


PointIdx = ItemIdx


class Point:
    """Immutable, hashable labelled vector."""

    __slots__ = 'vector', 'idx'

    def __init__(self, idx: PointIdx, vector: ndarray):
        self.idx: PointIdx = idx
        self.vector: ndarray = vector
        # self.vector is immutable
        self.vector.flags.writeable = False

    def __hash__(self):
        return hash((self.vector, self.idx))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class PointsInSpace:
    """
    Collection of points in a high-dimensional real vector space.
    """
    def __init__(self, data_matrix: ndarray):

        # An n_points x n_dims data matrix.
        self.data_matrix: ndarray = data_matrix

    @property
    def n_points(self) -> int:
        return 0 if self.data_matrix is None else self.data_matrix.shape[0]

    @property
    def n_dims(self) -> Optional[int]:
        """
        The number of dimensions in the space in which the points live.
        None if there are no points.
        """
        return None if self.data_matrix is None else self.data_matrix.shape[1]

    def iter_points(self) -> Iterator[Point]:
        for idx in range(self.n_points):
            vector = self.data_matrix[idx, :]
            yield Point(idx, vector)

    def point_with_idx(self, idx: int) -> Point:
        return Point(idx, self.data_matrix[idx, :])

    def _distances_to_point_with_idx(self, point_idx: PointIdx, distance_type: DistanceType) -> ndarray:
        if distance_type is DistanceType.cosine:
            distances = cdist(self.data_matrix,
                              self.point_with_idx(point_idx).vector,
                              metric=distance_type.name)
        else:
            raise NotImplementedError()
        return distances

    def points_between_spheres(self, centre_idx: PointIdx,
                               outer_radius: float, inner_radius: float,
                               distance_type: DistanceType) -> List[PointIdx]:
        """
        Points newly captured within a growing sphere.
        :param centre_idx:
        :param outer_radius:
            Includes outer boundary: Points will have from-centre distance <= this radius.
        :param inner_radius:
            Does not include inner boundary: Points will have from-centre distance > this radius.
        :param distance_type:
        :return:
        """
        distances = self._distances_to_point_with_idx(centre_idx, distance_type)
        contained_indices = [i for i, d in enumerate(distances) if inner_radius < d <= outer_radius]
        return contained_indices
