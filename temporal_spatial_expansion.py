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
from typing import Dict, Set, Optional, Iterator

import numpy
from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cdist

from ldm.core.utils.maths import DistanceType

Label = str


class DimensionalityError(Exception):
    pass


class Point:
    def __init__(self, vector: ndarray, label: Label):
        self.vector: ndarray = vector
        self.label: Label = label

    def __repr__(self):
        return f"Point( {self.vector}, \"{self.label}\" )"


class PointsInSpace:
    """
    Collection of labelled points in a high-dimensional real vector space.
    """
    def __init__(self, data_matrix: ndarray, labelling_dictionary: Dict[int, Label]):
        n_points_from_dict = len(labelling_dictionary)
        assert data_matrix.shape[0] == n_points_from_dict
        assert set(labelling_dictionary.keys()) == set(range(n_points_from_dict))

        self.idx2label: Dict[int, Label] = labelling_dictionary
        self.label2idx: Dict[Label, int] = {v: k for k, v in labelling_dictionary.items()}

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

    @property
    def labels(self) -> Set[Label]:
        return set(self.label2idx.keys())

    def iter_points(self) -> Iterator[Point]:
        for idx in range(self.n_points):
            vector = self.data_matrix[idx, :]
            label = self.idx2label[idx]
            yield Point(vector, label)

    def point_with_label(self, label: Label) -> Point:
        try:
            idx = self.label2idx[label]
        except KeyError:
            raise KeyError(f"No point with label {label}")

        return Point(self.data_matrix[idx, :], label)


class TemporalSpatialExpansion:
    def __init__(self, points: PointsInSpace):
        self.points: PointsInSpace = points

    def distances_to_point(self, p: Point, distance_type: DistanceType) -> ndarray:
        if distance_type is DistanceType.cosine:
            distances = cdist(self.points.data_matrix, p.vector, metric=distance_type.name)
        else:
            raise NotImplementedError()
        return distances


if __name__ == '__main__':

    pis = PointsInSpace()
    tse = TemporalSpatialExpansion(pis)
