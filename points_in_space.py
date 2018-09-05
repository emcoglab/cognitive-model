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

from typing import Dict, Optional, Set, Iterator

from numpy.core.multiarray import ndarray

Label = str


class DimensionalityError(Exception):
    pass


class Point:
    """Immutable, hashable labelled vector."""

    __slots__ = 'vector', 'label'

    def __init__(self, label: Label, vector: ndarray):
        self.label: Label = label
        self.vector: ndarray = vector
        # self.vector is immutable
        self.vector.flags.writeable = False

    def __hash__(self):
        return hash((self.vector, self.label))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return f"Point(label=\"{self.label}\", vector={self.vector})"

    def __str__(self):
        return repr(self)


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

    def iter_labels(self) -> Iterator[Label]:
        for p in self.iter_points():
            yield p.label

    def iter_points(self) -> Iterator[Point]:
        for idx in range(self.n_points):
            vector = self.data_matrix[idx, :]
            label = self.idx2label[idx]
            yield Point(label, vector)

    def point_with_label(self, label: Label) -> Point:
        try:
            idx = self.label2idx[label]
        except KeyError:
            raise KeyError(f"No point with label {label}")

        return Point(label, self.data_matrix[idx, :])