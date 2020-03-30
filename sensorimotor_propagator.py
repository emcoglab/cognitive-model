from __future__ import annotations

from os import path
from typing import Dict, List

from ldm.utils.maths import DistanceType
from model.basic_types import ItemIdx, ItemLabel, Node, ActivationValue
from model.events import ModelEvent
from model.graph import Graph
from model.graph_propagator import GraphPropagator, _load_labels, IMPULSE_PRUNING_THRESHOLD
from model.utils.logging import logger
from model.utils.maths import make_decay_function_lognormal
from preferences import Preferences


class SensorimotorPropagator(GraphPropagator):
    """
    Propagate activation by expanding spheres through space, where spheres have a maximum radius.
    Implemented by using the underlying graph of connections between points which are mutually within the maximum sphere
    radius.
    """

    # region __init__

    def __init__(self,
                 length_factor: int,
                 distance_type: DistanceType,
                 max_sphere_radius: int,
                 node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float,
                 use_prepruned: bool = False,
                 ):
        """
        :param distance_type:
            The metric used to determine distances between points.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param max_sphere_radius:
            What is the maximum radius of a sphere
        :param node_decay_lognormal_median:
            The node_decay_median of the lognormal decay.
        :param node_decay_lognormal_sigma:
            The node_decay_sigma parameter for the lognormal decay.
        :param use_prepruned:
            Whether to use the prepruned graphs or do pruning on load.
            Only to be used for testing purposes.
        """

        # region Validation

        # max_sphere_radius == 0 would be degenerate: no item can ever activate any other item.
        assert (max_sphere_radius > 0)
        # node_decay_lognormal_sigma or node_decay_lognormal_median == 0 will probably cause a division-by-zero error, and anyway is
        # degenerate: it causes everything to decay to 0 activation in a single tick.
        assert (node_decay_lognormal_median > 0)
        assert (node_decay_lognormal_sigma > 0)

        # endregion

        # Load graph
        idx2label = _load_labels_from_sensorimotor()
        super().__init__(
            graph=_load_graph(distance_type, length_factor, max_sphere_radius, use_prepruned, idx2label),
            idx2label=idx2label,
            node_decay_function=make_decay_function_lognormal(median=node_decay_lognormal_median, sigma=node_decay_lognormal_sigma),
            # Once pruning has been done, we don't need to decay activation in edges, as target items should receive the
            # full activations of their source items at the time they were last activated.
            # The maximal sphere radius is achieved by the initial graph pruning.
            edge_decay_function=None,
            # Impulses reach their destination iff their destination is within the max sphere radius.
            # The max sphere radius is baked into the underlying graph.
            # However we add a minute threshold here in case activation has been modulated down to zero
            impulse_pruning_threshold=IMPULSE_PRUNING_THRESHOLD,
        )

        # region Set once
        # These fields are set on first init and then don't need to change even if .reset() is used.

        self._model_spec.update({
            "Distance type": distance_type.name,
            "Length factor": length_factor,
            "Max sphere radius": max_sphere_radius,
            "Log-normal median": node_decay_lognormal_median,
            "Log-normal sigma": node_decay_lognormal_sigma,
        })

    # endregion


def _load_labels_from_sensorimotor() -> Dict[ItemIdx, ItemLabel]:
    return _load_labels(path.join(Preferences.graphs_dir, "sensorimotor words.nodelabels"))


def _load_graph(distance_type, length_factor, max_sphere_radius, use_prepruned, node_labelling_dictionary) -> Graph:
    if use_prepruned:
        logger.warning("Using pre-pruned graph. THIS SHOULD BE USED FOR TESTING PURPOSES ONLY!")

        edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {max_sphere_radius}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph: Graph = Graph.load_from_edgelist(file_path=edgelist_path, with_feedback=True)

        # nodes which got removed from the edgelist because all their edges got pruned
        for i, w in node_labelling_dictionary.items():
            sensorimotor_graph.add_node(Node(i))

    else:

        edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph: Graph = Graph.load_from_edgelist(file_path=edgelist_path,
                                                      ignore_edges_longer_than=max_sphere_radius)
    return sensorimotor_graph


class SensorimotorOneHopPropagator(SensorimotorPropagator):
    """A SensorimotorPropagator which allows only hops from the initial nodes."""
    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 max_sphere_radius: int,
                 node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float,
                 use_prepruned: bool = False,
                 ):

        super().__init__(
            distance_type=distance_type,
            length_factor=length_factor,
            max_sphere_radius=max_sphere_radius,
            node_decay_lognormal_median=node_decay_lognormal_median,
            node_decay_lognormal_sigma=node_decay_lognormal_sigma,
            use_prepruned=use_prepruned,
            )

        # region Resettable

        # Prevent additional impulses being created
        self._block_firing: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_firing = False

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        if self._block_firing:
            return
        else:
            super().schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_firing = True
        return model_events
