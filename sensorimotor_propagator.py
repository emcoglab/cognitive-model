from __future__ import annotations

from os import path
from typing import Dict, List

from .ldm.utils.maths import DistanceType
from .sensorimotor_norms.breng_translation.dictionary.version import VERSION as SM_BRENG_VERSION
from .basic_types import ItemIdx, ItemLabel, Node, ActivationValue, Component
from .events import ModelEvent
from .graph import Graph
from .graph_propagator import GraphPropagator, _load_labels, IMPULSE_PRUNING_THRESHOLD
from .utils.logging import logger
from .utils.maths import make_decay_function_lognormal
from .preferences.preferences import Preferences


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
                 max_sphere_radius: float,
                 node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float,
                 use_breng_translation: bool,
                 use_prepruned: bool = False,
                 ):
        """
        :param distance_type:
            The metric used to determine distances between points.
        :param length_factor:
            How distances are scaled into connection lengths.
        :param max_sphere_radius:
            What is the maximum radius of a sphere.
            Scale is for the source distance, not the quantised length.
        :param node_decay_lognormal_median:
            The node_decay_median of the lognormal decay.
            Scale is for the source distance, not the quantised length.
        :param node_decay_lognormal_sigma:
            The node_decay_sigma parameter for the lognormal decay.
        :param use_breng_translation:
            Whether to use the BrEng-translated form of the sensorimotor norms.
        :param use_prepruned:
            Whether to use the pre-pruned graphs or do pruning on load.
            Only to be used for testing purposes.
        """

        # region Validation

        # max_sphere_radius == 0 would be degenerate: no item can ever activate any other item.
        assert (max_sphere_radius > 0)
        # node_decay_lognormal_sigma or node_decay_lognormal_median == 0 will probably cause a division-by-zero error,
        # and anyway is degenerate: it causes everything to decay to 0 activation in a single tick.
        assert (node_decay_lognormal_median > 0)
        assert (node_decay_lognormal_sigma > 0)

        # endregion

        # Load graph
        idx2label = _load_labels_from_sensorimotor(use_breng_translation)
        super().__init__(
            graph=_load_graph(distance_type, length_factor, max_sphere_radius, use_prepruned, idx2label),
            idx2label=idx2label,
            node_decay_function=make_decay_function_lognormal(median=node_decay_lognormal_median * length_factor,
                                                              sigma=node_decay_lognormal_sigma),
            # Once pruning has been done, we don't need to decay activation in edges, as target items should receive the
            # full activations of their source items at the time they were last activated.
            # The maximal sphere radius is achieved by the initial graph pruning.
            edge_decay_function=None,
            # Impulses reach their destination iff their destination is within the max sphere radius.
            # The max sphere radius is baked into the underlying graph.
            # However we add a minute threshold here in case activation has been modulated down to zero
            impulse_pruning_threshold=IMPULSE_PRUNING_THRESHOLD,
            component=Component.sensorimotor,
        )


def _load_labels_from_sensorimotor(use_breng_translation: bool) -> Dict[ItemIdx, ItemLabel]:
    if use_breng_translation:
        # TODO: this logic should be centralised somewhere, not just copied and pasted everywhere it's used...
        nodelabels_filename = f"sensorimotor words BrEng v{SM_BRENG_VERSION}.nodelabels"
    else:
        nodelabels_filename = "sensorimotor words.nodelabels"
    return _load_labels(path.join(Preferences.graphs_dir, nodelabels_filename))


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
        sensorimotor_graph: Graph = Graph.load_from_edgelist(
            file_path=edgelist_path,
            ignore_edges_longer_than=max_sphere_radius * length_factor)

    return sensorimotor_graph


class SensorimotorOneHopPropagator(SensorimotorPropagator):
    """A SensorimotorPropagator which allows only hops from the initial nodes."""
    def __init__(self,
                 distance_type: DistanceType,
                 length_factor: int,
                 max_sphere_radius: int,
                 node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float,
                 use_breng_translation: bool = False,
                 use_prepruned: bool = False,
                 ):

        super().__init__(
            distance_type=distance_type,
            length_factor=length_factor,
            max_sphere_radius=max_sphere_radius,
            node_decay_lognormal_median=node_decay_lognormal_median,
            node_decay_lognormal_sigma=node_decay_lognormal_sigma,
            use_prepruned=use_prepruned,
            use_breng_translation=use_breng_translation,
            )

        # region Resettable

        # Prevent additional impulses being created
        self._block_firing: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_firing = False

    def _schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        if self._block_firing:
            return
        else:
            super()._schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_firing = True
        return model_events
