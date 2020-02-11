"""
===========================
Naïve linguistic models (no spreading activation).
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
from logging import getLogger
from typing import List

from numpy import Infinity

from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, ItemIdx
from model.events import ModelEvent
from model.graph import EdgePruningType
from model.linguistic_component import LinguisticComponent

logger = getLogger(__name__)


SPARSE_BATCH_SIZE = 1_000


class LinguisticOneHopComponent(LinguisticComponent):
    """A LinguisticComponent which allows only hops from the initial nodes."""
    def __init__(self, n_words: int, distributional_model: DistributionalSemanticModel, length_factor: int,
                 node_decay_factor: float, edge_decay_sd_factor: float, impulse_pruning_threshold: ActivationValue,
                 firing_threshold: ActivationValue, activation_cap: ActivationValue = Infinity,
                 distance_type: DistanceType = None, edge_pruning=None, edge_pruning_type: EdgePruningType = None):
        super().__init__(n_words, distributional_model, length_factor, node_decay_factor, edge_decay_sd_factor,
                         impulse_pruning_threshold, firing_threshold, activation_cap, distance_type, edge_pruning,
                         edge_pruning_type)

        # region Resettable

        # Prevent additional impulses being created
        self._block_firing: bool = False

        # endregion

    def reset(self):
        super().reset()
        self._block_firing = False

    def schedule_activation_of_item_with_idx(self, idx: ItemIdx, activation: ActivationValue, arrival_time: int):
        # TODO: this suggests it would simplify things to factor out a firing method, which could be blocked separately
        #  to individual scheduling.
        #  Not worth it unless we start adding more baseline models or tinkering with this one.
        if self._block_firing:
            return
        else:
            super().schedule_activation_of_item_with_idx(idx, activation, arrival_time)

    def _evolve_model(self) -> List[ModelEvent]:
        model_events = super()._evolve_model()
        self._block_firing = True
        return model_events

    def scheduled_activation_count(self) -> int:
        return sum([1
                    for tick, schedule_activation in self._scheduled_activations.items()
                    for idx, activation in schedule_activation.items()
                    if activation > 0])
