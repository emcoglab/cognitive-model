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
from typing import List, Optional

from ldm.utils.maths import DistanceType
from model.basic_types import ItemIdx, ActivationValue
from model.events import ModelEvent
from model.sensorimotor_component import SensorimotorComponent, NormAttenuationStatistic


class SensorimotorOneHopComponent(SensorimotorComponent):
    """A SensorimotorComponent which allows only hops from the initial nodes."""
    def __init__(self, distance_type: DistanceType, length_factor: int, max_sphere_radius: int, node_decay_lognormal_median: float,
                 node_decay_lognormal_sigma: float, buffer_capacity: Optional[int], accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue, accessible_set_threshold: ActivationValue,
                 activation_cap: ActivationValue, norm_attenuation_statistic: NormAttenuationStatistic,
                 use_prepruned: bool):

        super().__init__(distance_type, length_factor, max_sphere_radius, node_decay_lognormal_median, node_decay_lognormal_sigma,
                         buffer_capacity, accessible_set_capacity, buffer_threshold, accessible_set_threshold,
                         activation_cap, norm_attenuation_statistic, use_prepruned)

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

    def scheduled_activation_count(self) -> int:
        return sum([1
                    for tick, schedule_activation in self._scheduled_activations.items()
                    for idx, activation in schedule_activation.items()
                    if activation > 0])
