from __future__ import annotations

from typing import Optional, List, Dict

from .basic_types import ActivationValue, ItemIdx
from .buffer import WorkingMemoryBuffer
from .components import ModelComponentWithAccessibleSet
from .modulations import make_apply_activation_cap_modulation_for, make_attenuate_by_statistic_modulation_for
from .events import ModelEvent, ItemActivatedEvent
from .attenuation_statistic import AttenuationStatistic
from .propagator_sensorimotor import SensorimotorPropagator
from .utils.iterable import partition
from .utils.maths import prevalence_from_fraction_known, scale_prevalence_01
from .sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


class SensorimotorComponent(ModelComponentWithAccessibleSet):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: Optional[ActivationValue],
                 attenuation_statistic: AttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 use_breng_translation: bool,
                 ):

        self.sensorimotor_norms = SensorimotorNorms(use_breng_translation=use_breng_translation)

        super().__init__(propagator, accessible_set_threshold, accessible_set_capacity)
        assert isinstance(self.propagator, SensorimotorPropagator)

        if activation_cap is not None:
            assert (activation_cap
                    # If activation_cap == accessible_set_threshold, items will only enter the accessible set when fully
                    # activated.
                    >= self.accessible_set.threshold)

        def get_statistic_for_item(idx: ItemIdx):
            """Gets the correct statistic for an item."""
            if attenuation_statistic is AttenuationStatistic.FractionKnown:
                # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
                return self.sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])
            elif attenuation_statistic is AttenuationStatistic.Prevalence:
                # Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for
                # the purposes of attenuating the activation
                return scale_prevalence_01(prevalence_from_fraction_known(self.sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])))
            else:
                raise NotImplementedError()

        self._attenuation_statistic: Dict[ItemIdx, float] = {
            idx: get_statistic_for_item(idx)
            for idx in self.propagator.graph.nodes
        }

        # region modulations and guards

        # No pre-synaptic guards
        self.propagator.presynaptic_modulations.appendleft(
            make_attenuate_by_statistic_modulation_for(self._attenuation_statistic)
        )
        if activation_cap is not None:
            # Apply cap before attenuations
            self.propagator.presynaptic_modulations.appendleft(
                make_apply_activation_cap_modulation_for(activation_cap)
            )
        if activation_cap is not None:
            self.propagator.postsynaptic_modulations.extend([
                # Cap on a node's total activation after receiving incoming activations
                make_apply_activation_cap_modulation_for(activation_cap)
            ])
        # No post-synaptic guards

        # endregion


class BufferedSensorimotorComponent(SensorimotorComponent):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: Optional[ActivationValue],
                 attenuation_statistic: AttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue,
                 buffer_capacity: Optional[int],
                 use_breng_translation: bool,
                 ):
        """
        :param buffer_capacity:
            The maximum size of the buffer. After this, qualifying items will displace existing items rather than just
            being added.
        :param buffer_threshold:
            The minimum activation required for a concept to enter the working_memory_buffer.
        """

        super().__init__(
            propagator=propagator,
            activation_cap=activation_cap,
            attenuation_statistic=attenuation_statistic,
            accessible_set_threshold=accessible_set_threshold,
            accessible_set_capacity=accessible_set_capacity,
            use_breng_translation=use_breng_translation,
        )

        if activation_cap is not None:
            assert (activation_cap
                    # If activation_cap == buffer_threshold, items will only enter the buffer when fully activated.
                    >= buffer_threshold)
        assert (buffer_threshold
                # If buffer_pruning_threshold == accessible_set_threshold then the only things in the accessible set
                # will be those items which were displaced from the buffer before being pruned. We probably won't use
                # this but it's not invalid or degenerate.
                >= accessible_set_threshold)

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are currently being consciously considered.
        #
        # A fixed size (self.buffer_capacity).  Items may enter the buffer when they are activated and leave when they
        # decay sufficiently (self.buffer_pruning_threshold) or are displaced.
        #
        # This is updated each .tick() based on items which became activated (a prerequisite for entering the buffer)
        self.working_memory_buffer: WorkingMemoryBuffer = WorkingMemoryBuffer(buffer_threshold, buffer_capacity)

        # endregion

    def _pre_tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case buffer membership is used to modulate or guard anything)
        decay_events = self.working_memory_buffer.prune_decayed_items(
            activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
            time=self.propagator.clock)
        return decay_events

    def _post_tick(self,
                   pre_tick_events: List[ModelEvent],
                   propagator_events: List[ModelEvent],
                   time_at_start_of_tick: int,
                   ) -> List[ModelEvent]:
        activation_events, other_events = partition(propagator_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update buffer
        # Some events will get updated commensurately.
        # `activation_events` may now contain some non-activation events.
        activation_events = self.working_memory_buffer.present_items(
            activation_events=activation_events,
            activation_lookup=lambda item: self.propagator.activation_of_item_with_idx_at_time(item.idx, time=time_at_start_of_tick),
            time=time_at_start_of_tick)

        return pre_tick_events + activation_events + other_events

    def reset(self):
        super().reset()
        self.working_memory_buffer.clear()
