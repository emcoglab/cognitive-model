from __future__ import annotations

from typing import Optional, List, Dict

from model.basic_types import ActivationValue, ItemIdx
from model.buffer import WorkingMemoryBuffer
from model.components import ModelComponentWithAccessibleSet, FULL_ACTIVATION
from model.events import ModelEvent, ItemActivatedEvent
from model.attenuation_statistic import AttenuationStatistic
from model.sensorimotor_propagator import SensorimotorPropagator
from model.utils.iterable import partition
from model.utils.job import SensorimotorPropagationJobSpec, BufferedSensorimotorPropagationJobSpec
from model.utils.maths import prevalence_from_fraction_known, scale_prevalence_01
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


class SensorimotorComponent(ModelComponentWithAccessibleSet):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 attenuation_statistic: AttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 ):

        super().__init__(propagator, accessible_set_threshold, accessible_set_capacity)
        assert isinstance(self.propagator, SensorimotorPropagator)

        assert (activation_cap
                # If activation_cap == accessible_set_threshold, items will only enter the accessible set when fully
                # activated.
                >= self.accessible_set.threshold)

        # Data

        # Use >= and < to test for above/below
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        sensorimotor_norms = SensorimotorNorms()

        def get_statistic_for_item(idx: ItemIdx):
            """Gets the correct statistic for an item."""
            if attenuation_statistic is AttenuationStatistic.FractionKnown:
                # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
                return sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])
            elif attenuation_statistic is AttenuationStatistic.Prevalence:
                # Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the
                # purposes of attenuating the activation
                return scale_prevalence_01(prevalence_from_fraction_known(sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])))
            else:
                raise NotImplementedError()

        self._attenuation_statistic: Dict[ItemIdx, float] = {
            idx: get_statistic_for_item(idx)
            for idx in self.propagator.graph.nodes
        }

        # region modulations and guards

        # No pre-synaptic guards
        self.propagator.presynaptic_modulations.extendleft(
            # Apply cap before attenuations
            # when using extendleft, elements must be presented in reversed order to end up in the correct order
            reversed([
                self._apply_activation_cap(activation_cap),
                self._attenuate_by_statistic,
        ]))
        self.propagator.postsynaptic_modulations.extend([
            # Cap on a node's total activation after receiving incoming
            self._apply_activation_cap(activation_cap)
        ])
        # No post-synaptic guards

        # endregion

    @classmethod
    def from_spec(cls, spec: SensorimotorPropagationJobSpec, use_prepruned: bool = False) -> SensorimotorComponent:
        return cls(
            propagator=SensorimotorPropagator(
                distance_type=spec.distance_type,
                length_factor=spec.length_factor,
                max_sphere_radius=spec.max_radius,
                node_decay_lognormal_median=spec.node_decay_median,
                node_decay_lognormal_sigma=spec.node_decay_sigma,
                use_prepruned=use_prepruned,
            ),
            accessible_set_threshold=spec.accessible_set_threshold,
            accessible_set_capacity=spec.accessible_set_capacity,
            attenuation_statistic=spec.attenuation_statistic,
            activation_cap=FULL_ACTIVATION
        )

    # todo: make static modulation-producers
    def _attenuate_by_statistic(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return activation * self._attenuation_statistic[idx]


class BufferedSensorimotorComponent(SensorimotorComponent):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 attenuation_statistic: AttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue,
                 buffer_capacity: Optional[int],
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
        )

        assert (activation_cap
                # If activation_cap == buffer_threshold, items will only enter the buffer when fully activated.
                >= buffer_threshold
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

    def tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case buffer membership is used to modulate or guard anything)
        decay_events = self.working_memory_buffer.prune_decayed_items(
            activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
            time=self.propagator.clock)

        tick_events = super().tick()
        activation_events, other_events = partition(tick_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update buffer
        # Some events will get updated commensurately.
        # `activation_events` may now contain some non-activation events.
        activation_events = self.working_memory_buffer.present_items(
            activation_events,
            activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
            time=self.propagator.clock)

        return decay_events + activation_events + other_events

    def reset(self):
        super().reset()
        self.working_memory_buffer.clear()

    @classmethod
    def from_spec(cls, spec: BufferedSensorimotorPropagationJobSpec, use_prepruned: bool = False) -> BufferedSensorimotorComponent:
        return cls(
            propagator=SensorimotorPropagator(
                distance_type=spec.distance_type,
                length_factor=spec.length_factor,
                max_sphere_radius=spec.max_radius,
                node_decay_lognormal_median=spec.node_decay_median,
                node_decay_lognormal_sigma=spec.node_decay_sigma,
                use_prepruned=use_prepruned,
            ),
            accessible_set_threshold=spec.accessible_set_threshold,
            accessible_set_capacity=spec.accessible_set_capacity,
            attenuation_statistic=spec.attenuation_statistic,
            activation_cap=FULL_ACTIVATION,
            buffer_capacity=spec.buffer_capacity,
            buffer_threshold=spec.buffer_threshold,
        )
