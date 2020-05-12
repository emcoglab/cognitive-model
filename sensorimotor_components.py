from __future__ import annotations

from enum import Enum, auto
from typing import Optional, List, Dict

from model.basic_types import ActivationValue, ItemIdx
from model.buffer import WorkingMemoryBuffer
from model.components import ModelComponentWithAccessibleSet
from model.events import ModelEvent, ItemActivatedEvent
from model.sensorimotor_propagator import SensorimotorPropagator
from model.utils.iterable import partition
from model.utils.maths import prevalence_from_fraction_known, scale_prevalence_01
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


class SensorimotorComponent(ModelComponentWithAccessibleSet):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 norm_attenuation_statistic: NormAttenuationStatistic,
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
            if norm_attenuation_statistic is NormAttenuationStatistic.FractionKnown:
                # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
                return sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])
            elif norm_attenuation_statistic is NormAttenuationStatistic.Prevalence:
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

    # todo: make static modulation-producers
    def _attenuate_by_statistic(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return activation * self._attenuation_statistic[idx]


class BufferedSensorimotorComponent(SensorimotorComponent):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 norm_attenuation_statistic: NormAttenuationStatistic,
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
            norm_attenuation_statistic=norm_attenuation_statistic,
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


class NormAttenuationStatistic(Enum):
    """The statistic to use for attenuating activation of norms labels."""
    FractionKnown = auto()
    Prevalence = auto()

    @property
    def name(self) -> str:
        """The name of the NormAttenuationStatistic"""
        if self is NormAttenuationStatistic.FractionKnown:
            return "Fraction known"
        if self is NormAttenuationStatistic.Prevalence:
            return "Prevalence"
        else:
            raise NotImplementedError()

    @classmethod
    def from_slug(cls, slug: str) -> NormAttenuationStatistic:
        if slug.lower() in {"fraction-known", "fraction", "known", "fractionknown", NormAttenuationStatistic.FractionKnown.name.lower()}:
            return cls.FractionKnown
        elif slug.lower() in {"prevalence", NormAttenuationStatistic.Prevalence.name.lower()}:
            return cls.Prevalence
        else:
            raise NotImplementedError()
