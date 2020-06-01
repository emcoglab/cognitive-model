"""
===========================
Base classes for model components
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, List, Optional

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.buffer import AccessibleSet
from model.events import ModelEvent, ItemActivatedEvent
from model.graph_propagator import Modulation, GraphPropagator
from model.utils.iterable import partition
from model.utils.job import PropagationJobSpec

FULL_ACTIVATION = ActivationValue(1.0)


class ModelComponent(ABC):

    def __init__(self, propagator: GraphPropagator):
        self.propagator: GraphPropagator = propagator

        # This won't change so we set it once
        self._available_labels: Set[ItemLabel] = set(w for i, w in self.propagator.idx2label.items())

    @property
    def available_labels(self) -> Set[ItemLabel]:
        """Labels of concepts in the model component."""
        return self._available_labels

    def reset(self):
        self.propagator.reset()

    def tick(self) -> List[ModelEvent]:
        return self.propagator.tick()

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: PropagationJobSpec) -> ModelComponent:
        """Produce a component from a job spec, with choices set to defaults."""
        raise NotImplementedError()

    @staticmethod
    def _apply_activation_cap(activation_cap: ActivationValue) -> Modulation:
        def modulation(idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
            """If accumulated activation is over the cap, apply the cap."""
            return activation if activation <= activation_cap else activation_cap
        return modulation


class ModelComponentWithAccessibleSet(ModelComponent, ABC):

    def __init__(self,
                 propagator: GraphPropagator,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 ):

        super().__init__(propagator)

        # The set of items which are "accessible to conscious awareness"
        self.accessible_set: AccessibleSet = AccessibleSet(threshold=accessible_set_threshold, capacity=accessible_set_capacity)

        self.propagator.presynaptic_modulations.extend([
            self._apply_memory_pressure,
        ])
        self.propagator.postsynaptic_guards.extend([
            self._not_in_accessible_set
        ])

    def _apply_memory_pressure(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # When AS is empty, MP is 0, and activation is unaffected.
        # When AS is full,  MP is 1, and activation is killed.
        return activation * (1 - self.accessible_set.pressure)

    def _not_in_accessible_set(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node will only fire if it's not in the accessible set
        return idx not in self.accessible_set

    def reset(self):
        super().reset()
        self.accessible_set.clear()

    def tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case accessible set membership is used to modulate or guard anything)
        self.accessible_set.prune_decayed_items(activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
                                                time=self.propagator.clock)

        # Proceed with .tick() and record what became activated
        # Activation and firing may be affected by the size of or membership to the accessible set and the buffer, but
        # nothing will ENTER it until later, and everything that will LEAVE this tick already has done so.
        tick_events = super().tick()
        activation_events, other_events = partition(tick_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update accessible set
        self.accessible_set.present_items(activation_events=activation_events,
                                          activation_lookup=lambda item: self.propagator.activation_of_item_with_idx(item.idx),
                                          time=self.propagator.clock)

        return activation_events + other_events
