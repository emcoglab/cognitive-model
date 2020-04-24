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

from abc import ABC
from typing import Set, List

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.events import ModelEvent
from model.graph_propagator import Modulation, GraphPropagator


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

    @staticmethod
    def _apply_activation_cap(activation_cap: ActivationValue) -> Modulation:
        def modulation(idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
            """If accumulated activation is over the cap, apply the cap."""
            return activation if activation <= activation_cap else activation_cap
        return modulation
