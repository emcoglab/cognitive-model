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
from os import path
from pathlib import Path
from typing import Set, List, Optional, Dict

import yaml

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.events import ModelEvent
from model.graph_propagator import Modulation, GraphPropagator


class ModelComponent(ABC):

    def __init__(self, propagator: GraphPropagator):
        self.propagator: GraphPropagator = propagator

        # This won't change so we set it once
        self._available_labels: Set[ItemLabel] = set(w for i, w in self.propagator.idx2label.items())

    @property
    def _model_spec(self) -> Dict:
        return self.propagator._model_spec

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

    def save_model_spec(self, response_dir: Path, additional_fields: Optional[Dict] = None):
        """
        Save the model spec to the `response_dir`.
        :param response_dir:
        :param additional_fields:
            If provided and not None, add these fields to the spec.
        """
        spec = self._model_spec.copy()
        if additional_fields:
            spec.update(additional_fields)
        with open(Path(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
            yaml.dump(spec, spec_file, yaml.SafeDumper)

    @classmethod
    def load_model_spec(cls, response_dir) -> dict:
        with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
            return yaml.load(spec_file, yaml.SafeLoader)
