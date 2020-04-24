"""
===========================
Code for submission of jobs on SGE Wayland.
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
from __future__ import annotations

from pathlib import Path
from subprocess import run
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue
from model.graph import EdgePruningType
from model.sensorimotor_components import NormAttenuationStatistic
from model.version import VERSION


@dataclass
class JobSpec(ABC):

    @property
    @abstractmethod
    def shorthand(self) -> str:
        """
        A short name which may not uniquely define the spec, but can be used to
        disambiguate job names, etc.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def cli_args(self) -> List[str]:
        """
        List of key-value pairs in `--arg_name val` format.
        """
        raise NotImplementedError()

    @abstractmethod
    def output_location(self, for_version: str = VERSION) -> Path:
        """
        Relative path for a job's output to be saved.
        """
        raise NotImplementedError()


@dataclass
class PropagationJobSpec(JobSpec, ABC):
    length_factor: int
    run_for_ticks: Optional[int]
    bailout: Optional[int]

    @property
    @abstractmethod
    def cli_args(self) -> List[str]:
        args = [
            f"--length_factor {self.length_factor}",
        ]
        if self.run_for_ticks is not None:
            args.append(f"--run_for_ticks {self.run_for_ticks}")
        if self.bailout is not None:
            args.append(f"--bailout {self.bailout}")
        return args


@dataclass
class SensorimotorPropagationJobSpec(PropagationJobSpec):
    max_radius: int
    node_decay_sigma: float
    node_decay_median: float
    buffer_threshold: float
    accessible_set_threshold: float
    distance_type: DistanceType
    buffer_capacity: Optional[int]
    accessible_set_capacity: Optional[int]
    attenuation_statistic: NormAttenuationStatistic

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--distance_type {self.distance_type.name}",
            f"--max_sphere_radius {self.max_radius}",
            f"--accessible_set_capacity {self.accessible_set_capacity}",
            f"--buffer_capacity {self.buffer_capacity}",
            f"--buffer_threshold {self.buffer_threshold}",
            f"--accessible_set_threshold {self.accessible_set_threshold}",
            f"--length_factor {self.length_factor}",
            f"--node_decay_median {self.node_decay_median}",
            f"--node_decay_sigma {self.node_decay_sigma}",
            f"--attenuation {self.attenuation_statistic.name}",
        ]
        return args

    @property
    def shorthand(self) -> str:
        return f"sm_" \
               f"r{self.max_radius}_" \
               f"m{self.node_decay_median}_" \
               f"s{self.node_decay_sigma}_" \
               f"a{self.accessible_set_threshold}_" \
               f"ac{self.accessible_set_capacity if self.accessible_set_capacity is not None else '-'}_" \
               f"b{self.buffer_threshold}"

    # TODO:
    def output_location(self, for_version: str = VERSION) -> Path:
        return Path(
            f"Sensorimotor {for_version}",
            f"{self.distance_type.name} length {self.length_factor} attenuate {self.attenuation_statistic.name}",
            f"max-r {self.max_radius};"
            f" n-decay-median {self.node_decay_median};"
            f" n-decay-sigma {self.node_decay_sigma};"
            f" as-θ {self.accessible_set_threshold};"
            f" as-cap {self.accessible_set_capacity:,};"
            f" buff-θ {self.buffer_threshold};"
            f" buff-cap {self.buffer_capacity};"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}",
        )


@dataclass
class LinguisticPropagationJobSpec(PropagationJobSpec):
    n_words: int
    firing_threshold: float
    model_name: str
    model_radius: int
    corpus_name: str
    edge_decay_sd: float
    node_decay_factor: float
    impulse_pruning_threshold: ActivationValue
    pruning_type: Optional[EdgePruningType]
    pruning: Optional[int]
    distance_type: Optional[DistanceType] = None

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--words {self.n_words}",
            f"--firing_threshold {self.firing_threshold}",
            f"--model_name {self.model_name}",
            f"--radius {self.model_radius}",
            f"--corpus_name {self.corpus_name}",
            f"--edge_decay_sd_factor {self.edge_decay_sd}",
            f"--node_decay_factor {self.node_decay_factor}",
            f"--impulse_pruning_threshold {self.impulse_pruning_threshold}",
        ]
        if self.pruning is not None:
            if self.pruning_type == EdgePruningType.Importance:
                args.append(f"--prune_importance {self.pruning}")
            elif self.pruning_type == EdgePruningType.Percent:
                args.append(f"--prune_percent {self.pruning}")
            elif self.pruning_type == EdgePruningType.Length:
                args.append(f"--prune_length {self.pruning}")
            else:
                raise NotImplementedError()
        if self.distance_type is not None:
            args.append(f"--distance_type {self.distance_type.name}")
        return args

    def output_location(self, for_version: str = VERSION) -> Path:
        if self.pruning_type is None:
            pruning_suffix = ""
        elif self.pruning_type == EdgePruningType.Percent:
            pruning_suffix = f", longest {self.pruning}% edges removed"
        elif self.pruning_type == EdgePruningType.Importance:
            pruning_suffix = f", importance pruning {self.pruning}"
        else:
            raise NotImplementedError()
        return Path(
            f"Linguistic {for_version}",
            f"{self.model_name} {self.distance_type.name} {self.n_words:,} words,"
            f" length {self.length_factor}{pruning_suffix}",
            f"firing-θ {self.firing_threshold};"
            f" n-decay-f {self.node_decay_factor};"
            f" e-decay-sd {self.edge_decay_sd};"
            f" imp-prune-θ {self.impulse_pruning_threshold};"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}",
        )

    @property
    def shorthand(self):
        return f"{int(self.n_words / 1000)}k_" \
               f"f{self.firing_threshold}_" \
               f"s{self.edge_decay_sd}_" \
               f"{self.model_name}_" \
               f"pr{self.pruning}"


class LinguisticOneHopJobSpec(LinguisticPropagationJobSpec):
    def output_location(self, for_version: str = VERSION) -> Path:
        return Path(
            f"Linguistic one-hop {for_version}",
            f"{self.model_name}"
            f" {self.n_words:,} words, length {self.length_factor}",
            f"firing-θ {self.firing_threshold};"
            f" n-decay-f {self.node_decay_factor};"
            f" e-decay-sd {self.edge_decay_sd};"
            f" imp-prune-θ {self.impulse_pruning_threshold}"
        )


class SensorimotorOneHopJobSpec(SensorimotorPropagationJobSpec):
    def output_location(self, for_version: str = VERSION) -> Path:
        return Path(
            f"Sensorimotor one-hop {for_version}",
            f"{self.distance_type.name} length {self.length_factor} attenuate {self.attenuation_statistic.name}",
            f"max-r {self.max_radius};"
            f" n-decay-median {self.node_decay_median};"
            f" n-decay-sigma {self.node_decay_sigma};"
            f" as-θ {self.accessible_set_threshold};"
            f" as-cap {self.accessible_set_capacity:,};"
            f" buff-θ {self.buffer_threshold};"
            f" buff-cap {self.buffer_capacity}"
        )


@dataclass
class CombinedJobSpec(JobSpec, ABC):
    linguistic_spec: LinguisticPropagationJobSpec
    sensorimotor_spec: SensorimotorPropagationJobSpec

    @property
    def cli_args(self) -> List[str]:
        # TODO: this isn't right
        return self.linguistic_spec.cli_args + self.sensorimotor_spec.cli_args


@dataclass
class NoninteractiveCombinedJobSpec(CombinedJobSpec):
    @property
    def shorthand(self) -> str:
        return f"ni_{self.linguistic_spec.shorthand}_{self.sensorimotor_spec.shorthand}"


class Job(ABC):
    _shim = "model/utils/shim.sh"

    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: JobSpec,
                 ):
        self._number: str = script_number
        self.short_name: str = "j" + script_number.replace("_", "")
        self.script_name: str = script_name  # "../" + script_name
        self.module_name: str = Job._without_py(script_name)
        self.spec = spec

    @property
    def name(self) -> str:
        return f"{self.short_name}_{self.spec.shorthand}"

    @property
    @abstractmethod
    def _ram_requirement_g(self) -> float:
        raise NotImplementedError

    # @final <- TODO: wait for 3.8
    @property
    def qsub_command(self) -> str:
        """The qsub command to run, complete with arguments, to execute this job."""
        cmd = f"qsub"
        # qsub args
        cmd += f" -N {self.name}"
        cmd += f" -l h_vmem={self._ram_requirement_g}G"
        # script
        cmd += f" {self._shim} "
        cmd += self.command
        return cmd

    @property
    @abstractmethod
    def command(self) -> str:
        """The CLI command to run, complete with arguments, to execute this job."""
        cmd = self.script_name
        cmd += " ".join(self.spec.cli_args)
        return cmd

    def run_locally(self):
        print(self.command)
        run(f"python {self.command}", shell=True)

    def submit(self):
        print(self.qsub_command)
        run(self.qsub_command, shell=True)

    @classmethod
    def _without_py(cls, script_name: str) -> str:
        if script_name.endswith(".py"):
            return script_name[:-3]
        else:
            return script_name


class PropagationJob(Job, ABC):
    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: JobSpec):
        super().__init__(
            script_number=script_number,
            script_name=script_name,
            spec=spec)


class SensorimotorPropagationJob(PropagationJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: SensorimotorPropagationJobSpec
        super().__init__(*args, **kwargs)


class LinguisticPropagationJob(PropagationJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: LinguisticPropagationJobSpec
        super().__init__(*args, **kwargs)
