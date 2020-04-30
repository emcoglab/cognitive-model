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
from typing import Optional, List, Dict

import yaml

from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, Length
from model.graph import EdgePruningType
from model.sensorimotor_components import NormAttenuationStatistic
from model.version import VERSION, GIT_HASH

_SerialisableDict = Dict[str, str]


# region Job Specs

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
    def _to_dict(self) -> _SerialisableDict:
        """Serialise."""
        return {
            "Version": VERSION,
            "Commit": GIT_HASH,
        }

    @classmethod
    @abstractmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        """Deserialise.  Does not preserve Version or Commit."""
        raise NotImplementedError()

    @abstractmethod
    def output_location_relative(self) -> Path:
        """
        Relative path for a job's output to be saved.
        """
        raise NotImplementedError()

    def save(self, in_location: Path) -> None:
        """
        Save the model spec in a common format.
        Creates the output location if it doesn't already exist.
        """
        if not in_location.is_dir():
            in_location.mkdir(parents=True)
        with open(Path(in_location, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
            yaml.dump(self._to_dict(), spec_file, yaml.SafeDumper)

    @classmethod
    def load(cls, filename: Path):
        # This works
        # noinspection PyTypeChecker
        with open(filename, mode="r", encoding="utf-8") as file:
            return cls._from_dict(yaml.load(file, yaml.SafeLoader))


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

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Length factor": str(self.length_factor),
        })
        if self.run_for_ticks is not None:
            d.update({
                "Run for ticks": str(self.run_for_ticks),
            })
        if self.bailout is not None:
            d.update({
                "Bailout": str(self.bailout),
            })
        return d


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

    def output_location_relative(self) -> Path:
        return Path(
            f"Sensorimotor {VERSION}",
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

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Buffer capacity": str(self.buffer_capacity),
            "Buffer threshold": str(self.buffer_threshold),
            "Distance type": self.distance_type.name,
            "Length factor": str(self.length_factor),
            "Max sphere radius": str(self.max_radius),
            "Log-normal median": str(self.node_decay_median),
            "Log-normal sigma": str(self.node_decay_sigma),
            "Accessible set threshold": str(self.accessible_set_threshold),
            "Accessible set capacity": str(self.accessible_set_capacity),
            "Attenuation statistic": self.attenuation_statistic.name,
        })
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return SensorimotorPropagationJobSpec(
            length_factor=int(dictionary["Length factor"]),
            run_for_ticks=dictionary["Run for ticks"] if "Run for ticks" in dictionary else None,
            bailout=dictionary["Bailout"] if "Bailout" in dictionary else None,
            max_radius=Length(dictionary["Max radius"]),
            node_decay_sigma=float(dictionary["Log-normal sigma"]),
            node_decay_median=float(dictionary["Log-normal median"]),
            buffer_threshold=ActivationValue(dictionary["Buffer threshold"]),
            accessible_set_threshold=int(dictionary["Buffer capacity"]),
            distance_type=DistanceType.from_name(dictionary["Distance type"]),
            buffer_capacity=int(dictionary["Buffer capacity"]),
            accessible_set_capacity=int(dictionary["Accessible set capacity"]),
            attenuation_statistic=NormAttenuationStatistic.from_slug(dictionary["Attenuation statistic"]),
        )


@dataclass
class LinguisticPropagationJobSpec(PropagationJobSpec):
    n_words: int
    firing_threshold: ActivationValue
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

    def output_location_relative(self) -> Path:
        if self.pruning_type is None:
            pruning_suffix = ""
        elif self.pruning_type == EdgePruningType.Percent:
            pruning_suffix = f", longest {self.pruning}% edges removed"
        elif self.pruning_type == EdgePruningType.Importance:
            pruning_suffix = f", importance pruning {self.pruning}"
        else:
            raise NotImplementedError()

        if self.distance_type is not None:
            model_dir_name = (f"{self.model_name}"
                              f" {self.distance_type.name}"
                              f" {self.n_words:,} words, length {self.length_factor}{pruning_suffix}")
        else:
            model_dir_name = (f"{self.model_name}"
                              f" {self.n_words:,} words, length {self.length_factor}{pruning_suffix}")

        return Path(
            f"Linguistic {VERSION}",
            model_dir_name,
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

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Words": str(self.n_words),
            "Model name": self.model_name,
            "Model radius": str(self.model_radius),
            "Corpus name": self.corpus_name,
            "Length factor": str(self.length_factor),
            "SD factor": str(self.edge_decay_sd),
            "Node decay": str(self.node_decay_factor),
            "Firing threshold": str(self.firing_threshold),
            "Impulse pruning threshold": str(self.impulse_pruning_threshold),
        })
        if self.distance_type is not None:
            d.update({
                "Distance type": self.distance_type.name,
            })
        if self.pruning_type is not None:
            d.update({
                "Pruning type": self.pruning_type.name,
                "Pruning": str(self.pruning),
            })
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return LinguisticPropagationJobSpec(
            length_factor=int(dictionary["Length factor"]),
            run_for_ticks=dictionary["Run for ticks"] if "Run for ticks" in dictionary else None,
            bailout=dictionary["Bailout"] if "Bailout" in dictionary else None,
            distance_type=DistanceType.from_name(dictionary["Distance type"]) if "Distance type" in dictionary else None,
            n_words=int(dictionary["Words"]),
            firing_threshold=ActivationValue(dictionary["Firing threshold"]),
            model_name=str(dictionary["Model name"]),
            model_radius=int(dictionary["Model radius"]),
            corpus_name=str(dictionary["Corpus name"]),
            edge_decay_sd=float(dictionary["SD factor"]),
            node_decay_factor=float(dictionary["Node decay"]),
            impulse_pruning_threshold=ActivationValue(dictionary["Impulse pruning threshold"]),
            pruning_type=EdgePruningType.from_name(dictionary["Pruning type"]) if "Pruning type" in dictionary else None,
            pruning=int(dictionary["Pruning"]) if "Pruning" in dictionary else None,
        )


class LinguisticOneHopJobSpec(LinguisticPropagationJobSpec):
    def output_location_relative(self) -> Path:
        return Path(
            f"Linguistic one-hop {VERSION}",
            f"{self.model_name}"
            f" {self.n_words:,} words, length {self.length_factor}",
            f"firing-θ {self.firing_threshold};"
            f" n-decay-f {self.node_decay_factor};"
            f" e-decay-sd {self.edge_decay_sd};"
            f" imp-prune-θ {self.impulse_pruning_threshold}"
        )


class SensorimotorOneHopJobSpec(SensorimotorPropagationJobSpec):
    def output_location_relative(self) -> Path:
        return Path(
            f"Sensorimotor one-hop {VERSION}",
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

    def _to_dict(self) -> _SerialisableDict:
        return {
            **{
                f"(Linguistic) " + key: value
                for key, value in self.linguistic_spec._to_dict().items()
            },
            **{
                f"(Sensorimotor) " + key: value
                for key, value in self.sensorimotor_spec._to_dict().items()
            }
        }

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        def trim_and_filter_keys(d: _SerialisableDict, prefix: str):
            return {
                key[len(prefix)]: value
                for key, value in d.items()
                if key.startswith(prefix)
            }
        return CombinedJobSpec(
            linguistic_spec=LinguisticPropagationJobSpec._from_dict(
                trim_and_filter_keys(dictionary, "(Linguistic) ")),
            sensorimotor_spec=SensorimotorPropagationJobSpec._from_dict(
                trim_and_filter_keys(dictionary, "(Sensorimotor) ")),
        )


@dataclass
class NoninteractiveCombinedJobSpec(CombinedJobSpec):
    def output_location_relative(self) -> Path:
        pass

    @property
    def shorthand(self) -> str:
        return f"ni_{self.linguistic_spec.shorthand}_{self.sensorimotor_spec.shorthand}"

# endregion


# region Jobs

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
    def command(self) -> str:
        """The CLI command to run, complete with arguments, to execute this job."""
        cmd = self.script_name
        cmd += " "  # separates args from script name
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

# endregion
