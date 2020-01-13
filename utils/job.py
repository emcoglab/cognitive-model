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
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ldm.utils.maths import DistanceType


@dataclass
class Spec(ABC):

    @property
    @abstractmethod
    def shorthand(self) -> str:
        raise NotImplementedError()


@dataclass
class SASpec(Spec, ABC):
    length_factor: int


@dataclass
class SensorimotorSASpec(SASpec):
    max_radius: int
    node_decay_sigma: float
    node_decay_median: int
    buffer_threshold: float
    accessible_set_threshold: float
    distance_type: DistanceType
    buffer_capacity: Optional[int]
    accessible_set_capacity: Optional[int]

    @property
    def shorthand(self) -> str:
        return f"sm_" \
               f"r{self.max_radius}_" \
               f"m{self.node_decay_median}_" \
               f"s{self.node_decay_sigma}_" \
               f"a{self.accessible_set_threshold}_" \
               f"ac{self.accessible_set_capacity if self.accessible_set_capacity is not None else '-'}_" \
               f"b{self.buffer_threshold}"


@dataclass
class LinguisticSASpec(SASpec):
    graph_size: int
    firing_threshold: float
    model_name: str
    model_radius: int
    corpus_name: str
    edge_decay_sd: float
    impulse_pruning_threshold: float
    node_decay_factor: float
    pruning: Optional[int]
    distance_type: Optional[DistanceType] = None

    @property
    def shorthand(self):
        return f"{int(self.graph_size / 1000)}k_" \
               f"f{self.firing_threshold}_" \
               f"s{self.edge_decay_sd}_" \
               f"{self.model_name}_" \
               f"pr{self.pruning}"


class Job(ABC):
    _shim = "model/utils/shim.sh"

    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: Spec,
                 ):
        self._number: str = script_number
        self.short_name: str = "j" + script_number.replace("_", "")
        self.script_name: str = script_name  # "../" + script_name
        self.module_name: str = Job._without_py(script_name)
        self.spec = spec

    @property
    def name(self) -> str:
        return f"{self.short_name}{self.spec.shorthand}"

    @property
    @abstractmethod
    def _ram_requirement_g(self) -> float:
        raise NotImplementedError

    @property
    def qsub_command(self) -> str:
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
        raise NotImplementedError()

    def run_locally(self):
        print(self.command)
        subprocess.run(f"python {self.command}", shell=True)

    def submit(self):
        print(self.qsub_command)
        subprocess.run(self.qsub_command, shell=True)

    @classmethod
    def _without_py(cls, script_name: str) -> str:
        if script_name.endswith(".py"):
            return script_name[:-3]
        else:
            return script_name


class SAJob(Job, ABC):
    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: Spec,
                 run_for_ticks: Optional[int] = None,
                 bailout: Optional[int] = None):
        super().__init__(
            script_number=script_number,
            script_name=script_name,
            spec=spec)
        self.run_for_ticks: Optional[int] = run_for_ticks
        self.bailout: Optional[int] = bailout


class SensorimotorSAJob(SAJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: SensorimotorSASpec
        super().__init__(*args, **kwargs)


class LinguisticSAJob(SAJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: LinguisticSASpec
        super().__init__(*args, **kwargs)
