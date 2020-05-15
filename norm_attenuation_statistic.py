from __future__ import annotations
from enum import Enum, auto


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
