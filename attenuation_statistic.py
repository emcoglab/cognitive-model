from __future__ import annotations
from enum import Enum, auto


class AttenuationStatistic(Enum):
    """The statistic to use for attenuating activation of norms labels."""
    FractionKnown = auto()
    Prevalence = auto()

    @property
    def name(self) -> str:
        """The name of the AttenuationStatistic"""
        if self is AttenuationStatistic.FractionKnown:
            return "Fraction known"
        if self is AttenuationStatistic.Prevalence:
            return "Prevalence"
        else:
            raise NotImplementedError()

    @classmethod
    def from_slug(cls, slug: str) -> AttenuationStatistic:
        if slug.lower() in {"fraction-known", "fraction", "known", "fractionknown", AttenuationStatistic.FractionKnown.name.lower()}:
            return cls.FractionKnown
        elif slug.lower() in {"prevalence", AttenuationStatistic.Prevalence.name.lower()}:
            return cls.Prevalence
        else:
            raise NotImplementedError()
