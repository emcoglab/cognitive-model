"""
===========================
Math utils.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

from typing import Sequence, Tuple

from numpy import percentile, pi, sqrt
from scipy.special import ndtri
from scipy.stats import t as student_t

TAU: float = 2 * pi


def mean(*items):
    """Returns the arithmetic mean of its arguments."""
    return sum(items) / len(items)


def scale01(original_range: Tuple[float, float], value: float) -> float:
    """
    Scales a `value` which is known to exist in the range [`range_min`, `range_max`] to the range [0, 1].
    Ranges can be in RTL order if their endpoints are descending. The `value` does not need to be within its
    `original_range`; the affine transformation of the number-line maps the `original_range` to [0, 1].
    The `original_range` must have non-zero width.
    """
    range_start, range_end = original_range
    return affine_scale((range_start, range_end), (0, 1), value)


def affine_scale(original_range: Tuple[float, float], new_range: Tuple[float, float], value: float) -> float:
    """
    Scales a `value` which is known to exist in the range [`original_range[0]`, `original_range[1]`] to the range
    [`new_range[0]`, `new_range[1]`].
    Ranges can be in RTL order if their endpoints are descending. The `value` does not need to be within its
    `original_range`; the affine transformation of the number-line maps the `original_range` to the `new_range`.
    The `original_range` must have non-zero width.
    """
    original_start, original_end = original_range
    new_start, new_end = new_range
    original_width = original_end - original_start
    new_width = new_end - new_start

    if original_width == 0:
        raise ValueError("Original range cannot have zero width.")

    return new_start + (new_width * (value - original_start) / original_width)


def nearest_value_at_quantile(values, quantile):
    """
    Return items(s) at specified quantile.
    :param values
        Sequence of values from which to form a distribution.
    :param quantile:
        float in range of [0,1] (or sequence of floats)
    :return:
        value (or sequence of values) marking specified quantile.
        Returned values will not be interpolated - nearest values to the quantile will be given.
    """
    # If one quantile provided
    if isinstance(quantile, float):
        centile = 100 * quantile
    # If sequence of quantiles provided
    elif isinstance(quantile, Sequence):
        centile = [100 * q for q in quantile]
    else:
        raise TypeError()
    value = percentile(values, centile, interpolation="nearest")
    return value


def prevalence_from_fraction_known(fraction_known: float) -> float:
    """
    Brysbaert et al.'s (2019) formula for converting fraction-known values into prevalences.
    "The specific formula we used in Microsoft Excel was =NORM.INV(0.005+ Pknown*0.99;0;1)."
    This value is guaranteed to be in the range [-2.575829303548901, 2.5758293035489004]
    """
    return ndtri(0.005 + fraction_known * 0.99)


def scale_prevalence_01(prevalence: float) -> float:
    """
    Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the purposes of
    attenuating the activation.
    """
    return scale01((-2.575829303548901, 2.5758293035489004), prevalence)


def t_confidence_interval(sd: float, n: float, alpha: float) -> float:
    """
    Confidence interval for t distribution.
    Roughly equivalent to Excell's confidence.t()
    :param sd:
    :param n:
    :param alpha:
    :return:
    """
    sem = sd/sqrt(n)
    return sem * student_t.ppf((1 + alpha) / 2, df=n - 1)


def distance_from_similarity(similarity, max_similarity: float, min_similarity: float):
    """
    Converts similarities to distances.

    :param similarity:
        The similarity to convert.
    :param max_similarity:
        The maximum possible similarity.
    :param min_similarity:
        The minimum possible similarity.
    :return:
        The distance.
    """
    return (
        # Convert similarities to lengths by subtracting from the max value
        max_similarity - similarity
        # Add the minimum value to make sure we don't get zero-length edges
        # Use the absolute value in case the minimum is negative (e.g. with PMI).
        + abs(min_similarity)
    )


def cm_to_inches(cm: float) -> float:
    return cm * 0.3937008
