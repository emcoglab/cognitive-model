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
from typing import Sequence, Tuple, Callable

from numpy import percentile, float_power, pi, sqrt, log
from scipy.special import ndtri
from scipy.stats import t as student_t

from model.utils.maths_core import gaussian_decay, exponential_decay, lognormal_sf
from model.basic_types import ActivationValue

TAU: float = 2 * pi


# region Decay functions

def make_decay_function_constant() -> Callable[[int, ActivationValue], ActivationValue]:
    """
    Constant decay function (i.e. does not decay).
    :return:
    """
    def constant_decay_function(_age, original_activation):
        return original_activation

    return constant_decay_function


def make_decay_function_exponential_with_decay_factor(decay_factor) -> Callable[[int, ActivationValue], ActivationValue]:
    # Decay formula for activation a, original activation a_0, decay factor d, time t:
    #   a = a_0 d^t
    #
    # In traditional formulation of exponential decay, this is equivalent to:
    #   a = a_0 e^(-λt)
    # where λ is the decay constant.
    #
    # I.e.
    #   d = e^(-λ)
    #   λ = - ln d
    assert 0 < decay_factor <= 1

    def exponential_decay_function(age, original_activation):
        return exponential_decay(age=age, original_activation=original_activation, decay_factor=decay_factor)

    return exponential_decay_function


def make_decay_function_exponential_with_half_life(half_life) -> Callable[[int, ActivationValue], ActivationValue]:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return make_decay_function_exponential_with_decay_factor(decay_factor)


def make_decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> Callable[[int, ActivationValue], ActivationValue]:
    """
    Gaussian decay function with sd specifying the number of ticks.
    :param sd:
    :param height_coef:
    :param centre:
    :return:
    """
    assert height_coef > 0
    assert sd > 0

    # The actual normal pdf has height 1/sqrt(2 pi sd^2). We want its natural height to be 1 (which is then scaled
    # by the original activation), so we force that here.
    reset_height = sqrt(TAU * sd * sd)

    def gaussian_decay_function(age, original_activation):
        return gaussian_decay(age=age,
                              original_activation=original_activation,
                              height_coef=height_coef,
                              reset_height=reset_height,
                              centre=centre,
                              sd=sd)

    return gaussian_decay_function


def make_decay_function_lognormal(median: float, sigma: float) -> Callable[[int, ActivationValue], ActivationValue]:
    """
    Lognormal survival decay function.
    :param median:
        Median of the decay.
    :param sigma:
        The spread or shape.w
    :return:
    """

    # Where the lognormal_sf is paramterised by params mu and sigma, we can convert the median of the decay into the mu
    # by taking the log. (See Mueller, S. T., & Krawitz, A. (2009). Reconsidering the two-second decay hypothesis in
    # verbal working memory. Journal of Mathematical Psychology, 53(1), 14-25.)
    mu = log(median)

    def lognormal_decay_function(age, original_activation):
        return original_activation * lognormal_sf(x=age, mu=mu, sigma=sigma)

    return lognormal_decay_function

# endregion


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
