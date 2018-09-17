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

from numpy.core.umath import float_power, exp, pi
from scipy.stats import lognorm

TAU = 2 * pi


def decay_function_exponential_with_decay_factor(decay_factor) -> callable:
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

    def decay_function(age, original_activation):
        return original_activation * (decay_factor ** age)

    return decay_function


def decay_function_exponential_with_half_life(half_life) -> callable:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return decay_function_exponential_with_decay_factor(decay_factor)


def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
    """Gaussian decay with sd specifying the number of ticks."""
    assert height_coef > 0
    assert sd > 0

    def decay_function(age, original_activation):
        height = original_activation * height_coef
        return height * exp((-1) * (((age - centre) ** 2) / (2 * sd * sd)))

    return decay_function


def decay_function_lognormal_median(median: float, shape: float) -> callable:
    """
    Lognormal survival decay function, parameterised by the median and the shape.
    :param median:
    :param shape:
        The spread or shape
    :return:
        Decay function
    """
    def decay_function(age, original_activation):
        return original_activation * lognorm.sf(age, s=shape, scale=median)

    return decay_function


def decay_function_lognormal_mean(mu: float, shape: float) -> callable:
    """
    Lognormal survival decay function, parameterised by the mean and the shape.

    (Use of median is preferred.)
    :param mu:
    :param shape:
    :return:
    """
    return decay_function_lognormal_median(exp(mu), shape)


def mean(*items):
    return sum(items) / len(items)
