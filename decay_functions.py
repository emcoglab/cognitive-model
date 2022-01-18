"""
===========================
Decay functions
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2022
---------------------------
"""

from typing import Callable

from numpy import float_power, sqrt, log

from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.utils.maths_core import exponential_decay, gaussian_decay, lognormal_sf
from framework.cognitive_model.utils.maths import TAU

# Maps an elapsed time and and initial activation to a final activation
DecayFunction = Callable[[int, ActivationValue], ActivationValue]


# Functions to make decay functions

def make_decay_function_constant() -> DecayFunction:
    """
    Constant decay function (i.e. does not decay).
    :return:
    """
    def constant_decay_function(_age, original_activation):
        return original_activation

    return constant_decay_function


def make_decay_function_exponential_with_decay_factor(decay_factor) -> DecayFunction:
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


def make_decay_function_exponential_with_half_life(half_life) -> DecayFunction:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return make_decay_function_exponential_with_decay_factor(decay_factor)


def make_decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> DecayFunction:
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


def make_decay_function_lognormal(median: float, sigma: float) -> DecayFunction:
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
