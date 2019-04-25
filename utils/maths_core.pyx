"""
===========================
Cythonised versions of math functions.
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

import cython
from libc.math cimport exp, sqrt, pi, log, erf

TAU: cython.float = 2 * pi


def exponential_decay(age: cython.uint,
                      original_activation: cython.float,
                      decay_factor: cython.float) -> cython.float:
    """
    :param age:
    :param original_activation:
    :param decay_factor:
    :return:
    """
    return original_activation * (decay_factor ** age)


def gaussian_decay(age: cython.uint,
                   original_activation: cython.float,
                   height_coef: cython.float,
                   reset_height: cython.float,
                   centre: cython.float,
                   sd: cython.float,
                   ) -> cython.float:
    """
    :param age:
    :param original_activation:
    :param height_coef:
    :param reset_height:
        Should be sqrt(TAU * sd * sd)
    :param centre:
    :param sd:
    :return:
    """
    return original_activation * height_coef * reset_height * gaussian_pdf(x=age, mu=centre, sd=sd)


def gaussian_pdf(x: cython.float, mu: cython.float, sd: cython.float) -> cython.float:
    """
    Cythonised version of scipy.stats.norm.pdf for scalars.
    :param x:
    :param mu:
    :param sd:
    :return:
    """
    y:        cython.float = (x - mu) / sd
    exponent: cython.float = (-1) * y * y / 2
    denom:    cython.float = sqrt(TAU)

    e_term:   cython.float = exp(exponent) / denom

    return e_term / sd

def lognormal_sf(x: cython.float, sigma: cython.float) -> cython.float:
    """
    Cythonised approximation of the lognormal sf.
    Assumes mu is 0.
    :param x:
    :param sigma:
    :return:
    """
    return 1 - lognormal_cdf(x, sigma)


def lognormal_cdf(x: cython.float, sigma: cython.float) -> cython.float:
    """
    Cythonised approximation of the lognormal cdf.
    Assumes mu is 0.
    :param x:
    :param sigma:
    :return:
    """
    numerator:   cython.float = log(x)
    denomenator: cython.float = sqrt(2) * sigma
    fraction:    cython.float = numerator / denomenator
    return 0.5 + 0.5 * erf(fraction)


def lognormal_pdf(x: cython.float, sigma: cython.float) -> cython.float:
    """
    Cythonised approximation of the lognormal pdf.
    Assumes mu is 0.
    :param x:
    :param sigma:
    :return:
    """
    coefficient: cython.float = 1 / (sigma * x * sqrt(TAU))
    log_term:    cython.float = log(x)
    numerator:   cython.float = log_term * log_term
    denomenator: cython.float = 2 * sigma * sigma
    exponent:    cython.float = (-1) * numerator / denomenator
    return coefficient * exp(exponent)
