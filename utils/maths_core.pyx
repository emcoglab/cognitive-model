"""
===========================
Cythonised versions of math functions.
Compile using:
    python setup.py build_ext --inplace
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

# cython.double matches the precision of python's float

TAU: cython.double = 2 * pi


def exponential_decay(age: cython.uint,
                      original_activation: cython.double,
                      decay_factor: cython.double) -> cython.double:
    """
    :param age:
    :param original_activation:
    :param decay_factor:
    :return:
    """
    return original_activation * (decay_factor ** age)


def gaussian_decay(age: cython.uint,
                   original_activation: cython.double,
                   height_coef: cython.double,
                   reset_height: cython.double,
                   centre: cython.double,
                   sd: cython.double,
                   ) -> cython.double:
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


def gaussian_pdf(x: cython.double, mu: cython.double, sd: cython.double) -> cython.double:
    """
    Cythonised version of scipy.stats.norm.pdf for scalars.
    :param x:
    :param mu:
    :param sd:
    :return:
    """
    y:        cython.double = (x - mu) / sd
    exponent: cython.double = (-1) * y * y / 2
    denom:    cython.double = sqrt(TAU)

    e_term:   cython.double = exp(exponent) / denom

    return e_term / sd


def lognormal_sf(x: cython.double, mu: cython.double = 0, sigma: cython.double = 1) -> cython.double:
    """
    Cythonised approximation of the lognormal sf.
    Assumes mu is 0.
    :param x:
    :param mu:
        Mean of the log of the random variable.
    :param sigma:
        SD of the log of the random variable.
    :return:
    """
    return 1 - lognormal_cdf(x, mu, sigma)


def lognormal_cdf(x: cython.double, mu: cython.double = 0, sigma: cython.double = 1) -> cython.double:
    """
    Cythonised approximation of the lognormal cdf.
    :param x:
    :param mu:
        Mean of the log of the random variable.
    :param sigma:
        SD of the log of the random variable.
    :return:
    """
    numerator:   cython.double = log(x) - mu
    denomenator: cython.double = sqrt(2) * sigma
    fraction:    cython.double = numerator / denomenator
    return 0.5 + 0.5 * erf(fraction)


def lognormal_pdf(x: cython.double, mu: cython.double = 0, sigma: cython.double = 1) -> cython.double:
    """
    Cythonised approximation of the lognormal pdf.
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    coefficient: cython.double = 1 / (sigma * x * sqrt(TAU))
    log_term:    cython.double = log(x) - mu
    numerator:   cython.double = log_term * log_term
    denomenator: cython.double = 2 * sigma * sigma
    exponent:    cython.double = (-1) * numerator / denomenator
    return coefficient * exp(exponent)
