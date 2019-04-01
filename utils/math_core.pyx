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
from libc.math cimport exp, sqrt, pi

TAU: cython.float = 2 * pi

def gaussian_decay(age: cython.int,
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

    ret_val:  cython.float = e_term / sd

    return ret_val
