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
import math

TAU = math.tau


def mean(*items):
    return sum(items) / len(items)
