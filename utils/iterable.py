"""
===========================
Utility functions for iterables.
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

from typing import Tuple, List, Iterable


def partition(iterable: Iterable, predicate: callable) -> Tuple[List, List]:
    """
    Separates the an iterable into two sub-lists; those which satisfy predicate and those which don't.
    Thanks to https://stackoverflow.com/a/4578605/2883198 and https://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition#comment24295861_12135169.
    """
    trues = []
    falses = []
    for item in iterable:
        trues.append(item) if predicate(item) else falses.append(item)
    return trues, falses
