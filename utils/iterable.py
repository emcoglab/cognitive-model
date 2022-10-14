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


def all_and_any(iterable: Iterable) -> bool:
    """
    Equivalent to `all(iterable)` for nonempty iterables, but returns False for empty iterables
    (where `all` would return True, as it should).

    Equivalent to `all(iterable) and any(iterable)` but only traverses the iterable once.
    """
    at_least_one = False
    for element in iterable:
        if not element:
            return False
        at_least_one = True
    return at_least_one
