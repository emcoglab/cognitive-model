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


def partition(iterable, predicate):
    """
    Separates the an iterable into two sub-iterables; those which satisfy predicate and those which don't.
    Thanks to https://stackoverflow.com/a/4578605/2883198 and https://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition#comment24295861_12135169.
    """
    trues = []
    falses = []
    for item in iterable:
        trues.append(item) if predicate(item) else falses.append(item)
    return trues, falses


def set_partition(iterable, predicate):
    trues = set()
    falses = set()
    for item in iterable:
        trues.add(item) if predicate(item) else falses.add(item)
    return trues, falses
