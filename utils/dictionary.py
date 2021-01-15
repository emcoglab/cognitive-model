"""
===========================
Utility functions for dictinoaries.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2021
---------------------------
"""

from typing import Dict, Callable, Any


def forget_keys_for_values_satisfying(dictionary: Dict, predicate: Callable[[Any, Any], bool]):
    """
    For a dict `dictionary` and a predicate `predicate` (mapping objects to bools), this function will delete all keys
    from the `dictionary` iff their associated values evaluate to `True`.
    :param: dictionary
    :param: predicate
        (k, v) |-> bool
    """
    keys_to_forget = [
        k
        for k, v in dictionary.items()
        if predicate(k, v)
    ]
    for k in keys_to_forget:
        del dictionary[k]
