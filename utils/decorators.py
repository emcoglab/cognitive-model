"""
===========================
Useful function decorators.
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
from functools import wraps


def cached(func):
    """
    Decorate a function to cache its returned values.
    """
    cache = {}

    @wraps(func)
    def wrapped_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapped_func
