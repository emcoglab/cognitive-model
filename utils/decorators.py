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


def cache(func):
    """
    Decorate a function to cache its returned values.

    Kept in here for compatibility with Python versions < 3.9.
    (Otherwise just use functools.cache().)
    """
    cache_ = {}

    @wraps(func)
    def wrapped_func(*args):
        if args in cache_:
            return cache_[args]
        result = func(*args)
        cache_[args] = result
        return result

    return wrapped_func
