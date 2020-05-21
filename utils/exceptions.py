"""
===========================
Exception classes.
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


class ParseError(Exception):
    """Represents an error in parsing a string."""
    pass


class ItemNotFoundError(KeyError):
    """Represents an item not being found in a model or component."""
    pass
