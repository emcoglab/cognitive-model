"""
===========================
Manipulating paths.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""


from pathlib import Path
from typing import Union


def append_to_path_part(path: Path, append: str, at_index: int) -> Path:
    """
    Appends `append` to the `at_index` part of `path`.
    :raises: IndexError if `at_index` is too large for the number of parts of `path`.
    """
    parts = list(path.parts)
    parts[at_index] += append
    return Path(*parts)


def insert_part_to_path(path: Path, insert: Union[Path, str], at_index: int) -> Path:
    """
    Inserts `insert` to `path` at the `at_index` part.
    If `at_index` is too large, it goes at the end.
    If `insert` is an absolute path, it replaces everything before it in the returned path.
    """
    parts = list(path.parts)
    before = parts[:at_index]
    after = parts[at_index:]
    return Path(*before, insert, *after)


def append_to_path_name(path: Path, append: str) -> Path:
    """
    Appends `append` to the name (last part) of `path`.
    """
    return append_to_path_part(path=path, append=append, at_index=-1)
