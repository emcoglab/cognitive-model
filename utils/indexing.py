"""
===========================
Helper functions for dealing with indexing and filtering.
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

from typing import List, Tuple, Dict


def list_index_dictionaries(entries: List) -> Tuple[Dict, Dict]:
    """
    Given a list, this will return a pair of dictionaries which translate between the list entries themselves, and their
    indices in the list.

    So if entries = [3, 5, 10], this would give:

        entry2idx = {3:0, 5:1, 10:2}
        idx2entry = {0:3, 1:5, 2:10}

    This can be useful if you have a list of ids which gets filtered, and you need to translate between the ids and
    their indices in the filtered list.

    :param entries:
    :return: A tuple of dictionaries:
        entry2idx
            A dictionary which maps a filtered index to its index in the specific filtered matrix
        idx2entry
            A lookup dictionary which converts the index of a word in the specific filtered distance matrix to its index in the LDM
    """
    entry2idx = {}
    idx2entry = {}
    for subseq_pos, filtered_id in enumerate(entries):
        entry2idx[filtered_id] = subseq_pos
        idx2entry[subseq_pos] = filtered_id
    return entry2idx, idx2entry
