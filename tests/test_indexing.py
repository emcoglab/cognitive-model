"""
===========================
Tests for filtering functions.
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

import unittest

from ..utils.indexing import list_index_dictionaries


class TestListIndexDictionaryExample(unittest.TestCase):

    entries = [3, 5, 10]
    entry2idx, idx2entry = list_index_dictionaries(entries)

    def test_dictionary_lengths(self):
        self.assertEquals(len(self.entry2idx), 3)
        self.assertEquals(len(self.idx2entry), 3)

    def test_inversion(self):
        for entry in self.entries:
            self.assertEquals(entry, self.idx2entry[self.entry2idx[entry]])


if __name__ == '__main__':
    unittest.main()
