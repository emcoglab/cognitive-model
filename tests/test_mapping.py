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
2021
---------------------------
"""

import unittest

from cognitive_model.combined_cognitive_model import InterComponentMapping


class TestMapping(unittest.TestCase):

    def test_no_choices_example(self):
        sensorimotor_vocab = {"caramel", "caramelise"}
        linguistic_vocab   = {"caramel", "caramelise"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab,
                                        ignore_identity_mapping=False)
        self.assertDictEqual(mapping.linguistic_to_sensorimotor, {"caramel": {"caramel"}, "caramelise": {"caramelise"}})


if __name__ == '__main__':
    unittest.main()
