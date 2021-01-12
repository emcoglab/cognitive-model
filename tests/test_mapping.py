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
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "caramel": {"caramel"},
                "caramelise": {"caramelise"},
            })
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {
                "caramel": {"caramel"},
                "caramelise": {"caramelise"},
            })

    def test_single_sensorimotor_multiple_linguistic_with_preference(self):
        sensorimotor_vocab = {"colour"}
        linguistic_vocab   = {"colour", "color"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab,
                                        ignore_identity_mapping=False)
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "colour": {"colour"},
            }
        )
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {
                "colour": {"colour"},
                "color": {"colour"},
            }
        )

    def test_single_sensorimotor_multiple_linguistic_with_no_preference(self):
        sensorimotor_vocab = {"judgement"}
        linguistic_vocab   = {"judgement", "judgment"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab,
                                        ignore_identity_mapping=False)
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "judgement": {"judgement", "judgment"},
            }
        )
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {
                "judgement": {"judgement"},
                "judgment": {"judgement"},
            }
        )


if __name__ == '__main__':
    unittest.main()
