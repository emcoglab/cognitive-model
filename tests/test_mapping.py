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

from ..combined_cognitive_model import InterComponentMapping


class TestMapping(unittest.TestCase):

    def test_no_choices_example(self):
        sensorimotor_vocab = {"caramel", "caramelise"}
        linguistic_vocab   = {"caramel", "caramelise"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=False)
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
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=False)
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
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=False)
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

    # TODO: test_multiple_norms_single_linguistic
    #  (There are literally no examples in the model where this happens, because ADVERTIZE is not a word and thus does
    #  not appear in the dictionary.)

    def test_multiple_norms_multiple_linguistic_with_preference(self):
        sensorimotor_vocab = {"anaesthetise", "anesthetise"}
        linguistic_vocab   = {"anaesthetise", "anesthetise", "anaesthetize", "anesthetize"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab,
                                        ignore_identity_mapping=False)
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "anaesthetise": {"anaesthetise"},
                "anesthetise":  {"anaesthetise"},
            }
        )
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {
                "anaesthetise": {"anaesthetise"},
                "anesthetise":  {"anesthetise"},
                "anaesthetize": {"anaesthetise"},
                "anesthetize":  {"anaesthetise"},
            }
        )

    # TODO test_multiple_norms_multiple_linguistic_without_preference
    #  (I think there are no examples of this in the model either.)

    def test_lemmatisation(self):
        sensorimotor_vocab = {"cat", "run"}
        linguistic_vocab = {"cat", "cats", "run", "running"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=False)
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "cat": {"cat"},
                "run":  {"run"},
            }
        )
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {
                "cat": {"cat"},
                "cats":  {"cat"},
                "run": {"run"},
                "running":  {"run"},
            }
        )

    def test_tokenisation(self):
        sensorimotor_vocab = {"part of a pharmacy"}
        linguistic_vocab = {"part", "of", "a", "pharmacy", "chemist"}
        mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=True)
        self.assertDictEqual(
            mapping.sensorimotor_to_linguistic,
            {
                "part of a pharmacy": {"part", "pharmacy", "chemist"}
            }
        )
        self.assertDictEqual(
            mapping.linguistic_to_sensorimotor,
            {}
        )


if __name__ == '__main__':
    unittest.main()
