"""
===========================
Tests for maths functions.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

import unittest

from scipy.stats import lognorm

from ..utils.maths_core import lognormal_sf, lognormal_pdf, lognormal_cdf
from ..utils.maths import scale01, affine_scale


class TestCython(unittest.TestCase):

    def test_lognormal_pdf(self):

        x = 3.2
        sigma = 2.2

        self.assertAlmostEqual(
            # Library version
            lognorm.pdf(x, s=sigma),
            # cythonised version
            lognormal_pdf(x, sigma)
        )

    def test_lognormal_cdf(self):

        x = 3.2
        sigma = 2.2

        self.assertAlmostEqual(
            # Library version
            lognorm.cdf(x, s=sigma),
            # cythonised version
            lognormal_cdf(x, sigma)
        )

    def test_lognormal_sf_3_2(self):

        x = 3
        sigma = 2

        self.assertAlmostEqual(
            # Library version
            lognorm.sf(x, s=sigma),
            # cythonised version
            lognormal_sf(x, sigma)
        )

    def test_lognormal_sf_30_20(self):

        x = 30
        sigma = 20

        self.assertAlmostEqual(
            # Library version
            lognorm.sf(x, s=sigma),
            # cythonised version
            lognormal_sf(x, sigma)
        )

    def test_lognormal_sf_300_200(self):

        x = 300
        sigma = 200

        self.assertAlmostEqual(
            # Library version
            lognorm.sf(x, s=sigma),
            # cythonised version
            lognormal_sf(x, sigma)
        )

    def test_lognormal_sf_03_02(self):

        x = 0.3
        sigma = 0.2

        self.assertAlmostEqual(
            # Library version
            lognorm.sf(x, s=sigma),
            # cythonised version
            lognormal_sf(x, sigma)
        )


class TestAffineScale(unittest.TestCase):

    def test_scale01_9_10(self):
        self.assertEqual(
            scale01((0, 10), 7),
            0.7
        )

    def test_affine_scale_n2_2_10_20_1(self):
        self.assertEqual(
            affine_scale((-2, 2), (10, 20), 1),
            17.5
        )

    def test_affine_inverted_n1_n3_n1_3_n25(self):
        self.assertEqual(
            affine_scale((-1, -3), (-1, 3), -2.5),
            2
        )

    def test_affine_scale_to_zero(self):
        self.assertEqual(
            affine_scale((2, 3), (4, 4), 4),
            4
        )

    def test_affine_scale_from_zero(self):
        with self.assertRaises(Exception):
            affine_scale((4, 4), (2, 3), 4)


if __name__ == '__main__':
    unittest.main()
