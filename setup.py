"""
===========================
Run `python setup.py build_ext --inplace`.
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

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("utils/maths_core.pyx")
)
