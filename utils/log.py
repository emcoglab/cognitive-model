"""
===========================
Shared logging.
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

from sys import stdout
from logging import getLogger, basicConfig, INFO

logger = getLogger()
basicConfig(
    stream=stdout,
    format='%(asctime)s | %(levelname)s | %(funcName)s @ %(module)s:%(lineno)d |➤ %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=INFO,
)
