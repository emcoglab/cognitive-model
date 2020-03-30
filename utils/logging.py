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

from logging import getLogger, basicConfig, INFO

logger = getLogger()
basicConfig(
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=INFO)
