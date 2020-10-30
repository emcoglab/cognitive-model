"""
===========================
Metadata for the test graph.
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

from os import path

test_materials_dir: str = path.dirname(path.realpath(__file__))

test_graph_file_path: str = path.join(test_materials_dir, "test_graph.edgelist")
test_graph_importance_file_path: str = path.join(test_materials_dir, "test_graph_importance.edgelist")
