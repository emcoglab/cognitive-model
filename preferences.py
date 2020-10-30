"""
===========================
Analysis preferences for spreading activation models.
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


class Preferences(object):
    """
    Global preferences for spreading activation models.
    """

    data = "/Volumes/Data"
    local_data = "/Users/caiwingfield/Data"

    # Paths

    email_connection_details_path = "/Users/caiwingfield/Resilio Sync/Resilio Sync Lancaster/notify@cwcomplex.net.txt"
    target_email_address = "c.wingfield@lancaster.ac.uk"

    graphs_dir = path.join(local_data, "graphs")

    node_distributions_dir = path.join(data, "node_distributions/")

    output_dir = path.join(data, "spreading activation model/Model output/")
    results_dir = path.join(data, "spreading activation model/Evaluation/")
    ancillary_dir = path.join(data, "spreading activation model/Ancillary results/")

    figures_dir = path.join(data, "spreading activation model/Figures/")

    graph_sizes = [
        1_000,
        3_000,
        10_000,
        30_000,
        40_000,
        60_000,
    ]

    min_edges_per_node = 10
