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
2021
---------------------------
"""


class Preferences(object):
    """
    Global preferences for spreading activation models.
    """

    # Local import to prevent it being accidentally imported form this module
    from .config import Config

    # Static config
    _config: Config = Config()

    # Notification emails
    email_connection_details_path = _config.value_by_key_path("email_connection_details")
    target_email_address = _config.value_by_key_path("target_email_address")

    # Paths
    graphs_dir = _config.value_by_key_path("graphs_dir")
    node_distributions_dir = _config.value_by_key_path("node_distributions_dir")
    output_dir = _config.value_by_key_path("output_dir")
    results_dir = _config.value_by_key_path("results_dir")
    ancillary_dir = _config.value_by_key_path("ancillary_dir")
    figures_dir = _config.value_by_key_path("figures_dir")

    # Sizes of graphs to test
    graph_sizes = [
        1_000,
        3_000,
        10_000,
        30_000,
        40_000,
        60_000,
    ]

    # Minimum edges to retain per node when pruning
    min_edges_per_node = 10
