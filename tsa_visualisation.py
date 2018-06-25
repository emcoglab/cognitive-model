"""
===========================
Visualisation for TemporalSpreadingActivations.
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
import logging

import networkx
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

from model.temporal_spreading_activation import TemporalSpreadingActivation, EdgeDataKey

logger = logging.getLogger(__name__)


def draw_graph(tsa: TemporalSpreadingActivation, pdf=None, pos=None, frame_label=None):
    """Draws and saves or shows the graph."""

    # Use supplied position, or recompute
    if pos is None:
        pos = networkx.spring_layout(tsa.graph, iterations=500)

    cmap = pyplot.get_cmap("autumn")

    # Prepare labels

    node_labels = {}
    for n in tsa.graph.nodes:
        node_labels[n] = f"{tsa.node2label[n]}\n{tsa.activation_of_node(n):.3g}"

    edge_labels = {}
    for v1, v2, e_data in tsa.graph.edges(data=True):
        weight = e_data[EdgeDataKey.WEIGHT]
        length = e_data[EdgeDataKey.LENGTH]
        edge_labels[(v1, v2)] = f"w={weight:.3g}; l={length}"

    # Prepare impulse points and labels
    impulse_data = []
    for v1, v2, e_data in tsa.graph.edges(data=True):
        length = e_data[EdgeDataKey.LENGTH]
        impulses_this_edge = tsa.impulses_by_edge(v1, v2)
        if len(impulses_this_edge) == 0:
            continue
        x1, y1 = pos[v1]
        x2, y2 = pos[v2]
        for impulse in impulses_this_edge:

            age = impulse.age_at_time(tsa.clock)

            # Skip just-created impulses
            if age == 0:
                continue

            if impulse.target_node == v2:
                # Travelling v1 → v2
                fraction = age / length
            elif impulse.target_node == v1:
                # Travelling v2 → v1
                fraction = 1 - (age / length)
            else:
                raise Exception(f"Inappropriate target node {impulse.target_node}")
            x = x1 + (fraction * (x2 - x1))
            y = y1 + (fraction * (y2 - y1))

            colour = cmap(tsa.node_decay_function(age, impulse.departure_activation))

            impulse_data.append([x, y, colour, impulse, length])

    pyplot.figure()

    # Draw the nodes
    networkx.draw_networkx_nodes(
        tsa.graph, pos=pos, with_labels=False,
        node_color=[tsa.activation_of_node(n) for n in tsa.graph.nodes],
        cmap=cmap, vmin=0, vmax=1,
        node_size=400)
    networkx.draw_networkx_labels(tsa.graph, pos=pos, labels=node_labels)

    # Draw the edges
    networkx.draw_networkx_edges(
        tsa.graph, pos=pos, with_labels=False,
    )
    networkx.draw_networkx_edge_labels(tsa.graph, pos=pos, edge_labels=edge_labels, font_size=6)

    # Draw impulses
    for x, y, colour, impulse, length in impulse_data:
        pyplot.plot(x, y, marker='o', markersize=5, color=colour)
        age = impulse.age_at_time(tsa.clock)
        pyplot.text(x, y, f"{tsa.node_decay_function(age, impulse.departure_activation):.3g} ({age}/{length})")

    # Draw frame_label
    if frame_label is not None:
        pyplot.annotate(frame_label,
                        horizontalalignment='left', verticalalignment='bottom',
                        xy=(1, 0), xycoords='axes fraction')

    # Style figure
    pyplot.axis('off')

    # Save or show graph
    if pdf is not None:
        pdf.savefig()
        pyplot.close()
    else:
        pyplot.show()

    return pos


def run_with_pdf_output(tsa: TemporalSpreadingActivation, n_steps: int, path: str):

    with PdfPages(path) as pdf:

        i = 0
        pos = draw_graph(tsa=tsa, pdf=pdf, frame_label=str(i))

        for i in range(1, n_steps+1):
            logger.info(f"CLOCK = {i}")
            tsa.tick()
            draw_graph(tsa=tsa, pdf=pdf, pos=pos, frame_label=str(i))
