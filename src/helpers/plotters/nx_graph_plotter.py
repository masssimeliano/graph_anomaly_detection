"""
nx_graph_plotter.py
This file contains custom transfer method from "Graph" to NX-Graph.
The NX-Graph can be visualized if its wanted.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.helpers.time.timed import timed
from src.structure.graph import Graph


def visualize_graph(nx_graph: nx.Graph, node_color: list[str], title: str):
    logging.info("Visualizing NX graph...")

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(nx_graph, seed=42, k=0.15)

    nx.draw(
        nx_graph,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_color=node_color,
        node_size=20,
        edge_color="gray",
        width=0.3,
        alpha=0.7,
    )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


@timed
def to_networkx_graph(
    graph: Graph, do_visualize: bool = False, title: str = "Graph visualization"
) -> nx.Graph:
    logging.info("Forming a NX graph...")

    nx_graph = nx.Graph()
    node_color = []

    for node in graph.nodes:
        nx_graph.add_node(node.id, x=torch.tensor(node.features, dtype=torch.float))
        if do_visualize:
            if node.is_attr_anomaly:
                node_color.append("red")
            elif node.is_str_anomaly:
                node_color.append("blue")
            else:
                node_color.append("green")

    for node in graph.nodes:
        for neighbour in node.neighbours:
            nx_graph.add_edge(node.id, neighbour.id)

    if do_visualize:
        visualize_graph(nx_graph, node_color, title)

    return nx_graph
