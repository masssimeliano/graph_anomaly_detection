import time

import networkx as nx
import torch
import matplotlib.pyplot as plt

from src.structure.graph import Graph


def visualize_graph(nx_graph: nx.Graph, node_color: list[str], title):
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(nx_graph, seed=42, k=0.15)

    nx.draw(nx_graph,
            pos=pos,
            ax=ax,
            with_labels=False,
            node_color=node_color,
            node_size=20,
            edge_color='gray',
            width=0.3,
            alpha=0.7)

    ax.set_title(title,
            fontsize=20,
            fontweight='bold',
            pad=20)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def to_networkx_graph(graph: Graph,
                      visualize: bool = False,
                      title: str = "Graph Visualization") -> nx.Graph:
    print("Forming a NetworkX graph...")

    start_time = time.time()

    nx_graph = nx.Graph()
    node_color = []

    for node in graph.nodes:
        nx_graph.add_node(node.id, x=torch.tensor(node.features, dtype=torch.float))
        if visualize:
            if node.is_attr_anomaly:
                node_color.append("red")
            elif node.is_str_anomaly:
                node_color.append("blue")
            else:
                node_color.append("green")

    for node in graph.nodes:
        for neighbour in node.neighbours:
            nx_graph.add_edge(node.id, neighbour.id)

    if visualize:
        visualize_graph(nx_graph, node_color, title)

    print(f"Execution time: {(time.time() - start_time):.4f} sec")

    return nx_graph