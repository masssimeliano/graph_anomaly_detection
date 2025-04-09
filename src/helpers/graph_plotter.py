import networkx as nx
import torch
import matplotlib.pyplot as plt
from src.structure.graph import Graph


def to_networkx_graph(graph: Graph, visualize=False):
    print("Converting to networkx.Graph")

    nx_graph = nx.Graph()

    node_color = []

    for node in graph.nodes:
        nx_graph.add_node(node.id, x=torch.tensor(node.features, dtype=torch.float))

        if visualize:
            if node.is_attr_anomaly:
                node_color.append("orange")
            elif node.is_str_anomaly:
                node_color.append("yellow")
            else:
                node_color.append("blue")

    for node in graph.nodes:
        for neighbour in node.neighbours:
            nx_graph.add_edge(node.id, neighbour.id)

    if visualize:
        pos = nx.spring_layout(nx_graph, seed=42, k=0.2)
        nx.draw(nx_graph, pos=pos, with_labels=True, font_size=8, node_color=node_color)
        plt.show()

    return nx_graph