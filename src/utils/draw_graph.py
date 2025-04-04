import networkx as nx
import torch
import matplotlib.pyplot as plt

from src.structure.graph import Graph

def draw(graph: Graph):
    nx_graph  = nx.Graph()

    node_color = []
    for node in graph.nodes:
        nx_graph.add_node(node.id, x=torch.tensor(node.features, dtype=torch.float))
        """ if (node.is_attr_anomaly):
            node_color.append("orange")
            continue
        if (node.is_str_anomaly):
            node_color.append("yellow")
            continue
        if (not node.is_str_anomaly and not node.is_attr_anomaly):
            node_color.append("blue")
            continue
        node_color.append("red") """

    for node in graph.nodes:
        for neighbour in node.neighbours:
            if not nx_graph.has_edge(node.id, neighbour.id):
                nx_graph.add_edge(node.id, neighbour.id)

    # pos = nx.spring_layout(nx_graph, seed=42, k=0.2)
    # nx.draw(nx_graph, pos=pos, with_labels=True, font_weight='light', font_size=8, node_color=node_color)
    # plt.show()

    return nx_graph