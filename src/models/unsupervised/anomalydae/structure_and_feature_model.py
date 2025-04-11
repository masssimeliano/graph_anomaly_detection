from typing import List

import networkx as nx
import pyfglt.fglt as fg
import torch

from src.models.unsupervised.base_train import base_train


def train(graph: nx.Graph, labels: List[int],
               title_prefix: str = "",
               learning_rate: float = 0.005, hid_dim: int = 16,
               current_epoch: int = 100):
    extract_structure_features(graph)

    base_train(graph, labels, "Attr + Str", learning_rate, hid_dim, current_epoch)


def extract_structure_features(graph: nx.Graph):
    print("Adding features")

    F = fg.compute(graph)

    node_list = list(graph.nodes())
    F = F.reindex(node_list)

    for i, node in enumerate(node_list):
        features = graph.nodes[node]['x']

        graphlet_vals = F.iloc[i].values
        graphlet_tensor = torch.tensor(graphlet_vals, dtype=torch.float)

        graph.nodes[node]['x'] = torch.cat([features, graphlet_tensor])
