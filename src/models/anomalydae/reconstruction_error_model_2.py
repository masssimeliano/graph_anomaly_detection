import networkx as nx
from typing import List

import torch

from src.models.reconstruction_train import reconstruction_train


def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          data_set: str):
    normalize_node_features_minmax(graph)
    reconstruction_train(graph,
              labels,
              "Attr + Error2",
              learning_rate,
              hid_dim,
              data_set)

def normalize_node_features_minmax(graph: nx.Graph):
    features = [graph.nodes[n]['x'] for n in graph.nodes()]
    stacked = torch.stack(features)
    min_vals = stacked.min(dim=0)[0]
    max_vals = stacked.max(dim=0)[0]
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    for n in graph.nodes():
        graph.nodes[n]['x'] = (graph.nodes[n]['x'] - min_vals) / range_vals