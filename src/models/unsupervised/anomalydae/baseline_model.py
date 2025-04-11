from typing import List

import networkx as nx

from src.models.unsupervised.base_train import base_train


def train(graph: nx.Graph, labels: List[int],
               title_prefix: str = "",
               learning_rate: float = 0.005, hid_dim: int = 16,
               current_epoch: int = 100):
    base_train(graph, labels, "Attr", learning_rate, hid_dim, current_epoch)
