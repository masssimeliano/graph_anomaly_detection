from typing import List
import networkx as nx
from src.models.unsupervised.base_train import base_train

def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          current_epoch: int):
    base_train(graph, labels, "Attr", learning_rate, hid_dim, current_epoch)