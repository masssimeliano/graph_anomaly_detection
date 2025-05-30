import networkx as nx
from typing import List

from src.models.emd_train_1 import emd_train


def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          data_set: str):
    emd_train(graph,
              labels,
              "Attr + Emd1",
              learning_rate,
              hid_dim,
              data_set)