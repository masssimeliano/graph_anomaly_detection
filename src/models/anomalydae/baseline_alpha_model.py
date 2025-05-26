from typing import List
import torch_geometric

from src.models.base_train import base_train


def train(di_graph: torch_geometric.data.Data,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          data_set: str):
    base_train(di_graph,
               labels,
               title_prefix="Attr + Alpha",
               learning_rate=learning_rate,
               hid_dim=hid_dim,
               data_set=data_set,
               alpha=0)
