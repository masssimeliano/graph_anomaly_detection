from typing import List
import networkx as nx
import torch_geometric
from sympy.abc import alpha

from src.models.unsupervised.base_train import base_train

def train(
    di_graph: torch_geometric.data.Data,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    save_results: bool,
    data_set: str
):
    base_train(
        di_graph,
        labels,
        title_prefix="Attr",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        save_results=save_results,
        data_set=data_set
    )
