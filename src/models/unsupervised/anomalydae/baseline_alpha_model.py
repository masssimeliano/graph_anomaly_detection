from typing import List
import networkx as nx
from sympy.abc import alpha

from src.models.unsupervised.base_train import base_train

def train(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    current_epoch: int,
    save_results: bool,
    data_set: str
):
    base_train(
        graph,
        labels,
        title_prefix="Attr + Alpha",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        current_epoch=current_epoch,
        save_results=save_results,
        data_set=data_set,
        alpha=1
    )
