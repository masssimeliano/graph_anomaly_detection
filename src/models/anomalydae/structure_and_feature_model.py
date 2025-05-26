import time

import torch
import networkx as nx
import pyfglt.fglt as fg
from typing import List

from torch_geometric.utils import from_networkx

from src.models.base_train import base_train


def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          data_set: str):
    add_structure_features(graph)
    di_graph = from_networkx(graph)

    base_train(di_graph,
               labels,
               title_prefix="Attr + Str",
               learning_rate=learning_rate,
               hid_dim=hid_dim,
               data_set=data_set)

def add_structure_features(graph: nx.Graph):
    print("Adding structural graphlet features to graph nodes...")
    start_time = time.time()

    F = fg.compute(graph).reindex(list(graph.nodes()))

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        graphlet_feat = torch.tensor(F.iloc[i].values, dtype=torch.float)
        graph.nodes[node]['x'] = torch.cat([original_feat, graphlet_feat])

    print(f"Execution time: {(time.time() - start_time):.4f} sec")