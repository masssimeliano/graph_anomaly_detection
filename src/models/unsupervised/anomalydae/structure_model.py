from typing import List
import networkx as nx
import pyfglt.fglt as fg
import torch
from src.models.unsupervised.base_train import base_train

def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          current_epoch: int,
          save_results: bool):
    extract_structure_features(graph)
    base_train(graph, labels, "Str", learning_rate, hid_dim, current_epoch, save_results)

def extract_structure_features(graph: nx.Graph):
    print("Adding features")
    F = fg.compute(graph)
    node_list = list(graph.nodes())
    F = F.reindex(node_list)

    for i, node in enumerate(node_list):
        graphlet_vals = F.iloc[i].values
        graphlet_tensor = torch.tensor(graphlet_vals, dtype=torch.float)
        graph.nodes[node]['x'] = graphlet_tensor