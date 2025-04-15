import torch
import networkx as nx
from typing import List

from torch_geometric.utils import from_networkx

from src.helpers.loaders.emd_loader import load_emd_model
from src.models.unsupervised.base_train import base_train

def train(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    save_results: bool,
    data_set: str
):
    extract_embedding_features(graph, learning_rate, hid_dim, data_set)
    base_train(from_networkx(graph), labels, "Emd + Feature", learning_rate, hid_dim, save_results, data_set)

def extract_embedding_features(
    graph: nx.Graph,
    learning_rate: float,
    hid_dim: int,
    data_set: str
):
    print("Adding embedding features to graph nodes...")
    emd_model = load_emd_model(
        data_set=data_set,
        feature="Attr",
        lr=learning_rate,
        hid_dim=hid_dim,
        epoch=25
    )

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        embedding = emd_model[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, embedding]).detach().clone()
