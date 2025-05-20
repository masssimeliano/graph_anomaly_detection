import time

import numpy as np
import torch.nn.functional as F
import torch
import networkx as nx
from typing import List

from torch_geometric.utils import from_networkx

from src.models.base_train import base_train
from src.models.emd_train import emd_train
from src.models.encoder.node_feature_autoencoder import NodeFeatureAutoencoder


def train(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    data_set: str
):
    add_structure_features(graph)
    di_graph = from_networkx(graph)

    base_train(
        di_graph,
        labels,
        title_prefix="Attr + Error",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        data_set=data_set
    )

def extract_error_features(graph: nx.Graph) -> torch.Tensor:
    print("Extracting node error rates...")
    features = [graph.nodes[node]['x'] for node in graph.nodes()]
    features_tensor = torch.stack(features).float()

    normalized = (features_tensor - features_tensor.mean(dim=0)) / (features_tensor.std(dim=0))

    model = NodeFeatureAutoencoder(in_dim=normalized.shape[1], hid_dim=16)
    model.eval()
    with torch.no_grad():
        reconstructed = model(normalized)
        errors = F.mse_loss(reconstructed, normalized, reduction='none').mean(dim=1)
    return errors

def add_structure_features(graph: nx.Graph):
    errors = extract_error_features(graph)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        error_feat = torch.tensor([errors[i].item()], dtype=torch.float32)
        graph.nodes[node]['x'] = torch.cat([original_feat, error_feat])