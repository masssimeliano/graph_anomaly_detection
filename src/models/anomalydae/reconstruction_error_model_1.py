from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from src.helpers.config.training_config import *
from src.models.base_train import base_train
from src.models.encoder.node_feature_autoencoder import NodeFeatureAutoencoder


def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          data_set: str):
    normalize_node_features_minmax(graph)
    compare_anomaly_reconstruction_error(graph, labels_dict[data_set])
    add_structure_features(graph)
    di_graph = from_networkx(graph)

    base_train(di_graph,
               labels,
               title_prefix="Attr + Error1",
               learning_rate=learning_rate,
               hid_dim=hid_dim,
               data_set=data_set)


def normalize_node_features_minmax(graph: nx.Graph):
    features = [graph.nodes[n]['x'] for n in graph.nodes()]
    stacked = torch.stack(features)
    min_vals = stacked.min(dim=0)[0]
    max_vals = stacked.max(dim=0)[0]
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    for n in graph.nodes():
        graph.nodes[n]['x'] = (graph.nodes[n]['x'] - min_vals) / range_vals


def extract_error_features(graph: nx.Graph) -> torch.Tensor:
    print("Extracting node error rates...")
    features = [graph.nodes[node]['x'] for node in graph.nodes()]
    features_tensor = torch.stack(features).float()

    model = NodeFeatureAutoencoder(in_dim=features_tensor.shape[1], hid_dim=16)
    model.eval()
    with torch.no_grad():
        reconstructed = model(features_tensor)
        errors = F.mse_loss(reconstructed, features_tensor, reduction='none').mean(dim=1)
    return errors


def add_structure_features(graph: nx.Graph):
    errors = extract_error_features(graph)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        error_feat = torch.tensor([errors[i].item()], dtype=torch.float32)
        graph.nodes[node]['x'] = torch.cat([original_feat, error_feat])


# method for checking mean and median reconstruction errors of nodes
def compare_anomaly_reconstruction_error(graph: nx.Graph, labels: List[int]):
    print("Comparing reconstruction error between normal and anomalous nodes...")

    features = [graph.nodes[node]['x'] for node in graph.nodes()]
    features_tensor = torch.stack(features).float()

    model = NodeFeatureAutoencoder(in_dim=features_tensor.shape[1], hid_dim=16)
    model.eval()

    with torch.no_grad():
        reconstructed = model(features_tensor)
        errors = F.mse_loss(reconstructed, features_tensor, reduction='none').mean(dim=1).cpu().numpy()

    labels = np.array(labels)
    errors = np.array(errors)

    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    print(f"Normal nodes:")
    print(f"  Mean error:   {normal_errors.mean():.6f}")
    print(f"  Median error: {np.median(normal_errors):.6f}")

    print(f"Anomalous nodes:")
    print(f"  Mean error:   {anomaly_errors.mean():.6f}")
    print(f"  Median error: {np.median(anomaly_errors):.6f}")

    print("")
