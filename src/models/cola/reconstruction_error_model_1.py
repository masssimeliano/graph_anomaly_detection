import logging
from typing import List

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_ERROR1
from src.helpers.config.training_config import *
from src.helpers.time.timed import timed
from src.models.anomalydae.reconstruction_error_model_1 import compare_anomaly_reconstruction_error
from src.models.cola.base_train import base_train
from src.models.encoder.node_feature_autoencoder import NodeFeatureAutoencoder


def train(
    nx_graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    dataset: str,
    do_compare: bool = False,
):
    if do_compare:
        compare_anomaly_reconstruction_error(
            nx_graph=nx_graph, labels=labels_dict[dataset]
        )
    add_error_features(nx_graph=nx_graph)
    di_graph = from_networkx(G=nx_graph)

    base_train(
        di_graph=di_graph,
        labels=labels,
        title_prefix=FEATURE_LABEL_ERROR1,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )


def normalize_node_features_via_minmax_and_remove_nan(nx_graph: nx.Graph):
    node_features = [nx_graph.nodes[node]["x"] for node in nx_graph.nodes()]
    node_features_stacked = torch.stack(node_features)
    node_features_stacked_without_nan = torch.nan_to_num(
        input=node_features_stacked, nan=0.0
    )
    node_features_stacked_without_nan_min = node_features_stacked_without_nan.min(
        dim=0
    )[0]
    node_features_stacked_without_nan_max = node_features_stacked_without_nan.max(
        dim=0
    )[0]
    node_features_stacked_diff = (
        node_features_stacked_without_nan_max - node_features_stacked_without_nan_min
    )
    node_features_stacked_diff[node_features_stacked_diff == 0] = 1

    for n in nx_graph.nodes():
        nx_graph.nodes[n]["x"] = (
            nx_graph.nodes[n]["x"] - node_features_stacked_without_nan_min
        ) / node_features_stacked_diff


@timed
def extract_error_features(nx_graph: nx.Graph) -> torch.Tensor:
    logging.info("Extracting node error rates 1...")
    node_features = [nx_graph.nodes[node]["x"] for node in nx_graph.nodes()]
    node_features_tensor = torch.stack(node_features).float()

    model = NodeFeatureAutoencoder(in_dim=node_features_tensor.shape[1], hid_dim=16)
    model.eval()
    with torch.no_grad():
        reconstructed_node_features_tensor = model(node_features_tensor)
        node_errors = F.mse_loss(
            input=reconstructed_node_features_tensor,
            target=node_features_tensor,
            reduction="none",
        ).mean(dim=1)
    return node_errors


def add_error_features(nx_graph: nx.Graph):
    node_errors = extract_error_features(nx_graph)

    for i, node in enumerate(nx_graph.nodes()):
        original_node_features = nx_graph.nodes[node]["x"]
        node_error_features = torch.tensor(
            data=[node_errors[i].item()], dtype=torch.float32
        )
        nx_graph.nodes[node]["x"] = torch.cat(
            [original_node_features, node_error_features]
        )