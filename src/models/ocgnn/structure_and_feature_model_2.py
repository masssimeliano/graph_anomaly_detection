import logging
from typing import List

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_STR2
from src.helpers.time.timed import timed
from src.models.ocgnn.base_train import base_train


def train(
        nx_graph: nx.Graph,
        labels: List[int],
        learning_rate: float,
        hid_dim: int,
        dataset: str,
):
    add_structure_features(nx_graph=nx_graph)
    di_graph = from_networkx(G=nx_graph)

    base_train(
        di_graph=di_graph,
        labels=labels,
        title_prefix=FEATURE_LABEL_STR2,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )


def extract_node_features_tensor(nx_graph: nx.Graph) -> torch.Tensor:
    features = []
    average_neighbour_degree = nx.average_neighbor_degree(nx_graph)
    square_clust = nx.square_clustering(nx_graph)

    for node in nx_graph.nodes():
        ego = nx.ego_graph(nx_graph, node)

        degree = nx_graph.degree(node)
        clustering = nx.clustering(nx_graph, node)
        triangle_count = nx.triangles(nx_graph, node)
        ego_density = nx.density(ego)
        square_clustering = square_clust[node]

        node_features = [
            degree,
            clustering,
            triangle_count,
            average_neighbour_degree[node],
            ego_density,
            square_clustering,
        ]
        features.append(node_features)

    features_tensor = torch.tensor(features, dtype=torch.float32)

    return features_tensor


@timed
def add_structure_features(nx_graph: nx.Graph):
    logging.info("Extracting node features with NetworkX 1...")

    structural_features = extract_node_features_tensor(nx_graph)

    for i, node in enumerate(nx_graph.nodes()):
        original_node_features = nx_graph.nodes[node]["x"]
        nx_graph.nodes[node]["x"] = torch.cat(
            [original_node_features, structural_features[i]]
        )
