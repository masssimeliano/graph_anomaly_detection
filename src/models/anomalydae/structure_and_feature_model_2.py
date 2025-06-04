import logging
import time
from typing import List

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_STR2
from src.helpers.time.timed import timed
from src.models.anomalydae.reconstruction_error_model_1 import normalize_node_features_via_minmax_and_remove_nan
from src.models.base_train import base_train

logging.basicConfig(level=logging.INFO)


def train(nx_graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          dataset: str):
    normalize_node_features_via_minmax_and_remove_nan(nx_graph=nx_graph)
    add_structure_features(nx_graph=nx_graph)
    di_graph = from_networkx(G=nx_graph)

    base_train(di_graph=di_graph,
               labels=labels,
               title_prefix=FEATURE_LABEL_STR2,
               learning_rate=learning_rate,
               hid_dim=hid_dim,
               dataset=dataset)


def extract_node_features_tensor(nx_graph: nx.Graph) -> torch.Tensor:
    logging.info("Extracting node features with NetworkX 1...")
    start_time = time.time()

    features = []
    avg_neighbor_degree = nx.average_neighbor_degree(nx_graph)
    square_clust = nx.square_clustering(nx_graph)

    for node in nx_graph.nodes():
        neighbors = list(nx_graph.neighbors(node))
        ego = nx.ego_graph(nx_graph, node)

        degree = nx_graph.degree(node)
        clustering = nx.clustering(nx_graph, node)
        triangle_count = nx.triangles(nx_graph, node)
        avg_deg_of_neighbors = np.mean([nx_graph.degree(n) for n in neighbors]) if neighbors else 0
        ego_density = nx.density(ego)
        square_clustering = square_clust[node]
        num_neighbors = len(neighbors)

        node_features = [degree,
                         clustering,
                         triangle_count,
                         avg_neighbor_degree[node],
                         avg_deg_of_neighbors,
                         ego_density,
                         square_clustering,
                         num_neighbors, ]
        features.append(node_features)

    features_tensor = torch.tensor(features, dtype=torch.float32)

    return features_tensor


@timed
def add_structure_features(nx_graph: nx.Graph):
    logging.info("Extracting node features with NetworkX 1...")

    additional_feats = extract_node_features_tensor(nx_graph)

    for i, node in enumerate(nx_graph.nodes()):
        original_feat = nx_graph.nodes[node]['x']
        stat_feat = additional_feats[i]
        nx_graph.nodes[node]['x'] = torch.cat([original_feat, stat_feat])
