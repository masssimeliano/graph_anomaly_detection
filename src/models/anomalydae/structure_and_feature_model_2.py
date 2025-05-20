import time

import numpy as np
import torch
import networkx as nx
from typing import List

from torch_geometric.utils import from_networkx

from src.models.base_train import base_train

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
        title_prefix="Attr + Str2",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        data_set=data_set
    )

def extract_node_features_tensor(graph: nx.Graph) -> torch.Tensor:
    print("Extracting node features with NetworkX 1...")
    start_time = time.time()

    features = []
    avg_neighbor_degree = nx.average_neighbor_degree(graph)
    square_clust = nx.square_clustering(graph)

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        ego = nx.ego_graph(graph, node)

        degree = graph.degree(node)
        clustering = nx.clustering(graph, node)
        triangle_count = nx.triangles(graph, node)
        avg_deg_of_neighbors = np.mean([graph.degree(n) for n in neighbors]) if neighbors else 0
        ego_density = nx.density(ego)
        square_clustering = square_clust[node]
        num_neighbors = len(neighbors)

        node_features = [
            degree,
            clustering,
            triangle_count,
            avg_neighbor_degree[node],
            avg_deg_of_neighbors,
            ego_density,
            square_clustering,
            num_neighbors,]
        features.append(node_features)

    features_tensor = torch.tensor(features, dtype=torch.float32)

    print(f"Execution time: {(time.time() - start_time):.4f} sec")

    return features_tensor

def add_structure_features(graph: nx.Graph):
    additional_feats = extract_node_features_tensor(graph)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        stat_feat = additional_feats[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, stat_feat])