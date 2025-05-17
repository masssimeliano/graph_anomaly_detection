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
    save_emb: bool,
    data_set: str
):
    add_structure_features(graph)
    di_graph = from_networkx(graph)

    base_train(
        di_graph,
        labels,
        title_prefix="Attr + Str3",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        save_emb=save_emb,
        data_set=data_set
    )

def extract_node_features_tensor(graph: nx.Graph) -> torch.Tensor:
    print("Extracting node features with NetworkX 1...")
    start_time = time.time()

    features = []

    avg_neighbor_degree = nx.average_neighbor_degree(graph)
    square_clust = nx.square_clustering(graph)

    try:
        betweenness = nx.betweenness_centrality(graph)
    except Exception:
        betweenness = {n: 0.0 for n in graph.nodes}

    try:
        closeness = nx.closeness_centrality(graph)
    except Exception:
        closeness = {n: 0.0 for n in graph.nodes}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(graph)
    except Exception:
        eigenvector = {n: 0.0 for n in graph.nodes}

    try:
        pagerank = nx.pagerank(graph)
    except Exception:
        pagerank = {n: 0.0 for n in graph.nodes}

    try:
        core_number = nx.core_number(graph)
    except Exception:
        core_number = {n: 0 for n in graph.nodes}

    try:
        eccentricity = nx.eccentricity(graph)
    except Exception:
        eccentricity = {n: 0.0 for n in graph.nodes}

    articulation_points = set()
    try:
        articulation_points = set(nx.articulation_points(graph))
    except Exception:
        pass

    max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1

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

        # New features
        betw = betweenness.get(node, 0.0)
        close = closeness.get(node, 0.0)
        eig = eigenvector.get(node, 0.0)
        pr = pagerank.get(node, 0.0)
        core = core_number.get(node, 0)
        ecc = eccentricity.get(node, 0.0)
        is_cut_vertex = 1.0 if node in articulation_points else 0.0
        degree_ratio = degree / max_degree if max_degree > 0 else 0.0

        node_features = [
            degree,
            clustering,
            triangle_count,
            avg_neighbor_degree.get(node, 0.0),
            avg_deg_of_neighbors,
            ego_density,
            square_clustering,
            num_neighbors,
            betw,
            close,
            eig,
            pr,
            core,
            ecc,
            is_cut_vertex,
            degree_ratio]

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