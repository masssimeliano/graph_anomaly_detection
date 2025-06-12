"""
structure_and_feature_model_3.py
This file contains train wrapper for the model "Attr + Str3".
It also contains structural attribute extraction method.
"""

import logging
from typing import List

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_STR3
from src.helpers.time.timed import timed
from src.models.cola.base_train import base_train


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
        title_prefix=FEATURE_LABEL_STR3,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )


def extract_node_features_tensor(graph: nx.Graph) -> torch.Tensor:
    features = []

    average_neighbour_degree = nx.average_neighbor_degree(graph)
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

    max_degree = (
        max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1
    )

    for node in graph.nodes():
        node_neighbours = list(graph.neighbors(node))
        ego = nx.ego_graph(graph, node)

        degree = graph.degree(node)
        clustering = nx.clustering(graph, node)
        triangle_count = nx.triangles(graph, node)
        ego_density = nx.density(ego)
        square_clustering = square_clust[node]
        num_neighbours = len(node_neighbours)

        # new features
        betweenness_node = betweenness.get(node, 0.0)
        closeness_node = closeness.get(node, 0.0)
        eigenvector_node = eigenvector.get(node, 0.0)
        pagerank_node = pagerank.get(node, 0.0)
        core = core_number.get(node, 0)
        eccentricity_node = eccentricity.get(node, 0.0)
        is_cut_vertex = 1.0 if node in articulation_points else 0.0
        degree_ratio = degree / max_degree if max_degree > 0 else 0.0

        node_features = [
            degree,
            clustering,
            triangle_count,
            average_neighbour_degree.get(node, 0.0),
            ego_density,
            square_clustering,
            num_neighbours,
            betweenness_node,
            closeness_node,
            eigenvector_node,
            pagerank_node,
            core,
            eccentricity_node,
            is_cut_vertex,
            degree_ratio,
        ]
        features.append(node_features)

    features_tensor = torch.tensor(features, dtype=torch.float32)

    return features_tensor


@timed
def add_structure_features(nx_graph: nx.Graph):
    logging.info("Extracting node features with NetworkX 2...")

    structural_features = extract_node_features_tensor(nx_graph)

    for i, node in enumerate(nx_graph.nodes()):
        original_node_features = nx_graph.nodes[node]["x"]
        nx_graph.nodes[node]["x"] = torch.cat(
            [original_node_features, structural_features[i]]
        )
