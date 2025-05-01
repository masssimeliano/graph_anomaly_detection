import torch
import networkx as nx
import pyfglt.fglt as fg
from typing import List

from torch_geometric.utils import from_networkx

from src.models.unsupervised.base_train import base_train

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
        title_prefix="Attr + Str2",
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        save_emb=save_emb,
        data_set=data_set
    )

def add_structure_features(graph: nx.Graph):
    print("Adding structural and topological features to graph nodes...")

    clustering = nx.clustering(graph)
    triangles = nx.triangles(graph)
    core_number = nx.core_number(graph)

    ego_densities = {}
    for node in graph.nodes():
        ego = nx.ego_graph(graph, node, radius=1)
        ego_densities[node] = nx.density(ego)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']

        extra_feat = torch.tensor([
            clustering[node],
            triangles[node],
            ego_densities[node],
            core_number[node]
        ], dtype=torch.float)

        combined = torch.cat([original_feat, extra_feat])
        graph.nodes[node]['x'] = combined