import torch
import networkx as nx
from typing import List

from torch_geometric.utils import from_networkx

from src.models.unsupervised.base_train import base_train
from src.scripts.sout_nodes_information import extract_node_features_tensor


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
    print("Adding structural graphlet features to graph nodes...")
    additional_feats = extract_node_features_tensor(graph)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        stat_feat = additional_feats[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, stat_feat])