from typing import Literal

import networkx as nx
import pandas as pd
import torch

from src.helpers.config import CURRENT_DATASETS, MEDIUM_DATASETS
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.structure.data_set import DataSetSize


def main():
    for dataset in MEDIUM_DATASETS:
        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.MEDIUM)
        nx_graph = to_networkx_graph(graph=graph, visualize=False, title=dataset)
        extract_node_features_tensor(nx_graph=nx_graph)
        node_types = {
            node.id: (
                "attr_anomaly" if node.is_attr_anomaly else
                "struct_anomaly" if node.is_str_anomaly else
                "normal")
            for node in graph.nodes
        }
        summarize_structural_features(nx_graph, node_types, dataset)

def extract_node_features_tensor(nx_graph: nx.Graph) -> tuple[torch.Tensor, list[str], list[int]]:
    nodes = list(nx_graph.nodes())
    node_index = {node_id: idx for idx, node_id in enumerate(nodes)}
    num_nodes = len(nodes)

    features_dicts = {
        "degree": dict(nx_graph.degree()),
        "clustering": nx.clustering(nx_graph),
        "avg_neighbor_degree": nx.average_neighbor_degree(nx_graph),
        "triangles": nx.triangles(nx_graph),
        "harmonic_centrality": nx.harmonic_centrality(nx_graph),
    }

    feature_names = list(features_dicts.keys())
    features_tensor = torch.zeros((num_nodes, len(feature_names)), dtype=torch.float32)

    for i, name in enumerate(feature_names):
        values = features_dicts[name]
        for node_id, val in values.items():
            idx = node_index[node_id]
            features_tensor[idx, i] = val

    return features_tensor

def summarize_structural_features(
    nx_graph: nx.Graph,
    node_types: dict[int, Literal["normal", "attr_anomaly", "struct_anomaly"]],
    dataset: str
):
    tensor, feature_names, node_ids = extract_node_features_tensor(nx_graph)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    groups = {"normal": [], "attr_anomaly": [], "struct_anomaly": []}
    for nid in node_ids:
        t = node_types.get(nid, "normal")
        groups[t].append(node_id_to_idx[nid])

    stats = {}
    for group_name, indices in groups.items():
        if not indices:
            continue
        group_tensor = tensor[indices]
        stats[group_name] = group_tensor.mean(dim=0).tolist()

    print(f"\nMean of structural features ({dataset}):")
    df = pd.DataFrame(stats, index=feature_names).round(2)
    print(df)

    return df

if __name__ == "__main__":
    main()