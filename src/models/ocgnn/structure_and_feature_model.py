import logging
from typing import List

import networkx as nx
import pyfglt.fglt as fg
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_STR
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
        title_prefix=FEATURE_LABEL_STR,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )


@timed
def add_structure_features(nx_graph: nx.Graph):
    logging.info("Adding structural graphlet features to graph nodes...")

    F = fg.compute(A=nx_graph).reindex(list(nx_graph.nodes()))

    for i, node in enumerate(nx_graph.nodes()):
        original_node_features = nx_graph.nodes[node]["x"]
        node_graphlet_features = torch.tensor(F.iloc[i].values, dtype=torch.float)
        nx_graph.nodes[node]["x"] = torch.cat(
            [original_node_features, node_graphlet_features]
        )
