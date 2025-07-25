from typing import List

import networkx as nx

from src.helpers.config.const import FEATURE_LABEL_EMD1
from src.models.ocgnn.emd_train_1 import emd_train


def train(
    nx_graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    dataset: str,
):
    emd_train(
        nx_graph=nx_graph,
        labels=labels,
        title_prefix=FEATURE_LABEL_EMD1,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )
