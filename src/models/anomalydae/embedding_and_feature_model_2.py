"""
embedding_and_feature_model_2.py
This file contains train wrapper for the model "Attr + Emd2".
"""

from typing import List

import networkx as nx

from src.helpers.config.const import FEATURE_LABEL_EMD2
from src.models.anomalydae.emd_train_2 import emd_train


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
        title_prefix=FEATURE_LABEL_EMD2,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )
