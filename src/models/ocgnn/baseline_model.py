"""
baseline_model.py
This file contains train wrapper for the model "Attr".
"""

from typing import List

import torch_geometric

from src.helpers.config.const import FEATURE_LABEL_STANDARD
from src.models.ocgnn.base_train import base_train


def train(
    di_graph: torch_geometric.data.Data,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    dataset: str,
):
    base_train(
        di_graph=di_graph,
        labels=labels,
        title_prefix=FEATURE_LABEL_STANDARD,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
    )
