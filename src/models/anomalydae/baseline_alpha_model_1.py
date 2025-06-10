"""
baseline_alpha_model_1.py
This file contains train wrapper for the model "Attr + Alpha1".
"""

from typing import List

import torch_geometric

from src.helpers.config.const import FEATURE_LABEL_ALPHA1
from src.models.base_train import base_train


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
        title_prefix=FEATURE_LABEL_ALPHA1,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        dataset=dataset,
        alpha=0,
    )
