"""
emd_loader.py
This file returns loaded model from specific embedding.
"""

from typing import List

import torch

from pygod.pygod.detector import AnomalyDAE
from src.helpers.config.training_config import *
from src.helpers.loaders.emd_file_getter import get_emd_file


def load_emd_model(
    dataset: str,
    feature_label: str,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    epoch: int,
    alpha: float = ALPHA,
    eta: int = ETA,
    theta: float = THETA,
):
    model = AnomalyDAE(
        labels=labels,
        title_prefix=feature_label,
        data_set=dataset,
        lr=learning_rate,
        hid_dim=hid_dim,
        alpha=alpha,
        eta=eta,
        theta=theta,
        epoch=epoch,
    )
    emd_file = get_emd_file(
        dataset=dataset,
        title_prefix=feature_label,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        epoch=epoch,
    )
    model.emb = torch.load(f=emd_file)

    return model.emb
