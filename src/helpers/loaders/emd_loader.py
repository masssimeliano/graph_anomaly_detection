from typing import List

import torch

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.detector.base import get_emd_file
from src.helpers.config.training_config import *


def load_emd_model(dataset: str,
                   feature_label: str,
                   labels: List[int],
                   learning_rate: float,
                   hid_dim: int,
                   epoch: int):
    model = AnomalyDAE(labels=labels,
                       title_prefix=feature_label,
                       data_set=dataset,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       eta=ETA,
                       theta=THETA,
                       epoch=epoch)
    emd_file = get_emd_file(dataset=dataset,
                            title_prefix=feature_label,
                            learning_rate=learning_rate,
                            hid_dim=hid_dim,
                            epoch=epoch)
    model.emb = torch.load(f=emd_file)
    return model.emb
