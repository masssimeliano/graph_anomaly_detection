from typing import List

import torch

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.detector.base import get_emd_file
from src.helpers.config import RESULTS_DIR


def load_emd_model(data_set: str,
                   feature: str,
                   labels: List[int],
                   lr: float,
                   hid_dim: int,
                   epoch: int):
    model = AnomalyDAE(labels=labels,
                       title_prefix=feature,
                       data_set=data_set,
                       lr=lr,
                       hid_dim=hid_dim,
                       epoch=epoch)
    emd_file = get_emd_file(data_set,
                            feature,
                            lr,
                            hid_dim,
                            epoch)
    model.emb = torch.load(f=emd_file)
    return model.emb
