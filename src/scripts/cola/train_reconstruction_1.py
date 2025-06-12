"""
train_reconstruction_1.py
This file contains script to train a feature model "Attr + Err1"
"""

import logging
import random

import numpy as np
import torch

from src.helpers.config.const import FEATURE_LABEL_ERROR1
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.models.cola import reconstruction_error_model_1


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"-------------------------------")
        logging.info(f"--- Begin training on {dataset} ({FEATURE_LABEL_ERROR1}) ---")
        logging.info(f"-------------------------------")

        reconstruction_error_model_1.train(
            nx_graph=graph_dict[dataset],
            labels=labels_dict[dataset],
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            dataset=dataset,
        )

        logging.info(f"-------------------------------")
        logging.info(f"--- End training on {dataset} ({FEATURE_LABEL_ERROR1}) ---")
        logging.info(f"-------------------------------\n")


if __name__ == "__main__":
    main()
