"""
train_structure_and_feature_2.py
This file contains script to train a feature model "Attr + Str2"
"""

import logging
import random

import numpy as np
import torch

from src.helpers.config.const import FEATURE_LABEL_STR2
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import structure_and_feature_model_2
from src.models.anomalydae.reconstruction_error_model_1 import (
    normalize_node_features_via_minmax_and_remove_nan,
)

FEATURE_TYPE = "Attr + Str2"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"-------------------------------")
        logging.info(f"--- Begin training on {dataset} ({FEATURE_LABEL_STR2}) ---")
        logging.info(f"-------------------------------")

        nx_graph = to_networkx_graph(graph=graph_dict[dataset], do_visualize=False)
        normalize_node_features_via_minmax_and_remove_nan(nx_graph=nx_graph)
        structure_and_feature_model_2.train(
            nx_graph=nx_graph,
            labels=labels_dict[dataset],
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            dataset=dataset,
        )

        logging.info(f"-------------------------------")
        logging.info(f"--- End training on {dataset} ({FEATURE_LABEL_STR2}) ---")
        logging.info(f"-------------------------------\n")


if __name__ == "__main__":
    main()
