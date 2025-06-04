import logging
import random

import numpy as np
import torch

from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import reconstruction_error_model_1

logging.basicConfig(level=logging.INFO)

FEATURE_TYPE = "Attr + Error1"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(CURRENT_DATASETS, start=0):
        logging.info(f"-------------------------------")
        logging.info(f"--- Begin training on {dataset} ({FEATURE_TYPE}) ---")
        logging.info(f"-------------------------------")

        nx_graph = to_networkx_graph(graph=graph_dict[dataset], visualize=False)
        reconstruction_error_model_1.train(
            nx_graph,
            labels_dict[dataset],
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            data_set=dataset)

        logging.info(f"-------------------------------")
        logging.info(f"--- End training on {dataset} ({FEATURE_TYPE}) ---")
        logging.info(f"-------------------------------\n")


if __name__ == "__main__":
    main()
