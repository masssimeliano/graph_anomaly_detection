import logging
import random

import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_STANDARD
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.models.ocgnn import baseline_model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"-------------------------------")
        logging.info(f"--- Begin training on {dataset} ({FEATURE_LABEL_STANDARD}) ---")
        logging.info(f"-------------------------------")

        di_graph = from_networkx(G=graph_dict[dataset])
        baseline_model.train(
            di_graph=di_graph,
            labels=labels_dict[dataset],
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            dataset=dataset,
        )

        logging.info(f"-------------------------------")
        logging.info(f"--- End training on {dataset} ({FEATURE_LABEL_STANDARD}) ---")
        logging.info(f"-------------------------------\n")


if __name__ == "__main__":
    main()
