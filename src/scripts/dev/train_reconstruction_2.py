import random

import numpy as np
import torch

from src.helpers.config import SMALL_DATASETS, LEARNING_RATE, HIDDEN_DIMS, SEED, CURRENT_DATASETS, \
    CURRENT_DATASETS_SIZE, graph_dict, labels_dict
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import reconstruction_error_model_2


FEATURE_TYPE = "Attr + Error2"

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(CURRENT_DATASETS, start=0):
        print(f"-------------------------------")
        print(f"--- Begin training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------")

        nx_graph = to_networkx_graph(graph=graph_dict[dataset], visualize=False)
        reconstruction_error_model_2.train(
            nx_graph,
            labels_dict[dataset],
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            data_set=dataset)

        print(f"-------------------------------")
        print(f"--- End training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------\n")

if __name__ == "__main__":
    main()
