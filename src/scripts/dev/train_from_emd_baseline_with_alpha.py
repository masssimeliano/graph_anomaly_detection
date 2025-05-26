import random

import numpy as np
import torch

from src.helpers.config import CURRENT_DATASETS, LEARNING_RATE, HIDDEN_DIMS, SEED, CURRENT_DATASETS_SIZE, labels_dict, \
    graph_dict
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import embedding_and_feature_model


FEATURE_TYPE = "Attr + Emd"

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(CURRENT_DATASETS, start=0):
        print(f"-------------------------------")
        print(f"--- Begin training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------")

        nx_graph = to_networkx_graph(graph=graph_dict[dataset], visualize=False)
        embedding_and_feature_model.train(
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
