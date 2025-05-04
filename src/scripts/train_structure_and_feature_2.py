import random

import numpy as np
import torch

from src.helpers.config import CURRENT_DATASETS, MEDIUM_DATASETS, SMALL_DATASETS, LEARNING_RATE, HIDDEN_DIMS, SEED
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.unsupervised.anomalydae import structure_and_feature_model_2
from src.structure.data_set import DataSetSize

FEATURE_TYPE = "Attr + Str2"

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for dataset in MEDIUM_DATASETS:
        print(f"-------------------------------")
        print(f"--- Begin training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------")

        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.MEDIUM)
        nx_graph = to_networkx_graph(graph=graph, visualize=False)
        structure_and_feature_model_2.train(
            nx_graph,
            labels,
            learning_rate=LEARNING_RATE,
            hid_dim=HIDDEN_DIMS,
            save_emb=False,
            data_set=dataset)

        print(f"-------------------------------")
        print(f"--- End training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------\n")

if __name__ == "__main__":
    main()
