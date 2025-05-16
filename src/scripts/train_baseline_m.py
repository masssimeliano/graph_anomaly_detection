import random

import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config import CURRENT_DATASETS, LEARNING_RATE, HIDDEN_DIMS, SEED, MEDIUM_DATASETS
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import baseline_model
from src.structure.data_set import DataSetSize

FEATURE_TYPE = "Attr"

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
        di_graph = from_networkx(nx_graph)
        baseline_model.train(
            di_graph,
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
