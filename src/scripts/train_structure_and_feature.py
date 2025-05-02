import random

import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config import CURRENT_DATASETS
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.unsupervised.anomalydae import structure_and_feature_model
from src.structure.data_set import DataSetSize

CONFIG = {
    "learning_rate": 0.001,
    "hidden_dims": 16,
}

FEATURE_TYPE = "Attr + Str"

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for dataset in CURRENT_DATASETS:
        print(f"-------------------------------")
        print(f"--- Begin training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------")

        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.MEDIUM)
        nx_graph = to_networkx_graph(graph=graph, visualize=False)
        structure_and_feature_model.train(
            nx_graph,
            labels,
            learning_rate=CONFIG["learning_rate"],
            hid_dim=CONFIG["hidden_dims"],
            save_emb=False,
            data_set=dataset
        )

        print(f"-------------------------------")
        print(f"--- End training on {dataset} ({FEATURE_TYPE}) ---")
        print(f"-------------------------------\n")

if __name__ == "__main__":
    main()
