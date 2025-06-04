import logging
import random

import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.helpers.config.const import FEATURE_LABEL_ALPHA1
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.anomalydae import baseline_alpha_model_1
from src.models.anomalydae.reconstruction_error_model_1 import normalize_node_features_via_minmax_and_remove_nan

logging.basicConfig(level=logging.INFO)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"-------------------------------")
        logging.info(f"--- Begin training on {dataset} ({FEATURE_LABEL_ALPHA1}) ---")
        logging.info(f"-------------------------------")

        nx_graph = to_networkx_graph(graph=graph_dict[dataset],
                                     visualize=False)
        normalize_node_features_via_minmax_and_remove_nan(nx_graph=nx_graph)
        di_graph = from_networkx(G=nx_graph)
        baseline_alpha_model_1.train(di_graph=di_graph,
                                     labels=labels_dict[dataset],
                                     learning_rate=LEARNING_RATE,
                                     hid_dim=HIDDEN_DIMS,
                                     dataset=dataset)

        logging.info(f"-------------------------------")
        logging.info(f"--- End training on {dataset} ({FEATURE_LABEL_ALPHA1}) ---")
        logging.info(f"-------------------------------\n")


if __name__ == "__main__":
    main()
