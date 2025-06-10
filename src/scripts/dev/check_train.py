"""
check_train.py
This file contains script to run all given models and then plot results of their learning.
"""

import logging

import read_and_show_metrics
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.scripts.dev import train_reconstruction_2

logging.basicConfig(level=logging.INFO)


def main():
    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset, size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        graph_dict[dataset] = to_networkx_graph(graph=graph, do_visualize=False)

    train_reconstruction_2.main()

    read_and_show_metrics.plot_time()


if __name__ == "__main__":
    main()
