import logging

import read_and_show_metrics
import src.scripts.anomalydae.check_train
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.models.ocgnn.reconstruction_error_model_1 import (
    normalize_node_features_via_minmax_and_remove_nan,
)
from src.scripts.ocgnn import (
    train_reconstruction_2,
    train_structure_and_feature,
    train_structure_and_feature_3,
    train_structure_and_feature_2,
)

logging.basicConfig(level=logging.INFO)


def main():
    read_all()
    train_all()

    read_and_show_metrics.plot_time()


def read_all():
    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset, size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        nx_graph = to_networkx_graph(graph=graph, do_visualize=False)
        normalize_node_features_via_minmax_and_remove_nan(
            nx_graph=to_networkx_graph(graph=graph, do_visualize=False)
        )
        graph_dict[dataset] = nx_graph


def train_all():
    train_reconstruction_2.main()
    train_structure_and_feature.main()
    train_structure_and_feature_2.main()
    train_structure_and_feature_3.main()

    read_and_show_metrics.plot_time()


def check_all():
    read_all()
    train_all()
    src.scripts.anomalydae.check_train.train_all()
    src.scripts.cola.check_train.train_all()


if __name__ == "__main__":
    check_all()
