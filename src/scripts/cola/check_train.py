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
from src.models.cola.reconstruction_error_model_1 import (
    normalize_node_features_via_minmax_and_remove_nan,
)
from src.scripts.cola import train_baseline, train_reconstruction_2, train_reconstruction_1, \
    train_structure_and_feature, train_structure_and_feature_3, train_structure_and_feature_2, \
    train_and_save_emd_from_baseline_alpha_2, train_and_save_emd_from_baseline_alpha_1, \
    train_from_emd_baseline_with_alpha_1, train_from_emd_baseline_with_alpha_2
from src.scripts.cola.tg_bot import TelegramLogHandler

logging.basicConfig(level=logging.INFO)


def main():
    # Получаем root-логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Добавляем TelegramHandler к root-логгеру
    telegram_handler = TelegramLogHandler()
    telegram_handler.setLevel(logging.INFO)
    telegram_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(telegram_handler)

    # Также добавим консоль для наглядности
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(console_handler)

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
    train_baseline.main()
    train_and_save_emd_from_baseline_alpha_1.main()
    train_and_save_emd_from_baseline_alpha_2.main()
    train_from_emd_baseline_with_alpha_1.main()
    train_from_emd_baseline_with_alpha_2.main()
    train_reconstruction_1.main()
    train_reconstruction_2.main()
    train_structure_and_feature.main()
    train_structure_and_feature_2.main()
    train_structure_and_feature_3.main()

    read_and_show_metrics.plot_time()


if __name__ == "__main__":
    main()
