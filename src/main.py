import random

import networkx as nx
import numpy as np
import torch

from src.helpers.data_loader import load_graph_from_mat
from src.helpers.graph_plotter import to_networkx_graph
from src.helpers.logs_parser import open_logs, sort_logs
from src.models.unsupervised.anomalydae import (
    baseline_model as baseline,
    structure_and_feature_model as structure_and_feature
)

DATASETS = ["book.mat"]
EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200]
LEARNING_RATE = [0.0005, 0.001, 0.01]
HID_DIM = [16, 32, 64]


def main():
    analyze_logs()
    # train_models()
    # check_trained_model()

def analyze_logs():
    open_logs()
    sort_logs()


def check_trained_model():
    labels, graph = load_graph_from_mat("photo.mat")
    nx_graph = to_networkx_graph(graph)

    baseline.train(nx_graph, labels,
                   learning_rate=5,
                   hid_dim=5,
                   current_epoch=5)


def train_models():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for dataset in DATASETS:
        print("---------------------------------------")
        print("Working with " + dataset)
        print("---------------------------------------\n")

        labels, graph = load_graph_from_mat(dataset)
        nx_graph = to_networkx_graph(graph)

        for rate in LEARNING_RATE:
            print("Learning rate = ", rate)
            for epoch in EPOCHS:
                print("Epoch = ", epoch)
                for dim in HID_DIM:
                    # baseline.train(nx_graph, labels,
                                   # learning_rate=rate,
                                   # hid_dim=dim,
                                   # current_epoch=epoch)
                    structure_and_feature.train(nx_graph, labels,
                                                learning_rate=rate,
                                                hid_dim=dim,
                                                current_epoch=epoch)

if __name__ == "__main__":
    main()
