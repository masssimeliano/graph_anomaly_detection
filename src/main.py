import random
import numpy as np
import torch

from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.graph_plotter import to_networkx_graph
from src.helpers.logs.log_parser import LogParser
from src.models.unsupervised.anomalydae import (
    baseline_model as baseline,
    structure_and_feature_model as structure_and_feature
)
from src.structure.data_set import DataSetSize

DATASETS = ["book.mat"]
EPOCHS = [25, 50, 75, 100, 125, 150]
LEARNING_RATE = [0.0005, 0.001, 0.01]
HID_DIM = [16, 32, 64]

def main():
    analyze_logs()

def analyze_logs():
    parser = LogParser()
    parser.parse_logs()

    best_model, worst_model = parser.get_best_and_worst()
    print("Best model:", best_model)
    print("Worst model:", worst_model)

def train_models():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for dataset in DATASETS:
        print("---------------------------------------")
        print("Working with " + dataset)
        print("---------------------------------------\n")

        labels, graph = load_graph_from_mat(name="book.mat", size=DataSetSize.SMALL)
        nx_graph = to_networkx_graph(graph=graph, visualize=False)

        for rate in LEARNING_RATE:
            print("Learning rate = ", rate)
            for epoch in EPOCHS:
                print("Epoch = ", epoch)
                for dim in HID_DIM:
                    # baseline.train(...)
                    structure_and_feature.train(nx_graph, labels,
                                                learning_rate=rate,
                                                hid_dim=dim,
                                                current_epoch=epoch)

if __name__ == "__main__":
    main()