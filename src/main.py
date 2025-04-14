import os
import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.graph_plotter import to_networkx_graph
from src.helpers.logs.log_parser import LogParser
from src.models.unsupervised.anomalydae import (
    baseline_model as baseline,
    structure_and_feature_model as structure_and_feature,
    embedding_and_feature_model as embedding_and_feature
)
from src.structure.data_set import DataSetSize

DATASETS = ["book.mat", "citeseer.mat", "photo.mat"]
EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
LEARNING_RATE = [0.001]
HID_DIM = [16]

def main():
    plot_auc_all_models()

def plot_auc_all_models():
    parser = LogParser()
    parser.parse_logs()

    datasets = set(r["dataset"] for r in parser.results)
    feature_types = ["Attr", "Attr + Str", "Emd + Feature"]
    feature_labels = {
        "Attr": "Baseline",
        "Attr + Str": "Attr + Str",
        "Emd + Feature": "Emd + Feature"
    }
    feature_colors = {
        "Attr": "blue",
        "Attr + Str": "green",
        "Emd + Feature": "red"
    }

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        has_plot = False

        for feature in feature_types:
            filtered = [
                r for r in parser.results
                if r["dataset"] == dataset and r["features"] == feature
            ]

            if not filtered:
                continue

            epoch_auc = defaultdict(float)
            for r in filtered:
                epoch = r["epoch"]
                auc = r["auc_roc"]
                if auc > epoch_auc[epoch]:
                    epoch_auc[epoch] = auc

            if not epoch_auc:
                continue

            epochs = sorted(epoch_auc.keys())
            aucs = [epoch_auc[e] for e in epochs]
            label = feature_labels[feature]
            color = feature_colors[feature]
            plt.plot(epochs, aucs, marker='o', label=label, color=color)
            has_plot = True

        if has_plot:
            plt.title(f'AUC-ROC vs Epochs ({dataset})')
            plt.xlabel('Epochs')
            plt.ylabel('AUC-ROC')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

def analyze_logs():
    parser = LogParser()
    parser.parse_logs()

    best_model, worst_model = parser.get_best_and_worst()
    print("Best model:", best_model["auc_roc"], " ", best_model["filename"])
    print("Worst model:", worst_model["auc_roc"], " ", worst_model["filename"])

    for epoch in EPOCHS:
        best_model = None
        best_auc = -1
        best_dataset = None
        for feature in ["Attr", "Attr + Str", "Emd + Feature"]:
            for dataset in DATASETS:
                model = parser.get_result_by_params(
                    dataset=dataset.replace(".mat", ""),
                    features=feature,
                    lr=LEARNING_RATE[0],
                    hid_dim=HID_DIM[0],
                    epoch=epoch)

                if model["auc_roc"] > best_auc:
                    best_auc = model["auc_roc"]
                    best_model = model
                    best_dataset = dataset

        if best_model:
             print(f"Epoch: {epoch} → Best dataset: {best_dataset} | AUC: {best_auc:.4f} | name: {best_model["filename"]}")

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
                    embedding_and_feature.train(nx_graph,
                                   labels,
                                    learning_rate=rate,
                                    hid_dim=dim,
                                    current_epoch=epoch,
                                    save_results=True,
                                    data_set=dataset)
                    print("\n")

if __name__ == "__main__":
    main()