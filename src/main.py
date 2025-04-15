import os
import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import from_networkx

from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.graph_plotter import to_networkx_graph
from src.helpers.logs.log_parser import LogParser
from src.models.unsupervised.anomalydae import (
    baseline_model,
    structure_and_feature_model,
    embedding_and_feature_model,
    baseline_alpha_model
)
from src.models.unsupervised.anomalydae.structure_and_feature_model import extract_structure_features
from src.structure.data_set import DataSetSize

CONFIG = {
    "datasets": ["book.mat", "photo.mat", "BlogCatalog.mat", "cs.mat", "citeseer.mat"],
    "epochs": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
    "learning_rates": [0.001],
    "hidden_dims": [16],
}

FEATURE_TYPES = ["Attr", "Attr + Str", "Emd + Feature", "Attr + Alpha"]
FEATURE_COLORS = {
    "Attr": "blue",
    "Attr + Str": "green",
    "Attr + Alpha": "yellow",
    "Emd + Feature": "red"
}

def main():
    train_models()

def plot_auc_all_models():
    parser = LogParser()
    parser.parse_logs()

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        has_plot = False

        for feature in FEATURE_TYPES:
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
            plt.plot(epochs, aucs, marker='o', label=feature, color=FEATURE_COLORS[feature])
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

    for epoch in CONFIG["epochs"]:
        best_model = None
        best_auc = -1
        best_dataset = None

        for feature in FEATURE_TYPES:
            for dataset in CONFIG["datasets"]:
                result = parser.get_result_by_params(
                    dataset=dataset.replace(".mat", ""),
                    features=feature,
                    lr=CONFIG["learning_rates"][0],
                    hid_dim=CONFIG["hidden_dims"][0],
                    epoch=epoch
                )

                if result and result["auc_roc"] > best_auc:
                    best_auc = result["auc_roc"]
                    best_model = result
                    best_dataset = dataset

        if best_model:
            print(f"Epoch: {epoch} → Best dataset: {best_dataset} | AUC: {best_auc:.4f} | name: {best_model['filename']}")

def train_models():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for dataset in CONFIG["datasets"]:
        print(f"--- Training on {dataset} ---\n")

        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.SMALL)
        nx_graph = to_networkx_graph(graph=graph, visualize=False)
        di_graph = from_networkx(nx_graph)
        for rate in CONFIG["learning_rates"]:
            for dim in CONFIG["hidden_dims"]:
                baseline_alpha_model.train(
                    di_graph,
                    labels,
                    learning_rate=rate,
                    hid_dim=dim,
                    save_results=True,
                    data_set=dataset
                )
                baseline_model.train(
                    di_graph,
                    labels,
                    learning_rate=rate,
                    hid_dim=dim,
                    save_results=True,
                    data_set=dataset
                )
                structure_and_feature_model.train(
                    nx_graph,
                    labels,
                    learning_rate=rate,
                    hid_dim=dim,
                    save_results=True,
                    data_set=dataset
                )
                print()

if __name__ == "__main__":
    main()
