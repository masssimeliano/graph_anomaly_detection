import time
from typing import List

import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE
from torch_geometric.utils import from_networkx


def train(graph: nx.Graph, labels: List[int], plot_needed: bool, file_name: str):
    print("Begin model training")

    start_time = time.time()

    epochs = list(range(1, 252, 10))
    aucs = []

    di_graph = from_networkx(graph)

    for e in epochs:
        model = AnomalyDAE(epoch=e, lr=0.01, hid_dim=16)
        model.fit(di_graph)
        scores = model.decision_score_
        auc = roc_auc_score(labels, scores)
        aucs.append(auc)

        if e % 10 == 1:
            print(f"Epoch: {e} - AUC-ROC (Attr): {auc:.4f}")

    print(f"Execution time: {time.time() - start_time:.2f} sec\n")
    if plot_needed:
        plot_auc_curve(epochs, aucs, file_name)


def plot_auc_curve(epochs: List[int], aucs: List[float], file_name: str):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, aucs, marker='o')
    plt.title('AUC-ROC vs Epochs (Attr AnomalyDAE): ' + file_name)
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
