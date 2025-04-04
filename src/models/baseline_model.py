import torch
from networkx.classes import Graph
from pygod.detector import DOMINANT
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx

def train(graph: Graph, labels):
    di_graph = from_networkx(graph)

    model = DOMINANT()
    model.fit(di_graph)

    scores = model.decision_score_

    # print(scores)

    auc = roc_auc_score(labels, scores)

    print(f"AUC-ROC (Baseline): {auc:.4f}")