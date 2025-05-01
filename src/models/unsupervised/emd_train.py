import time
from typing import List

import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE

from src.helpers.config import RESULTS_DIR
from src.helpers.loaders.emd_loader import load_emd_model


def emd_train(
    nx_graph: nx.Graph,
    labels: List[int],
    title_prefix: str,
    learning_rate: float,
    hid_dim: int,
    save_emb: bool,
    data_set: str,
    alpha: float = 0.5):
    measure_time = time.time()
    for current_epoch in  [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
        start_time = time.time()

        extract_embedding_features(nx_graph, learning_rate, hid_dim, current_epoch, data_set)
        di_graph = from_networkx(nx_graph)

        model = AnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, alpha=alpha, save_emb=True)

        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            model.fit(di_graph)
            auc = roc_auc_score(labels, model.decision_score_)

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Execution time: {(time.time() - start_time):.4f} sec")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            print(f"Execution time: {(time.time() - start_time):.4f} sec")

        if save_emb:
            emd_file = RESULTS_DIR / f"emd_{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.pt"
            torch.save(model.emb, emd_file)

    print(f"Time: {(time.time() - measure_time):.4f} sec")
    
def extract_embedding_features(
    graph: nx.Graph,
    learning_rate: float,
    hid_dim: int,
    epoch: int,
    data_set: str):
    print("Loading embedding features to graph nodes...")
    emd_model = load_emd_model(
        data_set=data_set,
        feature="Attr + Alpha",
        lr=learning_rate,
        hid_dim=hid_dim,
        epoch=epoch)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        embedding = emd_model[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, embedding]).detach().clone()
