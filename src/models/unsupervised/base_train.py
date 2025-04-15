import time
from typing import List

import torch
import networkx as nx
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE
from torch_geometric.utils import from_networkx

from src.helpers.config import RESULTS_DIR

def base_train(
    graph: nx.Graph,
    labels: List[int],
    title_prefix: str,
    learning_rate: float,
    hid_dim: int,
    current_epoch: int,
    save_results: bool,
    data_set: str,
    alpha: float = 0.5
):
    di_graph = from_networkx(graph)
    model = AnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, alpha=alpha, save_emb=True)

    if save_results:
        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        emd_file = RESULTS_DIR / f"emd_{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.pt"

        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            start_time = time.time()
            model.fit(di_graph)
            torch.save(model.emb, emd_file)
            auc = roc_auc_score(labels, model.decision_score_)

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Execution time: {(time.time() - start_time):.4f} sec")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
    else:
        print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

        start_time = time.time()
        model.fit(di_graph)
        auc = roc_auc_score(labels, model.decision_score_)

        print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
        print(f"Execution time: {(time.time() - start_time):.4f} sec")