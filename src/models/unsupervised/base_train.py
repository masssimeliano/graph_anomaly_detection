import time
from typing import List
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE
from torch_geometric.utils import from_networkx
from src.helpers.config import RESULTS_DIR

def base_train(graph: nx.Graph,
               labels: List[int],
               title_prefix: str,
               learning_rate: float,
               hid_dim: int,
               current_epoch: int,
               save_results: bool):
    if save_results:
        log_file = RESULTS_DIR / (
            f"{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        )
        emd_file = RESULTS_DIR / (
            f"emd_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}"
        )

        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            print(f"Training a model with lr={learning_rate}, hid={hid_dim}, current={current_epoch}")
            start_time = time.time()

            di_graph = from_networkx(graph)

            model = AnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, save_emb=True)
            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            model.fit(di_graph)

            torch.save(model.emb, emd_file)

            scores = model.decision_score_
            auc = roc_auc_score(labels, scores)

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Execution time: {(time.time() - start_time):.4f} sec\n")
    else:
        start_time = time.time()

        di_graph = from_networkx(graph)

        model = AnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, save_emb=True)
        print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
        model.fit(di_graph)

        scores = model.decision_score_
        auc = roc_auc_score(labels, scores)

        print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")