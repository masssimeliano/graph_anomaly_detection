import time
from pathlib import Path
from typing import List

import networkx as nx
import torch
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE
from torch_geometric.utils import from_networkx


def base_train(graph: nx.Graph, labels: List[int],
               title_prefix: str = "",
               learning_rate: float = 0.005, hid_dim: int = 16,
               current_epoch: int = 100):
    log_file = (
        Path(__file__).resolve().parents[3] / "results" / "unsupervised" / "anomalyedae" /
            (
                    str(learning_rate).replace(".", "") +
                    "_" + str(hid_dim) +
                    "_" + str(current_epoch) +
                    ".txt"
            )
        )

    emd_file = (
            Path(__file__).resolve().parents[3] / "results" / "unsupervised" / "anomalyedae" /
            (
                    "emd_" + str(learning_rate).replace(".", "") +
                    "_" + str(hid_dim) +
                    "_" + str(current_epoch)
            )
    )

    with open(log_file, "w") as log:
        def write(msg):
            log.write(msg + "\n")

        print("Training")
        start_time = time.time()

        di_graph = from_networkx(graph)

        model = AnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, save_emb=True)
        write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
        model.fit(di_graph)

        torch.save(model.emb, emd_file)

        scores = model.decision_score_
        auc = roc_auc_score(labels, scores)

        write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
        write(f"Execution time: {(time.time() - start_time):.4f} sec\n")

