import time
from typing import List

import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE

from src.helpers.config import RESULTS_DIR, EPOCHS
from src.models.encoder.custom_anomalydae import CustomAnomalyDAE


def base_train(
    di_graph: torch_geometric.data.Data,
    labels: List[int],
    title_prefix: str,
    learning_rate: float,
    hid_dim: int,
    save_emb: bool,
    data_set: str,
    alpha: float = 0.5):
    measure_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)

    for current_epoch in EPOCHS:
        start_time = time.time()

        model = CustomAnomalyDAE(epoch=current_epoch, lr=learning_rate, hid_dim=hid_dim, alpha=alpha, save_emb=True, gpu=0)

        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            model.fit(di_graph)
            auc = roc_auc_score(labels, model.decision_score_)
            loss = model.loss_last / di_graph.num_nodes

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Execution time: {(time.time() - start_time):.4f} sec")
            write(f"Loss ({title_prefix}): {loss:.4f}")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            print(f"Execution time: {(time.time() - start_time):.4f} sec")
            print(f"Loss ({title_prefix}): {loss:.4f}")

        if save_emb:
            emd_file = RESULTS_DIR / f"emd_{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.pt"
            torch.save(model.emb, emd_file)

    print(f"Time: {(time.time() - measure_time):.4f} sec")