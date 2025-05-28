import time
from typing import List

import torch
import torch_geometric

from pygod.pygod.detector import AnomalyDAE
from src.helpers.config import RESULTS_DIR, EPOCHS


def base_train(di_graph: torch_geometric.data.Data,
               labels: List[int],
               title_prefix: str,
               learning_rate: float,
               hid_dim: int,
               data_set: str,
               alpha: float = 0.5):
    measure_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)

    data_set_name = f"{data_set.replace('.mat', '')}"

    # epoch does not matter here
    # the maximum epochs amount is set to 250 standard and will be retrained each 25 epochs
    model = AnomalyDAE(epoch=100,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       alpha=alpha,
                       gpu=0,
                       labels=labels,
                       title_prefix=title_prefix,
                       data_set=data_set_name)

    model.fit(di_graph)
    array_loss = model.array_loss
    array_precision_k = model.array_precision_k
    array_recall_k = model.array_recall_k
    array_auc_roc = model.array_auc_roc

    for i, current_epoch in enumerate(EPOCHS, start=0):
        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {array_auc_roc[i]:.4f}")
            write(f"Loss ({title_prefix}): {(array_loss[i] / di_graph.num_nodes):.4f}")
            write(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {array_recall_k[i]:.4f}")
            write(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {array_precision_k[i]:.4f}")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {array_auc_roc[i]:.4f}")
            print(f"Loss ({title_prefix}): {(array_loss[i] / di_graph.num_nodes):.4f}")
            print(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {array_recall_k[i]:.4f}")
            print(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {array_precision_k[i]:.4f}")

    print(f"Time: {(time.time() - measure_time):.4f} sec")