import logging
import time
from typing import List

import numpy as np
import torch
import torch_geometric

from pygod.pygod.detector import AnomalyDAE
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *

logging.basicConfig(level=logging.INFO)


def base_train(di_graph: torch_geometric.data.Data,
               labels: List[int],
               title_prefix: str,
               learning_rate: float,
               hid_dim: int,
               data_set: str,
               alpha: float = 0.5):
    measure_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    data_set_name = f"{data_set.replace('.mat', '')}"

    model = AnomalyDAE(epoch=EPOCH_TO_LEARN,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       alpha=alpha,
                       eta=ETA,
                       theta=THETA,
                       gpu=0,
                       labels=labels,
                       title_prefix=title_prefix,
                       data_set=data_set_name)

    array_loss = []
    array_precision_k = []
    array_recall_k = []
    array_auc_roc = []
    array_time = []

    for i in range(3):
        logging.info(f"Fitting x{i + 1}...")
        start_time = time.time()
        # adjusted regular method from AnomalyDAE
        model.fit(di_graph)

        if i == 0:
            array_loss = np.array(model.array_loss)
            array_precision_k = np.array(model.array_precision_k)
            array_recall_k = np.array(model.array_recall_k)
            array_auc_roc = np.array(model.array_auc_roc)
            array_time = np.array(model.array_time)
        else:
            array_loss += np.array(model.array_loss)
            array_precision_k += np.array(model.array_precision_k)
            array_recall_k += np.array(model.array_recall_k)
            array_auc_roc += np.array(model.array_auc_roc)
            array_time += np.array(model.array_time)

    array_loss /= 3
    array_precision_k /= 3
    array_recall_k /= 3
    array_auc_roc /= 3
    array_time /= 3

    for i, current_epoch in enumerate(EPOCHS, start=0):
        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            logging.info(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {array_auc_roc[i]:.4f}")
            write(f"Loss ({title_prefix}): {(array_loss[i] / di_graph.num_nodes):.4f}")
            write(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {array_recall_k[i]:.4f}")
            write(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {array_precision_k[i]:.4f}")
            write(f"Time: {array_time[i]:.4f}")
            logging.info(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {array_auc_roc[i]:.4f}")
            logging.info(f"Loss ({title_prefix}): {(array_loss[i] / di_graph.num_nodes):.4f}")
            logging.info(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {array_recall_k[i]:.4f}")
            logging.info(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {array_precision_k[i]:.4f}")

    logging.info(f"Time: {(time.time() - measure_time):.4f} sec")
