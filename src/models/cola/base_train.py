"""
base_train.py
This file contains training method realization for feature model "Attr".
"""

import logging
from typing import List

import numpy as np
import torch
import torch_geometric

from pygod.pygod.detector import CoLA
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.time.timed import timed
from src.models.cola.emd_train_1 import get_message_for_write_and_log


@timed
def base_train(
        di_graph: torch_geometric.data.Data,
        labels: List[int],
        title_prefix: str,
        dataset: str,
        learning_rate: float = LEARNING_RATE,
        hid_dim: int = HIDDEN_DIMS,
        gpu: int = 0 if torch.cuda.is_available() else 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    data_set_name = f"{dataset.replace('.mat', '')}"

    model = CoLA(
        epoch=EPOCH_TO_LEARN,
        lr=learning_rate,
        hid_dim=hid_dim,
        gpu=gpu,
        labels=labels,
        title_prefix=title_prefix,
        data_set=data_set_name,
    )

    array_loss = []
    array_precision_k = []
    array_recall_k = []
    array_auc_roc = []
    array_time = []

    for i in range(3):
        logging.info(f"Fitting x{i + 1}...")
        # adjusted regular method from CoLA
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
        log_file = (
                RESULTS_DIR
                / f"{dataset.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        )

        with open(log_file, "w") as log:
            message = get_message_for_write_and_log(
                epoch=current_epoch,
                learning_rate=learning_rate,
                hid_dim=hid_dim,
                title_prefix=title_prefix,
                loss=array_loss[i],
                auc_roc=array_auc_roc[i],
                recall_at_k=array_recall_k[i],
                precision_at_k=array_precision_k[i],
                k=labels.count(1),
                time=array_time[i],
            )

            log.write(message)
            logging.info(message)
