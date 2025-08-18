import logging
from typing import List

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

from pygod.pygod.detector import CoLA
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.time.timed import timed
from src.models.anomalydae.reconstruction_train import (
    get_reconstruction_errors as get_anomalydae_reconstruction_errors,
)
from src.models.cola.emd_train_1 import get_message_for_write_and_log


@timed
def reconstruction_train(
    nx_graph: nx.Graph,
    labels: List[int],
    title_prefix: str,
    learning_rate: float,
    hid_dim: int,
    dataset: str,
    gpu: int = 0 if torch.cuda.is_available() else 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    for epoch in EPOCHS:
        get_anomalydae_reconstruction_errors(
            graph=nx_graph,
            labels=labels,
            learning_rate=learning_rate,
            hid_dim=hid_dim,
            epoch=epoch,
            dataset=dataset,
        )
        di_graph = from_networkx(nx_graph)

        dataset_name = f"{dataset.replace('.mat', '')}"

        model = CoLA(
            epoch=epoch,
            labels=labels,
            title_prefix=title_prefix,
            data_set=dataset_name,
            lr=learning_rate,
            hid_dim=hid_dim,
            gpu=gpu,
            save_emb=True,
        )

        log_file = (
            RESULTS_DIR_COLA
            / f"{dataset.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.txt"
        )
        with open(log_file, "w") as log:
            loss = 0
            auc = 0
            recall = 0
            precision = 0
            timer = 0
            for i in range(3):
                logging.info(f"Fitting x{i + 1}...")

                (auc_i, recall_i, precision_i) = model.fit_emd(di_graph)

                loss += model.loss_last / di_graph.num_nodes
                auc += auc_i
                recall += recall_i
                precision += precision_i
                timer += model.last_time

            loss = loss / 3
            auc = auc / 3
            recall = recall / 3
            precision = precision / 3
            timer = timer / 3

            message = get_message_for_write_and_log(
                epoch=epoch,
                learning_rate=learning_rate,
                hid_dim=hid_dim,
                title_prefix=title_prefix,
                loss=loss,
                auc_roc=auc,
                recall_at_k=recall,
                precision_at_k=precision,
                k=labels.count(1),
                time=timer,
            )

            log.write(message)
            logging.info(message)
