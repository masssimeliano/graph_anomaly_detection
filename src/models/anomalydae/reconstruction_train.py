"""
reconstruction_train.py
This file contains:
- training method realization for feature model "Attr + Err2"
- reconstruction features extractor method.
"""

import logging
from typing import List

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

from pygod.pygod.detector import AnomalyDAE
from src.helpers.config.const import FEATURE_LABEL_ERROR2
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.time.timed import timed
from src.models.anomalydae.emd_train_1 import get_message_for_write_and_log
from src.models.anomalydae.reconstruction_error_model_1 import (
    normalize_node_features_via_minmax_and_remove_nan,
)


@timed
def reconstruction_train(
    nx_graph: nx.Graph,
    labels: List[int],
    title_prefix: str,
    learning_rate: float,
    hid_dim: int,
    dataset: str,
    alpha: float = ALPHA,
    eta: int = ETA,
    theta: int = THETA,
    gpu: int = 0 if torch.cuda.is_available() else 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    print(f"Device: {device}")

    for epoch in EPOCHS:
        get_reconstruction_errors(
            graph=nx_graph,
            labels=labels,
            learning_rate=learning_rate,
            hid_dim=hid_dim,
            epoch=epoch,
            dataset=dataset,
        )
        di_graph = from_networkx(nx_graph)

        dataset_name = f"{dataset.replace('.mat', '')}"

        model = AnomalyDAE(
            epoch=epoch,
            labels=labels,
            title_prefix=title_prefix,
            data_set=dataset_name,
            lr=learning_rate,
            hid_dim=hid_dim,
            alpha=alpha,
            eta=eta,
            theta=theta,
            gpu=gpu,
            save_emb=True,
        )

        log_file = (
            RESULTS_DIR_ANOMALYDAE
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


def get_reconstruction_errors(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    epoch: int,
    dataset: str,
    hid_dim: int = HIDDEN_DIMS,
    alpha: float = ALPHA,
    eta: int = ETA,
    theta: int = THETA,
    gpu: int = 0 if torch.cuda.is_available() else 1,
):
    logging.info("Calculating errors for graph nodes...")

    di_graph = from_networkx(graph)

    model = AnomalyDAE(
        epoch=epoch,
        lr=learning_rate,
        hid_dim=hid_dim,
        alpha=alpha,
        eta=eta,
        theta=theta,
        gpu=gpu,
        labels=labels,
        title_prefix=FEATURE_LABEL_ERROR2,
        data_set=dataset,
    )

    logging.info(f"Training-Fitting...")
    model.fit_emd(di_graph)
    structural_error_mean = model.stru_error_mean
    structural_error_std = model.stru_error_std
    attribute_error_mean = model.attr_error_mean
    attribute_error_std = model.attr_error_std

    for i, node in enumerate(graph.nodes()):
        original_node_features = graph.nodes[node]["x"]
        node_error_features = torch.tensor(
            [
                structural_error_mean[i].item(),
                structural_error_std[i].item(),
                attribute_error_mean[i].item(),
                attribute_error_std[i].item(),
            ],
            dtype=torch.float32,
        )
        graph.nodes[node]["x"] = (
            torch.cat([original_node_features, node_error_features]).detach().clone()
        )

    normalize_node_features_via_minmax_and_remove_nan(graph)

    return graph
