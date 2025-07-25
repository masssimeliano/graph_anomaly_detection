"""
emd_train_2.py
This file contains:
- training method realization for feature model "Attr + Emd2"
- embedding features extractor method.
"""

import logging
from typing import List

import networkx as nx
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.metric import eval_precision_at_k, eval_recall_at_k
from src.helpers.config.const import FEATURE_LABEL_ALPHA2
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.emd_loader import load_emd_model_anomalydae
from src.helpers.time.timed import timed
from src.models.anomalydae.emd_train_1 import get_message_for_write_and_log


@timed
def emd_train(
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

    for epoch in EPOCHS:
        extract_embedding_features(
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
        )

        log_file = (
            RESULTS_DIR_ANOMALYDAE
            / f"{dataset.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.txt"
        )

        loss = 0
        auc = 0
        recall = 0
        precision = 0
        timer = 0
        for i in range(3):
            logging.info(f"Fitting x{i + 1}...")
            model.fit_emd(di_graph)

            loss += model.loss_last / di_graph.num_nodes
            auc += roc_auc_score(labels, model.decision_score_)
            recall += eval_recall_at_k(
                torch.tensor(labels), model.decision_score_, labels.count(1)
            )
            precision += eval_precision_at_k(
                torch.tensor(labels), model.decision_score_, labels.count(1)
            )
            timer += model.last_time

        loss = loss / 3
        auc = auc / 3
        recall = recall / 3
        precision = precision / 3
        timer /= 3

        with open(log_file, "w") as log:
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


def extract_embedding_features(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    epoch: int,
    dataset: str,
):
    logging.info("Loading embedding features to graph nodes...")

    emd_model = load_emd_model_anomalydae(
        dataset=dataset.replace(".mat", ""),
        labels=labels,
        feature_label=FEATURE_LABEL_ALPHA2,
        learning_rate=learning_rate,
        hid_dim=hid_dim,
        epoch=epoch,
    )

    for i, node in enumerate(graph.nodes()):
        original_node_features = graph.nodes[node]["x"]
        embedding_node_features = emd_model[i]
        graph.nodes[node]["x"] = (
            torch.cat([original_node_features, embedding_node_features])
            .detach()
            .clone()
        )
