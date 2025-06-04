import logging
import time
from typing import List

import networkx as nx
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.detector.base import precision_at_k, recall_at_k
from src.helpers.config.const import FEATURE_LABEL_ERROR2
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *

logging.basicConfig(level=logging.INFO)


def reconstruction_train(nx_graph: nx.Graph,
                         labels: List[int],
                         title_prefix: str,
                         learning_rate: float,
                         hid_dim: int,
                         dataset: str,
                         alpha: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    measure_time = time.time()
    for epoch in EPOCHS:
        get_reconstruction_errors(graph=nx_graph,
                                  labels=labels,
                                  learning_rate=learning_rate,
                                  hid_dim=hid_dim,
                                  epoch=epoch,
                                  dataset=dataset)
        di_graph = from_networkx(nx_graph)

        dataset_name = f"{dataset.replace('.mat', '')}"

        model = AnomalyDAE(epoch=epoch,
                           labels=labels,
                           title_prefix=title_prefix,
                           data_set=dataset_name,
                           lr=LEARNING_RATE,
                           hid_dim=HIDDEN_DIMS,
                           alpha=alpha,
                           eta=ETA,
                           theta=THETA,
                           gpu=0)

        log_file = RESULTS_DIR / f"{dataset.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            loss = 0
            auc = 0
            recall = 0
            precision = 0
            timer = 0
            for i in range(3):
                logging.info(f"Fitting x{i + 1}...")
                start_time = time.time()
                # adjusted regular method from AnomalyDAE
                model.fit(di_graph)

                loss += model.loss_last / di_graph.num_nodes
                auc += roc_auc_score(labels, model.decision_score_)
                recall += recall_at_k(labels, model.decision_score_, labels.count(1))
                precision += precision_at_k(labels, model.decision_score_, labels.count(1))
                precision += precision_at_k(labels, model.decision_score_, labels.count(1))
                timer += model.last_time

            loss = loss / 3
            auc = auc / 3
            recall = recall / 3
            precision = precision / 3
            timer = timer / 3

            write(f"AnomalyDAE(epoch={epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            logging.info(f"AnomalyDAE(epoch={epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Loss ({title_prefix}): {loss:.4f}")
            write(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            write(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            write(f"Time: {timer:.4f}")
            logging.info(f"Epoch: {epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            logging.info(f"Loss ({title_prefix}): {loss:.4f}")
            logging.info(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            logging.info(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            logging.info(f"Execution time: {(time.time() - start_time):.4f} sec")

    logging.info(f"Time: {(time.time() - measure_time):.4f} sec")


def get_reconstruction_errors(graph: nx.Graph,
                              labels: List[int],
                              learning_rate: float,
                              hid_dim: int,
                              epoch: int,
                              dataset: str):
    logging.info("Calculating errors for graph nodes...")

    di_graph = from_networkx(graph)

    model = AnomalyDAE(epoch=epoch,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       alpha=0.5,
                       gpu=0,
                       labels=labels,
                       title_prefix=FEATURE_LABEL_ERROR2,
                       dataset=dataset)

    logging.info(f"Training-Fitting...")
    model.fit(di_graph)
    stru_error_mean = model.stru_error_mean
    stru_error_std = model.stru_error_std
    attr_error_mean = model.attr_error_mean
    attr_error_std = model.attr_error_std

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        error_feats = torch.tensor([
            stru_error_mean[i].item(),
            stru_error_std[i].item(),
            attr_error_mean[i].item(),
            attr_error_std[i].item()
        ], dtype=torch.float32)
        graph.nodes[node]['x'] = torch.cat([original_feat, error_feats]).detach().clone()
