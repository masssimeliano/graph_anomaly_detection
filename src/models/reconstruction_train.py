import logging
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
from src.helpers.time.timed import timed
from src.models.anomalydae.reconstruction_error_model_1 import normalize_node_features_via_minmax_and_remove_nan

logging.basicConfig(level=logging.INFO)


@timed
def reconstruction_train(nx_graph: nx.Graph,
                         labels: List[int],
                         title_prefix: str,
                         learning_rate: float,
                         hid_dim: int,
                         dataset: str,
                         alpha: float = ALPHA,
                         eta: int = ETA,
                         theta: int = THETA,
                         gpu: int = 0 if torch.cuda.is_available() else 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

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
                           lr=learning_rate,
                           hid_dim=hid_dim,
                           alpha=alpha,
                           eta=eta,
                           theta=theta,
                           gpu=gpu,
                           save_emb=True)

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
                model.fit_emd(di_graph)

                loss += model.loss_last / di_graph.num_nodes
                auc += roc_auc_score(labels, model.decision_score_)
                recall += recall_at_k(labels, model.decision_score_, labels.count(1))
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


def get_reconstruction_errors(graph: nx.Graph,
                              labels: List[int],
                              learning_rate: float,
                              epoch: int,
                              dataset: str,
                              hid_dim: int = HIDDEN_DIMS,
                              alpha: float = ALPHA,
                              eta: int = ETA,
                              theta: int = THETA,
                              gpu: int = 0 if torch.cuda.is_available() else 1):
    logging.info("Calculating errors for graph nodes...")

    di_graph = from_networkx(graph)

    model = AnomalyDAE(epoch=epoch,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       alpha=alpha,
                       eta=eta,
                       theta=theta,
                       gpu=gpu,
                       labels=labels,
                       title_prefix=FEATURE_LABEL_ERROR2,
                       data_set=dataset)

    logging.info(f"Training-Fitting...")
    model.fit_emd(di_graph)
    structural_error_mean = model.stru_error_mean
    structural_error_std = model.stru_error_std
    attribute_error_mean = model.attr_error_mean
    attribute_error_std = model.attr_error_std

    print(len(structural_error_mean))

    for i, node in enumerate(graph.nodes()):
        original_node_features = graph.nodes[node]['x']
        node_error_features = torch.tensor([
            structural_error_mean[i].item(),
            structural_error_std[i].item(),
            attribute_error_mean[i].item(),
            attribute_error_std[i].item()
        ], dtype=torch.float32)
        graph.nodes[node]['x'] = torch.cat([original_node_features, node_error_features]).detach().clone()

    normalize_node_features_via_minmax_and_remove_nan(graph)
