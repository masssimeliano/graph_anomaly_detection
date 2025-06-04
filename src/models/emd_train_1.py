import logging
import time
from typing import List

import networkx as nx
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.detector.base import precision_at_k, recall_at_k
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.emd_loader import load_emd_model

logging.basicConfig(level=logging.INFO)


def emd_train(nx_graph: nx.Graph,
              labels: List[int],
              title_prefix: str,
              learning_rate: float,
              hid_dim: int,
              dataset: str,
              alpha: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    measure_time = time.time()
    for current_epoch in EPOCHS:
        start_time = time.time()

        extract_embedding_features(nx_graph,
                                   labels,
                                   learning_rate,
                                   hid_dim,
                                   current_epoch,
                                   dataset)
        di_graph = from_networkx(nx_graph)

        dataset_name = f"{dataset.replace('.mat', '')}"

        model = AnomalyDAE(epoch=current_epoch,
                           labels=labels,
                           title_prefix=title_prefix,
                           data_set=dataset_name,
                           eta=ETA,
                           theta=THETA,
                           lr=learning_rate,
                           hid_dim=hid_dim,
                           alpha=alpha,
                           gpu=0)

        log_file = RESULTS_DIR / f"{dataset.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
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
            timer /= 3

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            logging.info(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Loss ({title_prefix}): {loss:.4f}")
            write(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            write(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            write(f"Time: {timer:.4f}")
            logging.info(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            logging.info(f"Loss ({title_prefix}): {loss:.4f}")
            logging.info(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            logging.info(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            logging.info(f"Execution time: {(time.time() - start_time):.4f} sec")

    logging.info(f"Time: {(time.time() - measure_time):.4f} sec")


def extract_embedding_features(graph: nx.Graph,
                               labels: List[int],
                               learning_rate: float,
                               hid_dim: int,
                               epoch: int,
                               data_set: str):
    logging.info("Loading embedding features to graph nodes...")

    emd_model = load_emd_model(data_set=data_set.replace(".mat", ""),
                               labels=labels,
                               feature="Attr + Alpha1",
                               lr=learning_rate,
                               hid_dim=hid_dim,
                               epoch=epoch)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        embedding = emd_model[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, embedding]).detach().clone()
