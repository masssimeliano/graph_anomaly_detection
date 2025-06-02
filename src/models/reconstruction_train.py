import time
from typing import List

import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.metrics import roc_auc_score

from pygod.pygod.detector import AnomalyDAE
from pygod.pygod.detector.base import precision_at_k, recall_at_k
from src.helpers.config import RESULTS_DIR, EPOCHS, ETA, THETA
from src.helpers.loaders.emd_loader import load_emd_model
from src.models.anomalydae.reconstruction_error_model_1 import normalize_node_features_minmax


def reconstruction_train(nx_graph: nx.Graph,
              labels: List[int],
              title_prefix: str,
              learning_rate: float,
              hid_dim: int,
              data_set: str,
              alpha: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)

    measure_time = time.time()
    for current_epoch in EPOCHS:
        get_reconstruction_errors(nx_graph,
                                   labels,
                                   learning_rate,
                                   hid_dim,
                                   current_epoch,
                                   data_set)
        di_graph = from_networkx(nx_graph)

        data_set_name = f"{data_set.replace('.mat', '')}"

        model = AnomalyDAE(epoch=current_epoch,
                           labels=labels,
                           title_prefix=title_prefix,
                           data_set=data_set_name,
                           lr=learning_rate,
                           hid_dim=hid_dim,
                           alpha=alpha,
                           eta=ETA,
                           theta=THETA,
                           gpu=0)

        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            loss = 0
            auc = 0
            recall = 0
            precision = 0
            timer = 0
            for i in range(3):
                print(f"Fitting x{i + 1}...")
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

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Loss ({title_prefix}): {loss:.4f}")
            write(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            write(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            write(f"Time: {timer:.4f}")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            print(f"Loss ({title_prefix}): {loss:.4f}")
            print(f"Recall@k ({title_prefix}) for k={labels.count(1)}: {recall:.4f}")
            print(f"Precision@k ({title_prefix}) for k={labels.count(1)}: {precision:.4f}")
            print(f"Execution time: {(time.time() - start_time):.4f} sec")

    print(f"Time: {(time.time() - measure_time):.4f} sec")
    
def get_reconstruction_errors(graph: nx.Graph,
                               labels: List[int],
                               learning_rate: float,
                               hid_dim: int,
                               epoch: int,
                               data_set: str):

    print("Calculating errors for graph nodes...")

    normalize_node_features_minmax(graph)
    di_graph = from_networkx(graph)

    model = AnomalyDAE(epoch=epoch,
                       lr=learning_rate,
                       hid_dim=hid_dim,
                       alpha=0.5,
                       gpu=0,
                       labels=labels,
                       title_prefix="Attr + Error2",
                       data_set=data_set)

    print(f"Training-Fitting...")
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
