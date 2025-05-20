import time
from typing import List

import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.metrics import roc_auc_score
from pygod.detector import AnomalyDAE

from src.helpers.config import RESULTS_DIR, EPOCHS
from src.helpers.loaders.emd_loader import load_emd_model
from src.models.encoder.custom_anomalydae import CustomAnomalyDAE


def emd_train(
    nx_graph: nx.Graph,
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
        start_time = time.time()

        extract_embedding_features(nx_graph, learning_rate, hid_dim, current_epoch, data_set)
        di_graph = from_networkx(nx_graph)

        data_set_name = f"{data_set.replace('.mat', '')}"

        model = CustomAnomalyDAE(epoch=current_epoch,
                                 lr=learning_rate,
                                 hid_dim=hid_dim,
                                 alpha=alpha,
                                 gpu=0,
                                 labels=labels,
                                 title_prefix=title_prefix,
                                 data_set=data_set_name)

        log_file = RESULTS_DIR / f"{data_set.replace('.mat', '')}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{current_epoch}.txt"
        with open(log_file, "w") as log:
            def write(msg):
                log.write(msg + "\n")

            model.fit_emd(di_graph)
            loss = model.loss_last / di_graph.num_nodes
            auc = roc_auc_score(labels, model.decision_score_)

            write(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")
            print(f"AnomalyDAE(epoch={current_epoch}, lr={learning_rate}, hid_dim={hid_dim})")

            write(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            write(f"Loss ({title_prefix}): {loss:.4f}")
            print(f"Epoch: {current_epoch} - AUC-ROC ({title_prefix}): {auc:.4f}")
            print(f"Loss ({title_prefix}): {loss:.4f}")
            print(f"Execution time: {(time.time() - start_time):.4f} sec")

    print(f"Time: {(time.time() - measure_time):.4f} sec")
    
def extract_embedding_features(
    graph: nx.Graph,
    learning_rate: float,
    hid_dim: int,
    epoch: int,
    data_set: str):

    print("Loading embedding features to graph nodes...")

    emd_model = load_emd_model(
        data_set=data_set,
        feature="Attr + Alpha",
        lr=learning_rate,
        hid_dim=hid_dim,
        epoch=epoch)

    for i, node in enumerate(graph.nodes()):
        original_feat = graph.nodes[node]['x']
        embedding = emd_model[i]
        graph.nodes[node]['x'] = torch.cat([original_feat, embedding]).detach().clone()
