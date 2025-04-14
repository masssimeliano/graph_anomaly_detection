from typing import List
import networkx as nx
import torch
from src.helpers.loaders.emd_loader import load_emd_model
from src.models.unsupervised.base_train import base_train

def train(graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          current_epoch: int,
          save_results: bool,
          data_set: str,):
    extract_embedding_features(graph, learning_rate, hid_dim, current_epoch, data_set)
    base_train(graph, labels, "Emd + Feature", learning_rate, hid_dim, current_epoch, save_results, data_set)

def extract_embedding_features(graph: nx.Graph,
                              learning_rate: float,
                              hid_dim: int,
                              current_epoch: int,
                              data_set: str):
    print("Adding features")
    node_list = list(graph.nodes())

    for i, node in enumerate(node_list):
        features = graph.nodes[node]['x']
        emd_model = load_emd_model(data_set=data_set, feature="Attr", lr=learning_rate, hid_dim=hid_dim, epoch=current_epoch)
        embedding = emd_model[i]
        graph.nodes[node]['x'] = torch.cat([features, embedding]).detach().clone()