import torch
import networkx as nx
from typing import List

from torch_geometric.utils import from_networkx

from src.helpers.loaders.emd_loader import load_emd_model
from src.models.unsupervised.emd_train import emd_train

def train(
    graph: nx.Graph,
    labels: List[int],
    learning_rate: float,
    hid_dim: int,
    save_emb: bool,
    data_set: str):
    emd_train(graph, labels, "Attr + Emd", learning_rate, hid_dim, save_emb, data_set)