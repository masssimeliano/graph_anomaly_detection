from pathlib import Path

import torch
from pygod.detector import AnomalyDAE
from torch_geometric.utils import from_networkx

from src.helpers.data_loader import load_graph_from_mat
from src.helpers.graph_plotter import to_networkx_graph


log_dir =  Path(__file__).resolve().parents[2] / "results" / "unsupervised" / "anomalyedae" / "best" / "emd_0001_16_100.pt"

def load_emd_model():
    model = AnomalyDAE(lr=0.0001, hid_dim=16, epoch=100)

    model.emb = torch.load(log_dir)

    labels, graph = load_graph_from_mat("book.mat")

    nx_graph = to_networkx_graph(graph)
    di_graph = from_networkx(nx_graph)

    model.fit(di_graph)

    return model, labels
