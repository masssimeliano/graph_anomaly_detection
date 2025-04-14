import torch
from torch_geometric.utils import from_networkx
from pygod.detector import AnomalyDAE
from src.helpers.plotters.graph_plotter import to_networkx_graph
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.config import BEST_MODEL_ATTR_STR_PATH
from src.structure.data_set import DataSetSize

def load_emd_model():
    model = AnomalyDAE(lr=0.0001, hid_dim=16, epoch=100)
    model.emb = torch.load(f=BEST_MODEL_ATTR_STR_PATH)

    labels, graph = load_graph_from_mat("book.mat", DataSetSize.SMALL)
    nx_graph = to_networkx_graph(graph, False)
    di_graph = from_networkx(nx_graph)
    model.fit(di_graph)

    return model, labels