import torch
from torch_geometric.utils import from_networkx
from pygod.detector import AnomalyDAE
from src.helpers.config import RESULTS_DIR

def load_emd_model(data_set: str, feature: str, lr: float, hid_dim: int, epoch: int):
    model = AnomalyDAE(lr=lr, hid_dim=hid_dim, epoch=epoch)
    emd_file = RESULTS_DIR / (
        f"emd_{data_set.replace(".mat", "")}_{str(feature)}_{str(lr).replace('.', '')}_{hid_dim}_{epoch}.pt"
    )
    model.emb = torch.load(f=emd_file)

    return model.emb