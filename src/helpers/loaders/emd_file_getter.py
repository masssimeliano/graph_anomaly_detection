from src.helpers.config.dir_config import RESULTS_DIR


def get_emd_file(dataset: str,
                 title_prefix: str,
                 learning_rate: float,
                 hid_dim: int,
                 epoch: int):
    return RESULTS_DIR / f"emd_{dataset}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.pt"
