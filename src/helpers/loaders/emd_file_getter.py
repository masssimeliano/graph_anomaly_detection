"""
emd_file_getter.py
This file creates get-method to specific embedding file for each model.
"""

from src.helpers.config.dir_config import (
    RESULTS_DIR_ANOMALYDAE,
    RESULTS_DIR_COLA,
    RESULTS_DIR_OCGNN,
)


def get_emd_file_anomalydae(
    dataset: str, title_prefix: str, learning_rate: float, hid_dim: int, epoch: int
):
    return (
        RESULTS_DIR_ANOMALYDAE
        / f"emd_{dataset}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.pt"
    )


def get_emd_file_cola(
    dataset: str, title_prefix: str, learning_rate: float, hid_dim: int, epoch: int
):
    return (
        RESULTS_DIR_COLA
        / f"emd_{dataset}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.pt"
    )


def get_emd_file_ocgnn(
    dataset: str, title_prefix: str, learning_rate: float, hid_dim: int, epoch: int
):
    return (
        RESULTS_DIR_OCGNN
        / f"emd_{dataset}_{title_prefix}_{str(learning_rate).replace('.', '')}_{hid_dim}_{epoch}.pt"
    )
