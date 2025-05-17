import matplotlib.pyplot as plt
from typing import List, Optional

def plot_auc_curve(
    epochs: List[int],
    aucs: List[float],
    file_name: str,
    title_prefix: str,
    save_fig: bool,
    save_path: Optional[str] = None):

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, aucs, marker='o')
    plt.title(f"AUC-ROC vs Epochs ({title_prefix} AnomalyDAE): {file_name}")
    plt.xlabel("Epochs")
    plt.ylabel("AUC-ROC")
    plt.grid(True)
    plt.tight_layout()

    if save_fig and save_path:
        plt.savefig(save_path)
    plt.show()

def plot_auc_curve(
    epochs: List[int],
    losses: List[float],
    file_name: str,
    title_prefix: str,
    save_fig: bool,
    save_path: Optional[str] = None):

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o')
    plt.title(f"AUC-ROC vs Loss ({title_prefix} AnomalyDAE): {file_name}")
    plt.xlabel("Epochs")
    plt.ylabel("AUC-ROC")
    plt.grid(True)
    plt.tight_layout()

    if save_fig and save_path:
        plt.savefig(save_path)
    plt.show()