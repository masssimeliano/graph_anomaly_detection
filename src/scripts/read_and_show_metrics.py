import os
from collections import defaultdict

from matplotlib import pyplot as plt

from src.helpers.config import EPOCHS, RESULTS_DIR
from src.helpers.logs.log_parser import LogParser

FEATURE_TYPES = [
    "Attr",
    "Attr + Str",
    "Attr + Str2",
    "Attr + Str3",
    "Attr + Error",
    "Attr + Emd"]
FEATURE_COLORS = {
    "Attr": "blue",
    "Attr + Str": "green",
    "Attr + Str2": "orange",
    "Attr + Str3": "red",
    "Attr + Error": "yellow",
    "Attr + Emd": "pink"
}
FEATURE_LABELS = {
    "Attr": "Attribute (alpha = 0.5)",
    "Attr + Str": "Attribute + Structure",
    "Attr + Str2": "Attribute + Structure 2",
    "Attr + Str3": "Attribute + Structure 3",
    "Attr + Emd": "Attribute + Embedding",
    "Attr + Error": "Attribute + Error",
}
DATASET_AUC_PAPER = {
    "cora": 0.762,
    "citeseer": 0.727,
    "BlogCatalog": 0.783,
    "weibo": 0.915,
    "Flickr": 0.751,
    "Reddit": 0.557
}

def main_auc_roc():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_TYPES:
            filtered = [
                r for r in parser.results
                if r["dataset"] == dataset and r["features"] == feature
            ]

            if not filtered:
                continue

            epoch_auc = defaultdict(float)
            for r in filtered:
                epoch = r["epoch"]
                auc = r["auc_roc"]
                if auc > epoch_auc[epoch]:
                    epoch_auc[epoch] = auc

            if not epoch_auc:
                continue

            aucs = [epoch_auc[e] for e in EPOCHS]
            plt.plot(EPOCHS, aucs, marker='o', label=FEATURE_LABELS[feature], color=FEATURE_COLORS[feature])

        save_path = os.path.join(save_dir, f"{dataset}_auc_plot.png")

        plt.title(f'AUC-ROC vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('AUC-ROC')
        if dataset in DATASET_AUC_PAPER:
            plt.axhline(y=DATASET_AUC_PAPER[dataset], color='purple', linestyle='--', label=f'Baseline ({DATASET_AUC_PAPER[dataset]})')
        else:
            plt.axhline(y=0.5, color='purple', linestyle='--', label='Baseline (0.5)')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

def main_loss():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_TYPES:
            filtered = [
                r for r in parser.results
                if r["dataset"] == dataset and r["features"] == feature
            ]

            if not filtered:
                continue

            epoch_loss = defaultdict(float)
            for r in filtered:
                epoch = r["epoch"]
                loss = r["loss"]
                if loss > epoch_loss[epoch]:
                    epoch_loss[epoch] = loss

            if not epoch_loss:
                continue

            losses = [epoch_loss[e] for e in EPOCHS]
            losses = [loss / (1.5 * max(losses)) for loss in losses]
            plt.plot(EPOCHS, losses, marker='o', label=FEATURE_LABELS[feature], color=FEATURE_COLORS[feature])

        save_path = os.path.join(save_dir, f"{dataset}_loss_plot.png")

        plt.title(f'Loss vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

if __name__ == "__main__":
    # main_loss()
    main_auc_roc()