import logging
import os
from collections import defaultdict

from matplotlib import pyplot as plt

from src.helpers.config.const import *
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.logs.log_parser import LogParser

logging.basicConfig(level=logging.INFO)

FEATURE_LABELS = [
    FEATURE_LABEL_STANDARD,
    FEATURE_LABEL_STR,
    FEATURE_LABEL_STR2,
    FEATURE_LABEL_STR3,
    FEATURE_LABEL_ERROR1,
    FEATURE_LABEL_ERROR2,
    FEATURE_LABEL_EMD1,
    FEATURE_LABEL_EMD2,
]
FEATURE_COLORS_DICT = {
    FEATURE_LABEL_STANDARD: "blue",
    FEATURE_LABEL_STR: "green",
    FEATURE_LABEL_STR2: "orange",
    FEATURE_LABEL_STR3: "red",
    FEATURE_LABEL_ERROR1: "yellow",
    FEATURE_LABEL_ERROR2: "pink",
    FEATURE_LABEL_EMD1: "black",
    FEATURE_LABEL_EMD2: "gray",
}
FEATURE_LABELS_DICT = {
    FEATURE_LABEL_STANDARD: "Attribute (alpha = 0.5)",
    FEATURE_LABEL_STR: "Attribute + Structure",
    FEATURE_LABEL_STR2: "Attribute + Structure 2",
    FEATURE_LABEL_STR3: "Attribute + Structure 3",
    FEATURE_LABEL_EMD1: "Attribute + Embedding 1",
    FEATURE_LABEL_EMD2: "Attribute + Embedding 2",
    FEATURE_LABEL_ERROR1: "Attribute + Error 1",
    FEATURE_LABEL_ERROR2: "Attribute + Error 2",
}


def main_auc_roc():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_LABELS:
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
            plt.plot(EPOCHS, aucs, marker='o', label=FEATURE_LABELS_DICT[feature], color=FEATURE_COLORS_DICT[feature])

        save_path = os.path.join(save_dir, f"{dataset}_auc_plot.png")

        plt.title(f'AUC-ROC vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('AUC-ROC')
        if dataset in AUC_ROC_PAPER:
            plt.axhline(y=AUC_ROC_PAPER[dataset], color='purple', linestyle='--',
                        label=f'Baseline ({AUC_ROC_PAPER[dataset]})')
        else:
            plt.axhline(y=0.5, color='purple', linestyle='--', label='Baseline (0.5)')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{dataset}_loss_auc.png", dpi=300)
        plt.show()


def main_loss():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_LABELS:
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
            plt.plot(EPOCHS, losses, marker='o', label=FEATURE_LABELS_DICT[feature], color=FEATURE_COLORS_DICT[feature])

        save_path = os.path.join(save_dir, f"{dataset}_loss_plot.png")

        plt.title(f'Loss vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{dataset}_loss_plot.png", dpi=300)
        plt.show()


def main_recall():
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

            epoch_recall = defaultdict(float)
            for r in filtered:
                epoch = r["epoch"]
                recall = r["recall"]
                if recall > epoch_recall[epoch]:
                    epoch_recall[epoch] = recall

            if not epoch_recall:
                continue

            recalls = [epoch_recall[e] for e in EPOCHS]
            plt.plot(EPOCHS, recalls, marker='o', label=FEATURE_LABELS_DICT[feature],
                     color=FEATURE_COLORS_DICT[feature])

        save_path = os.path.join(save_dir, f"{dataset}_recall_plot.png")

        plt.title(f'Recall@k vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('Recall@k')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{dataset}_recall_plot.png", dpi=300)
        plt.show()


def main_precision():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r["dataset"] for r in parser.results)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_LABELS:
            filtered = [
                r for r in parser.results
                if r["dataset"] == dataset and r["features"] == feature
            ]

            if not filtered:
                continue

            epoch_precision = defaultdict(float)
            for r in filtered:
                epoch = r["epoch"]
                precision = r["precision"]
                if precision > epoch_precision[epoch]:
                    epoch_precision[epoch] = precision

            if not epoch_precision:
                continue

            precisions = [epoch_precision[e] for e in EPOCHS]
            plt.plot(EPOCHS, precisions, marker='o', label=FEATURE_LABELS_DICT[feature],
                     color=FEATURE_COLORS_DICT[feature])

        save_path = os.path.join(save_dir, f"{dataset}_precision_plot.png")

        plt.title(f'Precision@k vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('Precision@k')
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{dataset}_precision_plot.png", dpi=300)
        plt.show()


def main_time():
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"

    datasets = set(r[DICT_DATASET] for r in parser.results)

    max_time = 0
    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for feature in FEATURE_LABELS:
            filtered = [
                r for r in parser.results
                if r[DICT_DATASET] == dataset and r[DICT_FEATURE_LABEL] == feature
            ]

            if not filtered:
                continue

            epoch_time = defaultdict(float)
            for r in filtered:
                epoch = r[DICT_EPOCH]
                time = r[DICT_TIME]
                if time > epoch_time[epoch]:
                    epoch_time[epoch] = time

            if not epoch_time:
                continue

            times = [epoch_time[e] for e in EPOCHS]
            plt.plot(EPOCHS, times, marker='o', label=FEATURE_LABELS_DICT[feature], color=FEATURE_COLORS_DICT[feature])
            if max(times) > max_time:
                max_time = max(times)

        save_path = os.path.join(save_dir, f"{dataset}_time_plot.png")

        plt.title(f'Time vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel('Time in s')
        plt.ylim(0.0, max_time)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{dataset}_time_plot.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    # main_loss()
    # main_auc_roc()
    # main_recall()
    # main_precision()
    main_time()
