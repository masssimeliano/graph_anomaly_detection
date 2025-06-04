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
    FEATURE_LABEL_STR: "Attribute + Structure (pyfglt)",
    FEATURE_LABEL_STR2: "Attribute + Structure 2 (NetworkX Features v1)",
    FEATURE_LABEL_STR3: "Attribute + Structure 3 (NetworkX Features v2)",
    FEATURE_LABEL_EMD1: "Attribute + Embedding 1 (Embedding of Attribute (alpha = 0))",
    FEATURE_LABEL_EMD2: "Attribute + Embedding 2 (Embedding of Attribute (alpha = 1)",
    FEATURE_LABEL_ERROR1: "Attribute + Error 1 (Reconstruction error from simple encoder)",
    FEATURE_LABEL_ERROR2: "Attribute + Error 2 (Reconstruction error from AnomalyDAE encoder)",
}


def plot_metric(metric_key: str,
                ylabel: str,
                baseline_dict=None):
    parser = LogParser()
    parser.parse_logs()

    save_dir = RESULTS_DIR / "graph" / "dev"
    datasets = set(r[DICT_DATASET] for r in parser.results)

    for dataset in datasets:
        max_value = get_max_value_for_dataset_and_metric(dataset=dataset,
                                                         parser=parser,
                                                         metric_key=metric_key)

        plt.figure(figsize=(10, 6))

        for feature in FEATURE_LABELS:
            filtered = [
                r for r in parser.results
                if r[DICT_DATASET] == dataset and r[DICT_FEATURE_LABEL] == feature
            ]
            if not filtered:
                continue

            epoch_values = defaultdict(float)
            for r in filtered:
                epoch = r[DICT_EPOCH]
                value = r.get(metric_key, 0.0)
                epoch_values[epoch] = value

            if not epoch_values:
                continue

            values = [epoch_values[e] for e in EPOCHS]
            plt.plot(EPOCHS, values, marker='o',
                     label=FEATURE_LABELS_DICT[feature],
                     color=FEATURE_COLORS_DICT[feature])

        plt.title(f'{ylabel} vs Epochs ({dataset})')
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.ylim(0.0, 1.5 * max_value)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if baseline_dict is not None and dataset in baseline_dict:
            plt.axhline(y=baseline_dict[dataset],
                        color='purple',
                        linestyle='--',
                        label=f'Baseline ({baseline_dict[dataset]})')
        elif baseline_dict:
            plt.axhline(y=0.5,
                        color='purple',
                        linestyle='--',
                        label='Baseline (0.5)')

        save_path = os.path.join(save_dir, f"{dataset}_{ylabel}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()


def get_max_value_for_dataset_and_metric(dataset: str,
                                         parser: LogParser,
                                         metric_key: str) -> float:
    max_val = 0

    for feature in FEATURE_LABELS:
        filtered = [
            r for r in parser.results
            if r[DICT_DATASET] == dataset and r[DICT_FEATURE_LABEL] == feature
        ]
        if not filtered:
            continue

        for r in filtered:
            value = r.get(metric_key, 0.0)
            if value > max_val:
                max_val = value

    return max_val


def plot_loss():
    plot_metric(metric_key=DICT_LOSS,
                ylabel=VALUE_LOSS)


def plot_auc_roc():
    plot_metric(metric_key=DICT_AUC_ROC,
                ylabel=VALUE_AUC_ROC,
                baseline_dict=AUC_ROC_PAPER)


def plot_recall():
    plot_metric(metric_key=DICT_RECALL,
                ylabel=VALUE_RECALL)


def plot_precision():
    plot_metric(metric_key=DICT_PRECISION,
                ylabel=VALUE_PRECISION)


def plot_time():
    plot_metric(metric_key=DICT_TIME,
                ylabel=VALUE_TIME)


if __name__ == "__main__":
    plot_loss()
    plot_auc_roc()
    plot_recall()
    plot_precision()
    plot_time()
