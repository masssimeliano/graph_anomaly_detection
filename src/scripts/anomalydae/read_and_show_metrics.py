"""
read_and_show_metrics.py
This file contains script to calculate and plot all metrics from .txt result files.
"""

import os
from collections import defaultdict

from matplotlib import pyplot as plot

from src.helpers.config.const import *
from src.helpers.config.datasets_config import CHECK_DATASETS_2
from src.helpers.config.dir_config import *
from src.helpers.config.training_config import *
from src.helpers.logs.log_parser import LogParser

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
    FEATURE_LABEL_ERROR1: "Attribute + Error 1 (Reconstruction error from simple encoder)",
    FEATURE_LABEL_ERROR2: "Attribute + Error 2 (Reconstruction error from AnomalyDAE encoder)",
    FEATURE_LABEL_EMD1: "Attribute + Embedding 1 (Embedding of Attribute (alpha = 0 (node features)))",
    FEATURE_LABEL_EMD2: "Attribute + Embedding 2 (Embedding of Attribute (alpha = 1 (adjacent matrix)))",
}


def create_metric_plot(
    metric_name: str, y_axis_label: str, baseline_dict: dict[str, float] = None
):
    parser = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser.parse_logs()

    datasets = set(result[DICT_DATASET] for result in parser.results)

    for dataset in CHECK_DATASETS_2:
        max_value = get_max_value_for_dataset_and_metric(
            dataset=dataset, parser=parser, metric_name=metric_name
        )
        min_value = get_min_value_for_dataset_and_metric(
            dataset=dataset, parser=parser, metric_name=metric_name
        )

        plot.figure(figsize=(10, 6))

        for feature_label in FEATURE_LABELS:
            filtered_feature_labels = [
                result
                for result in parser.results
                if result[DICT_DATASET] == dataset
                and result[DICT_FEATURE_LABEL] == feature_label
            ]
            if not filtered_feature_labels:
                continue

            value_per_epochs = defaultdict(float)
            for result in filtered_feature_labels:
                epoch = result[DICT_EPOCH]
                value = result.get(metric_name, 0)
                value_per_epochs[epoch] = value

            if not value_per_epochs:
                continue

            values = [value_per_epochs[epoch] for epoch in EPOCHS]
            plot.plot(
                EPOCHS,
                values,
                marker="o",
                label=FEATURE_LABELS_DICT[feature_label],
                color=FEATURE_COLORS_DICT[feature_label],
            )

        plot.title(f"AnomalyDAE - {y_axis_label} vs {VALUE_EPOCH} ({dataset})")
        plot.xlabel(VALUE_EPOCH)
        if y_axis_label == VALUE_TIME:
            plot.ylabel(y_axis_label + " in s")
        else:
            plot.ylabel(y_axis_label)

        # normalizing
        if metric_name == DICT_LOSS or metric_name == DICT_TIME:
            plot.ylim(min_value, max_value)
            plot.yscale("log")
        else:
            plot.ylim(0.9 * min_value, 0.75 * (max_value + min_value))

        plot.grid(True)
        plot.tight_layout()

        if metric_name == DICT_AUC_ROC:
            y = 0.5
            label = "Baseline (0.5)"
            # if benchmark result is given
            if baseline_dict is not None and dataset in baseline_dict:
                y = baseline_dict[dataset]
                label = f"Baseline ({baseline_dict[dataset]})"
            plot.axhline(y=y, color="purple", linestyle="--", label=label)
        plot.legend()

        save_path = os.path.join(SAVE_DIR_ANOMALYDAE, f"{dataset}_{y_axis_label}.png")
        plot.savefig(save_path, dpi=300)
        plot.show()


def get_max_value_for_dataset_and_metric(
    dataset: str, parser: LogParser, metric_name: str
) -> float:
    max_value = 0

    for feature_label in FEATURE_LABELS:
        filtered_parser_result = [
            result
            for result in parser.results
            if result[DICT_DATASET] == dataset
            and result[DICT_FEATURE_LABEL] == feature_label
        ]
        if not filtered_parser_result:
            continue

        for result in filtered_parser_result:
            value = result.get(metric_name, 0)
            if value > max_value:
                max_value = value

    return max_value


def get_min_value_for_dataset_and_metric(
    dataset: str, parser: LogParser, metric_name: str
) -> float:
    min_value = get_max_value_for_dataset_and_metric(
        dataset=dataset, parser=parser, metric_name=metric_name
    )

    for feature_label in FEATURE_LABELS:
        filtered_parser_result = [
            result
            for result in parser.results
            if result[DICT_DATASET] == dataset
            and result[DICT_FEATURE_LABEL] == feature_label
        ]
        if not filtered_parser_result:
            continue

        for result in filtered_parser_result:
            value = result.get(metric_name, 0)
            if value < min_value:
                min_value = value

    return min_value


def plot_loss():
    create_metric_plot(metric_name=DICT_LOSS, y_axis_label=VALUE_LOSS)


def plot_auc_roc():
    create_metric_plot(
        metric_name=DICT_AUC_ROC,
        y_axis_label=VALUE_AUC_ROC,
        baseline_dict=AUC_ROC_PAPER,
    )


def plot_recall():
    create_metric_plot(metric_name=DICT_RECALL, y_axis_label=VALUE_RECALL)


def plot_precision():
    create_metric_plot(metric_name=DICT_PRECISION, y_axis_label=VALUE_PRECISION)


def plot_time():
    create_metric_plot(metric_name=DICT_TIME, y_axis_label=VALUE_TIME)


if __name__ == "__main__":
    plot_loss()
    plot_auc_roc()
    plot_recall()
    plot_precision()
    plot_time()
