"""
read_and_show_metrics.py
This file contains script to calculate and plot all metrics from .txt result files.
"""

from collections import defaultdict

from matplotlib import pyplot as plot

from src.helpers.config.const import *
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
    FEATURE_LABEL_STANDARD: "Basic node features",
    FEATURE_LABEL_STR: "Basic node features with structural features (pyFGLT)",
    FEATURE_LABEL_STR2: "Basic node features with structural features (NetworkX Features v1)",
    FEATURE_LABEL_STR3: "Basic node features with structural features (NetworkX Features v2)",
    FEATURE_LABEL_ERROR1: "Basic node features with reconstruction errors 1 (Simple autoencoder)",
    FEATURE_LABEL_ERROR2: "Basic node features with reconstruction errors 2 (AnomalyDAE autoencoder)",
    FEATURE_LABEL_EMD1: "Basic node features with embeddings 1 (alpha = 0 (node features))",
    FEATURE_LABEL_EMD2: "Basic node features with embeddings 2 (alpha = 1 (adjacent matrix))",
}


def create_metric_plot(
        metric_name: str, y_axis_label: str,
        baseline_dict_1: dict[str, float] = None,
        baseline_dict_2: dict[str, float] = None
):
    parser_1 = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser_1.parse_logs()
    parser_2 = LogParser(log_dir=RESULTS_DIR_COLA)
    parser_2.parse_logs()

    results_1 = sorted(parser_1.results, key=lambda x: x[DICT_DATASET])
    parser_1.results = results_1

    results_2 = sorted(parser_2.results, key=lambda x: x[DICT_DATASET])
    parser_2.results = results_2

    datasets = [result[DICT_DATASET] for result in results_1]
    datasets = list(set(datasets))

    for dataset in datasets:
        for i in range(0, 2):
            if i == 0:
                parser = parser_1
            else:
                parser = parser_2
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

            if i == 0:
                plot.title(f"AnomalyDAE - {y_axis_label} vs {VALUE_EPOCH} ({dataset})")
            else:
                plot.title(f"CoLA - {y_axis_label} vs {VALUE_EPOCH} ({dataset})")
            plot.xlabel(VALUE_EPOCH)
            if y_axis_label == VALUE_TIME:
                plot.ylabel(y_axis_label + " in s")
            else:
                plot.ylabel(y_axis_label)

            # normalizing
            if metric_name == DICT_LOSS or metric_name == DICT_TIME:
                plot.ylim(min_value, max_value)
                plot.yscale("log")

            plot.grid(True)
            plot.tight_layout()

            if metric_name == DICT_AUC_ROC:
                y = 0.5
                label = "Baseline (0.5)"
                # if benchmark result is given
                if i == 0:
                    baseline_dict = baseline_dict_1
                else:
                    baseline_dict = baseline_dict_2
                if baseline_dict is not None and dataset in baseline_dict:
                    y = baseline_dict[dataset]
                    label = f"Baseline ({baseline_dict[dataset]})"
                plot.axhline(y=y, color="purple", linestyle="--", label=label)
            plot.subplots_adjust(bottom=0.3)
            plot.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=2,
                frameon=True,
                fontsize="small",
                borderaxespad=0.0,
            )

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
        baseline_dict_1=AUC_ROC_PAPER_ANOMALYDAE,
        baseline_dict_2=AUC_ROC_PAPER_COLA,
    )


def plot_recall():
    create_metric_plot(metric_name=DICT_RECALL, y_axis_label=VALUE_RECALL)


def plot_precision():
    create_metric_plot(metric_name=DICT_PRECISION, y_axis_label=VALUE_PRECISION)


def plot_time():
    create_metric_plot(metric_name=DICT_TIME, y_axis_label=VALUE_TIME)


if __name__ == "__main__":
    # plot_loss()
    plot_auc_roc()
    # plot_recall()
    # plot_precision()
    # plot_time()
