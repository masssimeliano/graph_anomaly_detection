"""
read_and_show_metrics.py
This file contains script to calculate and plot all metrics from .txt result files.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
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
    FEATURE_LABEL_STANDARD: "No enrichment",
    FEATURE_LABEL_STR: "Structural features (pyFGLT)",
    FEATURE_LABEL_STR2: "Structural features (NetworkX Features v1)",
    FEATURE_LABEL_STR3: "Structural features (NetworkX Features v2)",
    FEATURE_LABEL_ERROR1: "Reconstruction errors 1 (Simple autoencoder)",
    FEATURE_LABEL_ERROR2: "Reconstruction errors 2 (AnomalyDAE autoencoder)",
    FEATURE_LABEL_EMD1: "Embeddings 1 (alpha = 0 (node features))",
    FEATURE_LABEL_EMD2: "Embeddings 2 (alpha = 1 (adjacent matrix))",
}


def create_metric_plot(
    metric_name: str, y_axis_label: str, baseline_dict: dict[str, float] = None
):
    parser = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser.parse_logs()

    datasets = set(result[DICT_DATASET] for result in parser.results)

    for dataset in datasets:
        datasets_current = [dataset]

        for dataset in datasets_current:
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
        plot.subplots_adjust(bottom=0.3)
        plot.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=True,
            fontsize="small",
            borderaxespad=0.0,
        )

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
        baseline_dict=AUC_ROC_PAPER_ANOMALYDAE,
    )


def plot_recall():
    create_metric_plot(metric_name=DICT_RECALL, y_axis_label=VALUE_RECALL)


def plot_precision():
    create_metric_plot(metric_name=DICT_PRECISION, y_axis_label=VALUE_PRECISION)


def plot_time():
    create_metric_plot(metric_name=DICT_TIME, y_axis_label=VALUE_TIME)


def make_epoch_pivot_table(
    target_epoch: int = 100,
    metrics: tuple[str, ...] = (DICT_AUC_ROC, DICT_RECALL, DICT_PRECISION),
    metric_display: dict[str, str] = None,
):
    if metric_display is None:
        metric_display = {
            DICT_AUC_ROC: "AUC-ROC",
            DICT_RECALL: "Recall",
            DICT_PRECISION: "Precision",
        }

    parser = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser.parse_logs()

    datasets = sorted(set(r[DICT_DATASET] for r in parser.results))
    enrichments = [FEATURE_LABELS_DICT[f] for f in FEATURE_LABELS]
    metrics_disp = [metric_display[m] for m in metrics]

    index = pd.MultiIndex.from_product(
        [metrics_disp, enrichments], names=["metric", "enrichment"]
    )
    df = pd.DataFrame(index=index, columns=datasets, dtype=float)

    for m in metrics:
        m_name = metric_display[m]
        for f in FEATURE_LABELS:
            f_name = FEATURE_LABELS_DICT[f]
            for d in datasets:
                vals = [
                    r.get(m, np.nan)
                    for r in parser.results
                    if r[DICT_DATASET] == d
                    and r[DICT_FEATURE_LABEL] == f
                    and r[DICT_EPOCH] == target_epoch
                ]
                val = vals[-1] if vals else np.nan
                df.loc[(m_name, f_name), d] = val

    save_csv_path = os.path.join(
        SAVE_DIR_ANOMALYDAE, f"pivot_metrics_epoch{target_epoch}.csv"
    )
    df.to_csv(save_csv_path, index=True)

    out_path = "pivot_metrics_epoch100.tex"
    latex = df.to_latex(
        index=True,
        multirow=True,
        escape=False,
        float_format="%.3f",
        caption="Metrics at the 100th epoch (AUC-ROC, Recall, Precision)",
        label="tab:metrics_epoch100",
        bold_rows=False,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return df

if __name__ == "__main__":
    parser = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser.parse_logs()
    parser = LogParser(log_dir=RESULTS_DIR_COLA)
    parser.parse_logs()
    parser = LogParser(log_dir=RESULTS_DIR_OCGNN)
    parser.parse_logs()
