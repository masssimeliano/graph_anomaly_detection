"""
read_and_show_metrics.py
This file contains script to calculate and plot all metrics from .txt result files.
"""
import os

import numpy as np
import seaborn as sns
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


def plot_auc_roc():
    from src.scripts.anomalydae.read_and_show_metrics import plot_auc_roc as plot1
    from src.scripts.cola.read_and_show_metrics import plot_auc_roc as plot2
    from src.scripts.ocgnn.read_and_show_metrics import plot_auc_roc as plot3
    plot1()
    plot2()
    plot3()


def plot_recall():
    from src.scripts.anomalydae.read_and_show_metrics import plot_recall as plot1
    from src.scripts.cola.read_and_show_metrics import plot_recall as plot2
    from src.scripts.ocgnn.read_and_show_metrics import plot_recall as plot3
    plot1()
    plot2()
    plot3()


def plot_precision():
    from src.scripts.anomalydae.read_and_show_metrics import plot_precision as plot1
    from src.scripts.cola.read_and_show_metrics import plot_precision as plot2
    from src.scripts.ocgnn.read_and_show_metrics import plot_precision as plot3
    plot1()
    plot2()
    plot3()


def plot_time():
    from src.scripts.anomalydae.read_and_show_metrics import plot_time as plot1
    from src.scripts.cola.read_and_show_metrics import plot_time as plot2
    from src.scripts.ocgnn.read_and_show_metrics import plot_time as plot3
    plot1()
    plot2()
    plot3()


def plot_heatmap_with_models(
        metric_name: str,
        title: str,
        models: list[str] = ["AnomalyDAE", "OCGNN", "CoLA"],
        cmap: str = "viridis"
):
    parser_1 = LogParser(log_dir=RESULTS_DIR_ANOMALYDAE)
    parser_1.parse_logs()
    parser_2 = LogParser(log_dir=RESULTS_DIR_COLA)
    parser_2.parse_logs()
    parser_3 = LogParser(log_dir=RESULTS_DIR_OCGNN)
    parser_3.parse_logs()

    results = parser_1.results + parser_2.results + parser_3.results

    datasets = sorted(set(r[DICT_DATASET] for r in results))
    target_epoch = EPOCH_TO_LEARN

    values = []
    for feature_label in FEATURE_LABELS:
        row = []
        for ds in datasets:
            for model in models:
                items = [
                    r
                    for r in results
                    if r[DICT_DATASET] == ds
                    and r.get("model") == model
                    and r[DICT_FEATURE_LABEL] == feature_label
                    and r[DICT_EPOCH] == target_epoch
                ]
                row.append(items[0].get(metric_name, np.nan) if items else np.nan)
        values.append(row)

    values = np.array(values).T

    x_index = [FEATURE_LABELS_DICT[l] for l in FEATURE_LABELS]

    fig, ax = plot.subplots(figsize=(36, 50))
    sns.heatmap(
        values,
        cmap=cmap,
        cbar=True,
        annot=False,
        linewidths=0.5,
        xticklabels=x_index,
        yticklabels=False,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=20)

    n_datasets = len(datasets)
    m = len(models)
    nrows = n_datasets * m

    row_centers = np.arange(nrows) + 0.5

    group_centers = np.arange(n_datasets) * m + (m / 2) + 0.5
    ax.set_yticks(group_centers)
    ax.set_yticklabels(datasets, rotation=0, va="center", fontsize=20)

    ax.set_yticks(row_centers, minor=True)
    ax.set_yticklabels(
        [models[j % m] for j in range(nrows)],
        minor=True,
        rotation=0,
        va="center",
        fontsize=20,
    )

    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        pad=80,
    )

    plot.title(f"{title} (Epoch {target_epoch})", fontweight="bold")
    plot.tight_layout()

    save_path = os.path.join(
        RESULTS_DIR, f"heatmap_{metric_name}_epoch{target_epoch}_swapped.png"
    )
    plot.savefig(save_path, dpi=300)
    plot.show()


def plot_heatmap_auc_roc():
    plot_heatmap_with_models(DICT_AUC_ROC, VALUE_AUC_ROC)

def plot_heatmap_precision():
    plot_heatmap_with_models(DICT_PRECISION, VALUE_PRECISION)

def plot_heatmap_recall():
    plot_heatmap_with_models(DICT_RECALL, VALUE_RECALL)


if __name__ == "__main__":
    # plot_loss()
    # plot_auc_roc()
    # plot_recall()
    # plot_precision()
    # plot_time()
    plot_heatmap_auc_roc()
    plot_heatmap_precision()
    plot_heatmap_recall()

