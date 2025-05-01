from collections import defaultdict

from matplotlib import pyplot as plt

from src.helpers.logs.log_parser import LogParser

CONFIG = {
    "epochs": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
}

FEATURE_TYPES = ["Attr",
                 "Attr + Str",
                 "Attr + Str2",
                 "Attr + Emd"]
FEATURE_COLORS = {
    "Attr": "blue",
    "Attr + Str": "green",
    "Attr + Str2": "brown",
    "Attr + Alpha": "yellow",
    "Attr + Emd": "red",
}
FEATURE_LABELS = {
    "Attr": "Attribute (alpha = 0.5)",
    "Attr + Str": "Attribute + Structure",
    "Attr + Str2": "Attribute + Structure 2",
    "Attr + Alpha": "Attribute (alpha = 0)",
    "Attr + Emd": "Attribute + Embedding"
}
DATASET_AUC_PAPER = {
    "cora": 0.762,
    "citeseer": 0.727,
    "BlogCatalog": 0.783,
    "weibo": 0.915
}

def main():
    parser = LogParser()
    parser.parse_logs()

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

            epochs = [25, 50, 75, 100, 125, 150]
            aucs = [epoch_auc[e] for e in epochs]
            plt.plot(epochs, aucs, marker='o', label=FEATURE_LABELS[feature], color=FEATURE_COLORS[feature])

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
        plt.show()

if __name__ == "__main__":
    main()