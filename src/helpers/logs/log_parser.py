import logging
from pathlib import Path

from src.helpers.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)

class LogParser:
    def __init__(self, log_dir: Path = RESULTS_DIR):
        self.log_dir = log_dir
        self.results = []

    def parse_logs(self):
        for file in self.log_dir.glob("*.txt"):
            try:
                name = file.stem
                parts = name.split("_")

                dataset = parts[0]

                if "Attr + Str" in name:
                    features = "Attr + Str"
                elif "Attr + Alpha" in name:
                    features = "Attr + Alpha"
                elif "Attr + Emd" in name:
                    features = "Attr + Emd"
                else:
                    features = "Attr"

                lr_raw = parts[2]
                lr = int(lr_raw) / (10 ** (len(lr_raw) - 1))
                hid_dim = int(parts[3])
                epoch = int(parts[4])

                with open(file, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        continue
                    auc_line = lines[1].strip()
                    auc_roc = float(auc_line.split("AUC-ROC")[1].split(":")[1].strip())

                self.results.append({
                    "filename": file.name,
                    "dataset": dataset,
                    "features": features,
                    "lr": lr,
                    "epoch": epoch,
                    "hid_dim": hid_dim,
                    "auc_roc": auc_roc
                })

            except Exception as e:
                logging.warning(f"Failed to parse {file.name}: {e}")

    def get_best_by_key(self, key: str, value: float):
        filtered = [r for r in self.results if r[key] == value]
        return max(filtered, key=lambda x: x["auc_roc"], default=None)

    def get_best_and_worst(self):
        return (
            max(self.results, key=lambda x: x["auc_roc"]),
            min(self.results, key=lambda x: x["auc_roc"])
        )

    def get_result_by_params(self, dataset=None, features=None, lr=None, hid_dim=None, epoch=None):
        for result in self.results:
            if (
                (dataset is None or result["dataset"] == dataset) and
                (features is None or result["features"] == features) and
                (lr is None or result["lr"] == lr) and
                (hid_dim is None or result["hid_dim"] == hid_dim) and
                (epoch is None or result["epoch"] == epoch)
            ):
                return result
        return None
