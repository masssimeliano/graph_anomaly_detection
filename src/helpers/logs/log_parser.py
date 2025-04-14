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
                with open(file, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        continue

                    config_line = lines[0].strip()
                    auc_line = lines[1].strip()

                    lr = float(config_line.split("lr=")[1].split(",")[0])
                    hid_dim = int(config_line.split("hid_dim=")[1].split(")")[0])
                    epoch = int(config_line.split("epoch=")[1].split(",")[0])
                    auc_roc = float(auc_line.split("AUC-ROC")[1].split(":")[1].strip())

                    self.results.append({
                        "filename": file.name,
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
        return max(self.results, key=lambda x: x["auc_roc"]), min(self.results, key=lambda x: x["auc_roc"])
