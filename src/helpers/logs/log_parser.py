import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.helpers.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)


class LogParser:
    def __init__(self,
                 log_dir: Path = RESULTS_DIR):
        self.log_dir = log_dir
        self.results = []

    def parse_logs(self):
        for file in self.log_dir.glob("*.txt"):
            try:
                name = file.stem
                parts = name.split("_")

                dataset = parts[0]

                if "Attr + Str2" in name:
                    features = "Attr + Str2"
                elif "Attr + Str3" in name:
                    features = "Attr + Str3"
                elif "Attr + Alpha" in name:
                    features = "Attr + Alpha"
                elif "Attr + Emd" in name:
                    features = "Attr + Emd"
                elif "Attr + Str" in name:
                    features = "Attr + Str"
                elif "Attr + Error" in name:
                    features = "Attr + Error"
                else:
                    features = "Attr"

                lr_raw = parts[2]
                lr = int(lr_raw) / (10 ** (len(lr_raw) - 1))
                hid_dim = int(parts[3])
                epoch = int(parts[4])

                with open(file, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 5:
                        continue

                    auc_line = next((line for line in lines if "AUC-ROC" in line), None)
                    loss_line = next((line for line in lines if "Loss" in line), None)
                    recall_line = next((line for line in lines if "Recall" in line), None)
                    precision_line = next((line for line in lines if "Precision" in line), None)
                    auc_roc = float(auc_line.split("AUC-ROC")[1].split(":")[1].strip())
                    loss_value = float(loss_line.split("Loss")[1].split(":")[1].strip())
                    recall = float(recall_line.split("Recall")[1].split(":")[1].strip())
                    precision = float(precision_line.split("Precision")[1].split(":")[1].strip())

                self.results.append({"filename": file.name,
                                     "dataset": dataset,
                                     "features": features,
                                     "lr": lr,
                                     "epoch": epoch,
                                     "hid_dim": hid_dim,
                                     "auc_roc": auc_roc,
                                     "loss": loss_value,
                                     "recall": recall,
                                     "precision": precision})

            except Exception as e:
                logging.warning(f"Failed to parse {file.name}: {e}")

    def get_best_by_key(self,
                        key: str,
                        value: float) -> Optional[Dict[str, Any]]:
        filtered = [r for r in self.results if r[key] == value]
        return max(filtered, key=lambda x: x["auc_roc"], default=None)

    def get_best_and_worst(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        return (max(self.results, key=lambda x: x["auc_roc"]),
                min(self.results, key=lambda x: x["auc_roc"]))

    def get_result_by_params(self,
                             dataset=None,
                             features=None,
                             lr=None,
                             hid_dim=None,
                             epoch=None,
                             loss_value=None) -> Optional[Dict[str, Any]]:
        for result in self.results:
            if ((dataset is None or result["dataset"] == dataset) and
                (features is None or result["features"] == features) and
                (lr is None or result["lr"] == lr) and
                (hid_dim is None or result["hid_dim"] == hid_dim) and
                (epoch is None or result["epoch"] == epoch) and
                (loss_value is None or result["loss"] == loss_value)):
                return result
        return None