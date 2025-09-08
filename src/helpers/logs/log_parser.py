"""
log_parser.py
This file generates log dictionaries from .txt result files.
"""
import json
import logging
from collections import defaultdict
from typing import Dict, Any, List

from src.helpers.config.const import *
from src.helpers.config.dir_config import *


def extract_value(line: str) -> float:
    line_splits: list[str] = line.split(":")
    return float(line_splits[len(line_splits) - 1].strip())


class LogParser:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.results: List[Dict[str, Any]] = []
        self.output_dir = log_dir

    def parse_logs(self):
        logging.info("Parsing logs...")

        for file in self.log_dir.glob("*.txt"):
            try:
                parts = file.stem.split("_")
                if len(parts) < 5:
                    logging.warning(f"Invalid filename format: {file.name}")
                    continue

                dataset = parts[0]
                feature_label = next(
                    (f for f in FEATURE_LABELS if f in file.stem),
                    FEATURE_LABEL_STANDARD,
                )

                # e.g. string 0001 -> float 0.001
                lr = int(parts[2]) / (10 ** (len(parts[2]) - 1))
                hid_dim = int(parts[3])
                epoch = int(parts[4])

                with file.open() as f:
                    content = f.read()

                if self.log_dir == RESULTS_DIR_ANOMALYDAE:
                    model = "AnomalyDAE"
                else:
                    if self.log_dir == RESULTS_DIR_COLA:
                        model = "CoLA"
                    else:
                        model = "OCGNN"

                self.results.append(
                    {
                        DICT_FILE_NAME: file.name,
                        DICT_DATASET: dataset,
                        DICT_FEATURE_LABEL: feature_label,
                        DICT_LR: lr,
                        DICT_EPOCH: epoch,
                        DICT_HID_DIM: hid_dim,
                        DICT_AUC_ROC: extract_value(content.splitlines()[1]),
                        DICT_LOSS: extract_value(content.splitlines()[2]),
                        DICT_RECALL: extract_value(content.splitlines()[3]),
                        DICT_PRECISION: extract_value(content.splitlines()[4]),
                        DICT_TIME: extract_value(content.splitlines()[5]),
                        DICT_MODEL: model,
                    }
                )

            except Exception as e:
                logging.warning(f"Failed to parse {file.name}: {e}")


    def export_metric_jsons(self):
        metrics = {
            "aucroc": DICT_AUC_ROC,
            "loss": DICT_LOSS,
            "precision": DICT_PRECISION,
            "recall": DICT_RECALL,
        }

        data_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for r in self.results:
            dataset = r[DICT_DATASET]
            feature = r[DICT_FEATURE_LABEL]
            epoch = r[DICT_EPOCH]
            for name, key in metrics.items():
                value = r[key]
                data_map[dataset][feature][name].append({"epoch": epoch, "value": value})

        for dataset, feat_map in data_map.items():
            for feature, mdata in feat_map.items():
                for metric_name, arr in mdata.items():
                    arr.sort(key=lambda x: x["epoch"])
                    safe_feat = feature.replace(" ", "_").replace("/", "_")
                    out_path = self.output_dir / f"{dataset}_{metric_name}_{safe_feat}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(arr, f, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {out_path}")