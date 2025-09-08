"""
log_parser.py
This file generates log dictionaries from .txt result files.
"""

import logging
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
