import logging
from typing import Optional, Dict, Any, List, AnyStr

from src.helpers.config.const import *
from src.helpers.config.dir_config import *

logging.basicConfig(level=logging.INFO)


def extract_value(label: str, content: AnyStr) -> float:
    for line in content.splitlines():
        if label in line:
            return float(line.split(":")[1].strip())
    raise ValueError(f"{label} not found")


class LogParser:
    def __init__(self,
                 log_dir: Path = RESULTS_DIR):
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
                feature_label = next((f for f in self.FEATURE_LABELS if f in file.stem), self.FEATURE_LABEL_STANDARD)
                # e.g. string 0001 -> float 0.001
                lr = int(parts[2]) / (10 ** (len(parts[2]) - 1))
                hid_dim = int(parts[3])
                epoch = int(parts[4])

                with file.open() as f:
                    content = f.read()

                self.results.append({
                    DICT_FILE_NAME: file.name,
                    DICT_DATASET: dataset,
                    DICT_FEATURE_LABEL: feature_label,
                    DICT_LR: lr,
                    DICT_EPOCH: epoch,
                    DICT_HID_DIM: hid_dim,
                    DICT_AUC_ROC: extract_value(VALUE_AUC_ROC, content),
                    DICT_LOSS: extract_value(VALUE_LOSS, content),
                    DICT_RECALL: extract_value(VALUE_RECALL, content),
                    DICT_PRECISION: extract_value(VALUE_PRECISION, content),
                    DICT_TIME: extract_value(VALUE_TIME, content)
                })

            except Exception as e:
                logging.warning(f"Failed to parse {file.name}: {e}")

    def get_result_by_params(self,
                             file_name: str = None,
                             dataset: str = None,
                             feature_label: str = None,
                             lr: float = None,
                             epoch: int = None,
                             hid_dim: int = None) -> Optional[Dict[str, Any]]:
        for result in self.results:
            if file_name is not None and result[DICT_FILE_NAME] != file_name:
                continue
            if dataset is not None and result[DICT_DATASET] != dataset:
                continue
            if feature_label is not None and result[DICT_FEATURE_LABEL] != feature_label:
                continue
            if lr is not None and result[DICT_LR] != lr:
                continue
            if epoch is not None and result[DICT_EPOCH] != epoch:
                continue
            if hid_dim is not None and result[DICT_HID_DIM] != hid_dim:
                continue
            return result
        return None
