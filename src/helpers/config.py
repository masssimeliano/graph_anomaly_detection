from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "unsupervised" / "anomalyedae"
BEST_MODEL_ATTR_STR_PATH = RESULTS_DIR / "best" / "emd_0001_16_100.pt"
DATASETS_DIR = BASE_DIR / "datasets"

__all__ = [
    "BASE_DIR",
    "RESULTS_DIR",
    "BEST_MODEL_ATTR_STR_PATH",
    "DATASETS_DIR",
]