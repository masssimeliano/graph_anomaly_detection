from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results" / "anomalyedae"
DATASETS_DIR = BASE_DIR / "datasets"
SAVE_DIR = RESULTS_DIR / "graph" / "dev"
