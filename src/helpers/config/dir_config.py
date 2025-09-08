"""
dir_config.py
This config file contains:
- constants used for routes and directories in project.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR_ANOMALYDAE = BASE_DIR / "results" / "anomalyedae"
RESULTS_DIR_COLA = BASE_DIR / "results" / "cola"
RESULTS_DIR_OCGNN = BASE_DIR / "results" / "ocgnn"
DATASETS_DIR = BASE_DIR / "datasets"
SAVE_DIR_ANOMALYDAE = RESULTS_DIR_ANOMALYDAE
SAVE_DIR_COLA = RESULTS_DIR_COLA
SAVE_DIR_OCGNN = RESULTS_DIR_OCGNN
