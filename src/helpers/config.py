from enum import Enum
from pathlib import Path

class DataSetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "anomalyedae"
DATASETS_DIR = BASE_DIR / "datasets"

SMALL_DATASETS = ["Disney.mat",
                  "book.mat",
                  "BlogCatalog.mat",
                  "citeseer.mat",
                  "computers.mat",
                  "cora.mat",
                  "cs.mat",
                  "photo.mat",
                  "weibo.mat"]
MEDIUM_DATASETS = ["Flickr.mat",
                  "Reddit.mat"]

TO_EMD_DATASETS = SMALL_DATASETS

CURRENT_DATASETS = SMALL_DATASETS + MEDIUM_DATASETS
CURRENT_DATASETS_SIZE = [DataSetSize.SMALL] * len(SMALL_DATASETS) + [DataSetSize.MEDIUM] * len(MEDIUM_DATASETS)

EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

labels_dict = {}
graph_dict = {}

LEARNING_RATE = 0.001
HIDDEN_DIMS = 16

SEED = 42