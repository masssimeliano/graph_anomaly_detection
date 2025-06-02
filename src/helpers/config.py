from enum import Enum
from pathlib import Path

class DataSetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "anomalyedae"
DATASETS_DIR = BASE_DIR / "datasets"

SMALL_DATASETS = [
                  "cs.mat",
                  "photo.mat",
                  "weibo.mat"]
MEDIUM_DATASETS = ["Flickr.mat",
                  "Reddit.mat"]

TO_EMD_DATASETS = SMALL_DATASETS

DATASETS = [
            "cora.mat"]

CURRENT_DATASETS = SMALL_DATASETS
CURRENT_DATASETS_SIZE = [DataSetSize.SMALL] * len(CURRENT_DATASETS)

EPOCH_TO_LEARN = 100
EPOCHS = list(range(10, 101, 10))

labels_dict = {}
graph_dict = {}

LEARNING_RATE = 0.005
HIDDEN_DIMS = 16
ETA = 5
THETA = 40

SEED = 42