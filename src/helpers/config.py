from pathlib import Path

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

CURRENT_DATASETS = ["Disney", "book.mat"]

EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

LEARNING_RATE = 0.001
HIDDEN_DIMS = 16

SEED = 42