from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "unsupervised" / "anomalyedae"
DATASETS_DIR = BASE_DIR / "datasets"

SMALL_DATASETS = ["BlogCatalog.mat",
                  "book.mat",
                  "citeseer.mat",
                  "computers.mat",
                  "cora.mat",
                  "cs.mat",
                  "Disney.mat",
                  "photo.mat",
                  "weibo.mat"]
MEDIUM_DATASETS = ["Flickr.mat",
                  "Reddit.mat"]

TO_EMD_DATASETS = SMALL_DATASETS

CURRENT_DATASETS = MEDIUM_DATASETS

LEARNING_RATE = 0.001
HIDDEN_DIMS = 16

SEED = 42