from enum import Enum


class DataSetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


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
LARGE_DATASETS = []

SMALL_DATASETS_2 = ["Disney",
                    "book",
                    "BlogCatalog",
                    "citeseer"]

DATASETS = ["cora.mat",
            "cs.mat",
            "photo.mat",
            "weibo.mat"]

CHECK_DATASETS = ["Disney.mat"]

CURRENT_DATASETS = DATASETS
CURRENT_DATASETS_SIZE = [DataSetSize.SMALL] * len(DATASETS)  # '+ [DataSetSize.MEDIUM] * len(MEDIUM_DATASETS)
