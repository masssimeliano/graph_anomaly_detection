"""
datasets_config.py
This file contains constants used for datasets.
"""

from enum import Enum


class DataSetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


SMALL_DATASETS = [
    "Disney.mat",
    "book.mat",
    "BlogCatalog.mat",
    "citeseer.mat",
    "computers.mat",
    "cora.mat",
    "cs.mat",
    "photo.mat",
    "weibo.mat",
]
MEDIUM_DATASETS = ["Flickr.mat", "Reddit.mat"]
LARGE_DATASETS = []

CHECK_DATASETS = ["Disney.mat"]

CURRENT_DATASETS = CHECK_DATASETS
CURRENT_DATASETS_SIZE = [DataSetSize.SMALL] * len(CHECK_DATASETS)
