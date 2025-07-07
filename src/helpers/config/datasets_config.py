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

CHECK_DATASETS_TRAIN_SCRIPT = ["Disney.mat"]
CHECK_DATASETS = ["Disney"]
CHECK_DATASETS_2 = [
    "book",
    "BlogCatalog",
    "citeseer",
    "computers",
    "cora",
]

CURRENT_DATASETS = ["Enron.mat", "questions.mat", "tolokers.mat"]
CURRENT_DATASETS_SIZE = [DataSetSize.MEDIUM] * len(CURRENT_DATASETS)
