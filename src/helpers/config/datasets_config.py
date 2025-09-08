"""
datasets_config.py
This config file contains:
- datasets names and their sizes with corresponding enum class
- names of current classes that are used to train models.
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
MEDIUM_DATASETS = ["Flickr.mat", "Reddit.mat", "tolokers.mat"]
LARGE_DATASETS = []

ONE_CLASS_DATASETS = ["tolokers.mat"]

ALL_DATASETS = SMALL_DATASETS + MEDIUM_DATASETS + LARGE_DATASETS
ALL_DATASETS_SIZE = (
        [DataSetSize.SMALL] * len(SMALL_DATASETS)
        + [DataSetSize.MEDIUM] * len(MEDIUM_DATASETS)
        + [DataSetSize.LARGE] * len(LARGE_DATASETS)
)

CURRENT_DATASETS = ["Disney.mat"]
CURRENT_DATASETS_SIZE = ALL_DATASETS_SIZE
