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
    "Disney.mat", # Disney movies; nodes (movies and their prices and ratings); edges (co-purchased)
    "book.mat", # books; nodes (books and their prices and ratings); edges (co-purchased)
    "BlogCatalog.mat", # Social blogs; nodes (bloggers); edges (friendships among bloggers)
    "citeseer.mat", # Citation network; nodes (document and its bag-of-words); edges (document cite each other)
    "computers.mat", # Amazon co-purchase graph; ...
    "cora.mat", # Citation network; ...
    "cs.mat", # Amazon co-purchase graph; nodes (goods and their bag-of-words); edges (same shopping cart)
    "photo.mat", # Amazon co-purchase graph; ...
    "weibo.mat", # Micro-blogging website weibo; nodes (weibo users); edges (same hashtag)
]
MEDIUM_DATASETS = [
    "Flickr.mat", # Flickr network; nodes (Flickr users and their attributes); edges (social links between users)
    "Reddit.mat", # Reddit posts, nodes (post title, number of comments, score); edges (same user commented on both)
    "tolokers.mat" # Toloka crowdsourcing platform; nodes (represent tolokers(workers) that have participated in at least one of projects); edges (workes on the same task)
]
LARGE_DATASETS = []

ONE_CLASS_DATASETS = [
    "tolokers.mat"
]


ALL_DATASETS = SMALL_DATASETS + MEDIUM_DATASETS + LARGE_DATASETS
ALL_DATASETS_SIZE = (
        [DataSetSize.SMALL] * len(SMALL_DATASETS)
        + [DataSetSize.MEDIUM] * len(MEDIUM_DATASETS)
        + [DataSetSize.LARGE] * len(LARGE_DATASETS)
)

CURRENT_DATASETS = ALL_DATASETS
CURRENT_DATASETS_SIZE = ALL_DATASETS_SIZE
