"""
trainig_config.py
This file contains constants for hyperparameters used in machine learning models.
"""

EPOCH_TO_LEARN = 100
EPOCHS = list(range(10, 101, 10))

# hyperparameters from benchmark
LEARNING_RATE = 0.005
HIDDEN_DIMS = 16
ETA = 5
THETA = 40
ALPHA = 0.5

# custom seed
SEED = 42

# AUC-ROC from benchmark
AUC_ROC_PAPER = {
    "cora": 0.762,
    "citeseer": 0.727,
    "BlogCatalog": 0.783,
    "weibo": 0.915,
    "Flickr": 0.751,
    "Reddit": 0.557,
}

labels_dict = {}
graph_dict = {}
