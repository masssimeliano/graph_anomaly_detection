EPOCH_TO_LEARN = 100
EPOCHS = list(range(10, 101, 10))

LEARNING_RATE = 0.005
HIDDEN_DIMS = 16
ETA = 5
THETA = 40
ALPHA = 0.5

SEED = 42

AUC_ROC_PAPER = {
    "cora": 0.762,
    "citeseer": 0.727,
    "BlogCatalog": 0.783,
    "weibo": 0.915,
    "Flickr": 0.751,
    "Reddit": 0.557
}

labels_dict = {}
graph_dict = {}
