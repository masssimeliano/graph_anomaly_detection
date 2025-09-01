"""
const.py
This file contains:
- constants used for logging, file naming, and labeling in machine learning tasks.
"""

FEATURE_LABEL_STR2 = "Attr + Str2"
FEATURE_LABEL_STR3 = "Attr + Str3"
FEATURE_LABEL_STR = "Attr + Str"
FEATURE_LABEL_ALPHA1 = "Attr + Alpha1"
FEATURE_LABEL_ALPHA2 = "Attr + Alpha2"
FEATURE_LABEL_EMD1 = "Attr + Emd1"
FEATURE_LABEL_EMD2 = "Attr + Emd2"
FEATURE_LABEL_ERROR1 = "Attr + Error1"
FEATURE_LABEL_ERROR2 = "Attr + Error2"
FEATURE_LABEL_STANDARD = "Attr"

# order is important here because in parse_logs() it must search for all feature labels through contains()
# and before FEATURE_LABEL_STANDARD it has to check FEATURE_LABEL_STR2 e.g.
FEATURE_LABELS = [
    FEATURE_LABEL_STR2,
    FEATURE_LABEL_STR3,
    FEATURE_LABEL_STR,
    FEATURE_LABEL_ALPHA1,
    FEATURE_LABEL_ALPHA2,
    FEATURE_LABEL_EMD1,
    FEATURE_LABEL_EMD2,
    FEATURE_LABEL_ERROR1,
    FEATURE_LABEL_ERROR2,
    FEATURE_LABEL_STANDARD,
]

DICT_FILE_NAME = "file_name"
DICT_DATASET = "dataset"
DICT_FEATURE_LABEL = "feature_label"
DICT_LR = "lr"
DICT_EPOCH = "epoch"
VALUE_EPOCH = "Epochs"
DICT_HID_DIM = "hid_dim"
DICT_AUC_ROC = "auc_roc"
VALUE_AUC_ROC = "AUC-ROC"
DICT_LOSS = "loss"
VALUE_LOSS = "Loss"
DICT_RECALL = "recall"
VALUE_RECALL = "Recall"
DICT_PRECISION = "precision"
VALUE_PRECISION = "Precision"
DICT_TIME = "time"
VALUE_TIME = "Time"
DICT_MODEL = "model"
