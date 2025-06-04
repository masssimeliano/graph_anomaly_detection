# order is important here because in parse_logs() it must search for all feature labels through contains()
FEATURE_LABELS = ["Attr + Str2", "Attr + Str3", "Attr + Str", "Attr + Alpha1", "Attr + Alpha2",
                  "Attr + Emd1", "Attr + Emd2", "Attr + Error1", "Attr + Error2"]

FEATURE_LABEL_STANDARD = "Attr"

DICT_FILE_NAME = "file_name"
DICT_DATASET = "dataset"
DICT_FEATURE_LABEL = "feature_label"
DICT_LR = "lr"
DICT_EPOCH = "epoch"
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
