from pathlib import Path


log_dir =  Path(__file__).resolve().parents[2] / "results" / "unsupervised" / "anomalyedae"
results = []

def open_logs():
    print("Opening logs")

    for file in log_dir.glob("*.txt"):
        with open(file, "r") as f:
            lines = f.readlines()

            if len(lines) < 3:
                print("File is too short")

            config_line = lines[0].strip()
            auc_line = lines[1].strip()

            try:
                lr = float(config_line.split("lr=")[1].split(",")[0])
                hid_dim = int(config_line.split("hid_dim=")[1].split(")")[0])
                epoch = int(config_line.split("epoch=")[1].split(",")[0])
            except Exception as e:
                print("File error 1 : ", e)
                continue

            try:
                auc_roc = float(auc_line.split("AUC-ROC")[1].split(":")[1].strip())
            except Exception as e:
                print("File error 2 : ", e)
                continue

            results.append({
                "filename": file.name,
                "lr": lr,
                "epoch": epoch,
                "hid_dim": hid_dim,
                "auc_roc": auc_roc
            })


def sort_logs():
    target_lr_1 = 0.0005
    target_lr_2 = 0.001
    target_lr_3 = 0.01

    filtered_by_lr_1 = [r for r in results if r["lr"] == target_lr_1]
    filtered_by_lr_2 = [r for r in results if r["lr"] == target_lr_2]
    filtered_by_lr_3 = [r for r in results if r["lr"] == target_lr_3]

    target_hid_dim_1 = 16
    target_hid_dim_2 = 32
    target_hid_dim_3 = 64

    filtered_by_hid_dim_1 = [r for r in results if r["hid_dim"] == target_hid_dim_1]
    filtered_by_hid_dim_2 = [r for r in results if r["hid_dim"] == target_hid_dim_2]
    filtered_by_hid_dim_3 = [r for r in results if r["hid_dim"] == target_hid_dim_3]

    best_model_lr_1 = max(filtered_by_lr_1, key=lambda x: x["auc_roc"])
    best_model_lr_2 = max(filtered_by_lr_2, key=lambda x: x["auc_roc"])
    best_model_lr_3 = max(filtered_by_lr_3, key=lambda x: x["auc_roc"])
    best_model_hid_dim_1 = max(filtered_by_hid_dim_1, key=lambda x: x["auc_roc"])
    best_model_hid_dim_2 = max(filtered_by_hid_dim_2, key=lambda x: x["auc_roc"])
    best_model_hid_dim_3 = max(filtered_by_hid_dim_3, key=lambda x: x["auc_roc"])
    best_model = max(results, key=lambda x: x["auc_roc"])
    worst_model = min(results, key=lambda x: x["auc_roc"])

    print("Best model with lr=0.0005 : ", best_model_lr_1)
    print("Best model with lr=0.001 : ", best_model_lr_2)
    print("Best model with lr=0.01 : ", best_model_lr_3)
    print("Best model with hid_dim=16 : ", best_model_hid_dim_1)
    print("Best model with hid_dim=32 : ", best_model_hid_dim_2)
    print("Best model with hid_dim=64 : ", best_model_hid_dim_3)
    print("Best model : ", best_model)
    print("Worst model : ", worst_model)

