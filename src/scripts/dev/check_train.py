import os
import sys
import time

import read_and_show_metrics
from src.helpers.config import CURRENT_DATASETS, CURRENT_DATASETS_SIZE, labels_dict, graph_dict
from src.helpers.loaders.mat_loader import load_graph_from_mat
import train_baseline, train_reconstruction_1, train_reconstruction_2, train_structure_and_feature, train_structure_and_feature_2, train_structure_and_feature_3
import train_from_emd_baseline_with_alpha_1, train_from_emd_baseline_with_alpha_2, train_and_save_emd_from_baseline_alpha_1, train_and_save_emd_from_baseline_alpha_2
from src.scripts.dev.tg_bot import TelegramLogger, bot, CHAT_ID


def main():
    sys.stdout = TelegramLogger()

    for i, dataset in enumerate(CURRENT_DATASETS):
        print(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset,
                                            size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        graph_dict[dataset] = graph

    train_baseline.main()
    train_structure_and_feature.main()
    train_structure_and_feature_2.main()
    train_structure_and_feature_3.main()
    train_reconstruction_1.main()
    train_reconstruction_2.main()
    train_and_save_emd_from_baseline_alpha_1.main()
    train_and_save_emd_from_baseline_alpha_2.main()
    train_from_emd_baseline_with_alpha_1.main()
    train_from_emd_baseline_with_alpha_2.main()

    read_and_show_metrics.main_loss()
    read_and_show_metrics.main_auc_roc()
    read_and_show_metrics.main_recall()
    read_and_show_metrics.main_precision()
    read_and_show_metrics.main_time()

    time.sleep(2)
    for file_name in os.listdir('.'):
        if file_name.endswith('.png'):
            with open(file_name, 'rb') as f:
                bot.send_document(CHAT_ID, f)
                time.sleep(1)



if __name__ == "__main__":
    main()