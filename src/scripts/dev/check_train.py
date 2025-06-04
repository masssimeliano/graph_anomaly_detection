import logging
import os
import sys
import time

import read_and_show_metrics
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.scripts.dev import train_baseline
from src.scripts.dev.tg_bot_sender import TelegramLogger, bot, CHAT_ID

logging.basicConfig(level=logging.INFO)


def main():
    sys.stdout = TelegramLogger()

    for i, dataset in enumerate(CURRENT_DATASETS):
        logging.info(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset,
                                            size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        graph_dict[dataset] = graph

    train_baseline.main()

    read_and_show_metrics.main_time()

    time.sleep(2)
    for file_name in os.listdir('.'):
        if file_name.endswith('.png'):
            with open(file_name, 'rb') as f:
                bot.send_document(CHAT_ID, f)
                time.sleep(1)


if __name__ == "__main__":
    main()
