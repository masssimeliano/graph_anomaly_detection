import logging

import read_and_show_metrics
from src.helpers.config.datasets_config import *
from src.helpers.config.training_config import *
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.scripts.dev import train_baseline
from src.scripts.dev import train_from_emd_baseline_with_alpha_1
from src.scripts.dev import train_from_emd_baseline_with_alpha_2
from src.scripts.dev import train_and_save_emd_from_baseline_alpha_1
from src.scripts.dev import train_and_save_emd_from_baseline_alpha_2
from src.scripts.dev import train_structure_and_feature
from src.scripts.dev import train_structure_and_feature_2
from src.scripts.dev import train_structure_and_feature_3
from src.scripts.dev import train_reconstruction_1
from src.scripts.dev import train_reconstruction_2

logging.basicConfig(level=logging.INFO)


def main():
    for i, dataset in enumerate(iterable=CURRENT_DATASETS):
        logging.info(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset,
                                            size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        graph_dict[dataset] = graph

    # train_baseline.main()
    # train_and_save_emd_from_baseline_alpha_1.main()
    # train_and_save_emd_from_baseline_alpha_2.main()
    # train_from_emd_baseline_with_alpha_1.main()
    # train_from_emd_baseline_with_alpha_2.main()
    train_structure_and_feature.main()
    train_structure_and_feature_2.main()
    train_structure_and_feature_3.main()
    train_reconstruction_1.main()
    train_reconstruction_2.main()

    read_and_show_metrics.plot_time()


if __name__ == "__main__":
    main()
