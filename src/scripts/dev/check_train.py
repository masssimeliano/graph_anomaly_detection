import train_reconstruction
import read_and_show_metrics
from src.helpers.config import CURRENT_DATASETS, CURRENT_DATASETS_SIZE, labels_dict, graph_dict
from src.helpers.loaders.mat_loader import load_graph_from_mat
import train_baseline
import train_from_emd_baseline_with_alpha_1, train_from_emd_baseline_with_alpha_2


def main():
    for i, dataset in enumerate(CURRENT_DATASETS):
        print(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset,
                                            size=CURRENT_DATASETS_SIZE[i])
        labels_dict[dataset] = labels
        graph_dict[dataset] = graph

    train_baseline.main()
    train_from_emd_baseline_with_alpha_1.main()
    train_from_emd_baseline_with_alpha_2.main()

    read_and_show_metrics.main_loss()
    read_and_show_metrics.main_auc_roc()
    read_and_show_metrics.main_recall()
    read_and_show_metrics.main_precision()

if __name__ == "__main__":
    main()