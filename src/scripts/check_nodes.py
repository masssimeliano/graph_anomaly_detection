import train_reconstruction
import read_and_show_metrics
from src.helpers.config import CURRENT_DATASETS, CURRENT_DATASETS_SIZE, labels_dict, graph_dict, DataSetSize
from src.helpers.loaders.mat_loader import load_graph_from_mat

SET = ["Disney.mat"]

def main():
    for i, dataset in enumerate(SET):
        print(f"Preparing {dataset}...")
        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.SMALL)
        labels_dict[dataset] = labels
        graph_dict[dataset] = graph

    train_reconstruction.main()

    read_and_show_metrics.main_auc_roc()
    read_and_show_metrics.main_loss()

if __name__ == "__main__":
    main()