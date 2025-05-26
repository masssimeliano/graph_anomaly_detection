from src.helpers.config import CURRENT_DATASETS, MEDIUM_DATASETS
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph
from src.structure.data_set import DataSetSize


def main():
    for dataset in MEDIUM_DATASETS:
        labels, graph = load_graph_from_mat(name=dataset, size=DataSetSize.MEDIUM)
        to_networkx_graph(graph=graph, visualize=True, title=dataset)

if __name__ == "__main__":
    main()