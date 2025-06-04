from src.helpers.config.datasets_config import *
from src.helpers.loaders.mat_loader import load_graph_from_mat
from src.helpers.plotters.nx_graph_plotter import to_networkx_graph


# visualize datasets
def main():
    for i, dataset in enumerate(CURRENT_DATASETS):
        labels, graph = load_graph_from_mat(name=dataset,
                                            size=CURRENT_DATASETS_SIZE[i])
        to_networkx_graph(graph=graph,
                          visualize=True,
                          title=dataset)


if __name__ == "__main__":
    main()
