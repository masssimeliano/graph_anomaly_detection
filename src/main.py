import networkx as nx
import torch
from src.helpers.data_loader import load_graph_from_mat
from src.helpers.graph_plotter import to_networkx_graph
from src.models.unsupervised.anomalydae import (
    baseline_anomalydae_model as baseline,
    structure_and_feature_model as structure_and_feature
)

DATASETS = ["Disney.mat", "book.mat", "citeseer.mat"]

def train_on_diffent_datasets(nx_graph: nx.Graph, labels):
    baseline.train(nx_graph, labels)
    structure_and_feature.train(nx_graph, labels)

    baseline.train(nx_graph, labels)
    structure_and_feature.train(nx_graph, labels)


def main():

    for dataset in DATASETS:
        print("---------------------------------------")
        print("Working with " + dataset)
        print("---------------------------------------\n")

        labels, graph = load_graph_from_mat(dataset)
        nx_graph = to_networkx_graph(graph)

        baseline.train(nx_graph, labels, True, dataset)
        structure_and_feature.train(nx_graph, labels, True, dataset)


if __name__ == "__main__":
    main()
