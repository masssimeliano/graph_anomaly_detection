import networkx as nx
import torch
from networkx.classes import Graph
from pygod.detector import DOMINANT
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import from_networkx

import src.structure.graph


def train(graph: Graph, labels):
    degree_dict = dict(graph.degree())
    clustering_dict = nx.clustering(graph)
    pagerank_dict = nx.pagerank(graph)
    betweenness_centrality_dict = nx.betweenness_centrality(graph)
    closeness_centrality_dict = nx.closeness_centrality(graph)

    for node in graph.nodes:
        deg = degree_dict[node]
        clust = clustering_dict[node]
        rank = pagerank_dict[node]
        betweenness_centrality = betweenness_centrality_dict[node]
        closeness_centrality = closeness_centrality_dict[node]

        features = graph.nodes[node]['x']
        extra_features = torch.tensor([deg, clust, rank, betweenness_centrality, closeness_centrality], dtype=torch.float)
        graph.nodes[node]['x'] = torch.cat([features, extra_features])

    di_graph = from_networkx(graph)

    model = DOMINANT()
    model.fit(di_graph)

    scores = model.decision_score_

    # print(scores)

    auc = roc_auc_score(labels, scores)

    print(f"AUC-ROC (Structure and feature): {auc:.4f}")