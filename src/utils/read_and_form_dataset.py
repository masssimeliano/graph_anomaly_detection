import scipy.io

from src.structure.graph import Graph
from src.structure.node import Node

DATASET_LOCATION = 'datasets/Disney.mat'

def form():
    data = scipy.io.loadmat(DATASET_LOCATION)

    adj_matrix = data.get('Network').toarray()
    attributes = data.get('Attributes').toarray()
    labels = data.get('Label').flatten()
    is_str_anomaly = data.get('str_anomaly_label').flatten()
    is_attr_anomaly = data.get('attr_anomaly_label').flatten()

    # print(is_str_anomaly)
    # print(labels)
    # print(is_attr_anomaly)

    graph = Graph(list())
    for i in range(0, adj_matrix.shape[0]):
        node = Node(i,
                    labels[i],
                    list(),
                    attributes[i].tolist(),
                    is_str_anomaly[i] != 0,
                    is_attr_anomaly[i] != 0)
        graph.nodes.append(node)

    for i in range(0, adj_matrix.shape[0]):
        for j in range(i, adj_matrix.shape[1]):
            if adj_matrix[i][j] != 0:
                graph.nodes[i].neighbours.append(graph.nodes[j])
                graph.nodes[j].neighbours.append(graph.nodes[i])

    # print(len(graph.nodes))
    # print(graph)
    # print(graph.nodes[0])

    return labels, graph

