import scipy.io
from src.structure.graph import Graph
from src.structure.node import Node
from src.structure.data_set import DataSet, DataSetSize


def load_graph_from_mat(name="Disney.mat", size=DataSetSize.SMALL):
    print("Reading from .mat file")

    dataset = DataSet(name, size)
    data = scipy.io.loadmat(dataset.location)

    adj_matrix = data["Network"].toarray()
    attributes = data["Attributes"].toarray()
    labels = data["Label"].flatten()
    is_str_anomaly = data["str_anomaly_label"].flatten()
    is_attr_anomaly = data["attr_anomaly_label"].flatten()

    graph = Graph([])

    for i in range(len(adj_matrix)):
        node = Node(
            id=i,
            label=labels[i],
            neighbours=[],
            features=attributes[i].tolist(),
            is_str_anomaly=is_str_anomaly[i] != 0,
            is_attr_anomaly=is_attr_anomaly[i] != 0,
        )
        graph.nodes.append(node)

    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] != 0:
                graph.nodes[i].neighbours.append(graph.nodes[j])
                graph.nodes[j].neighbours.append(graph.nodes[i])

    return labels.tolist(), graph
