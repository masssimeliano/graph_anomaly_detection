import scipy.io
from src.structure.graph import Graph
from src.structure.node import Node
from src.structure.data_set import DataSet, DataSetSize


def build_nodes(adj_matrix, attributes, labels, is_str_anomaly, is_attr_anomaly):
    print("Building nodes")

    nodes = []
    for i in range(len(adj_matrix)):
        node = Node(
            id=i,
            label=labels[i],
            neighbours=[],
            features=attributes[i].tolist(),
            is_str_anomaly=is_str_anomaly[i] != 0,
            is_attr_anomaly=is_attr_anomaly[i] != 0,
        )
        nodes.append(node)
    return nodes


def build_edges(nodes, adj_matrix):
    print("Building edges")

    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] != 0:
                nodes[i].neighbours.append(nodes[j])
                nodes[j].neighbours.append(nodes[i])


def load_graph_from_mat(name: str = "Disney.mat", size: DataSetSize = DataSetSize.SMALL) -> tuple[list[int], Graph]:
    print("Reading from .mat file")

    dataset = DataSet(name, size)
    data = scipy.io.loadmat(dataset.location)

    adj_matrix = data["Network"].toarray()
    attributes = data["Attributes"].toarray()
    labels = data["Label"].flatten()
    is_str_anomaly = data["str_anomaly_label"].flatten()
    is_attr_anomaly = data["attr_anomaly_label"].flatten()

    nodes = build_nodes(adj_matrix, attributes, labels, is_str_anomaly, is_attr_anomaly)
    build_edges(nodes, adj_matrix)

    return labels.tolist(), Graph(nodes)