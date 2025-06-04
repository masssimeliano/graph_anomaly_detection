import logging
from typing import List, Tuple

import numpy as np
import scipy.io

from src.helpers.config.datasets_config import *
from src.helpers.time.timed import timed
from src.structure.data_set import DataSet
from src.structure.graph import Graph
from src.structure.node import Node

logging.basicConfig(level=logging.INFO)


def build_nodes(adj_matrix: np.ndarray,
                attributes: np.ndarray,
                labels: np.ndarray,
                is_str_anomaly: np.ndarray,
                is_attr_anomaly: np.ndarray) -> List[Node]:
    return [
        Node(
            id=i,
            label=labels[i],
            neighbours=[],
            features=attributes[i].tolist(),
            is_str_anomaly=bool(is_str_anomaly[i]),
            is_attr_anomaly=bool(is_attr_anomaly[i]))

        for i in range(len(adj_matrix))]


def build_edges(nodes: List[Node],
                adj_matrix:
                np.ndarray):
    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] != 0:
                nodes[i].neighbours.append(nodes[j])
                nodes[j].neighbours.append(nodes[i])


@timed
def load_graph_from_mat(name: str,
                        size: DataSetSize) -> Tuple[List[int], Graph]:
    logging.info("Loading and building a graph...")

    dataset = DataSet(name, size)
    data = scipy.io.loadmat(dataset.location)

    adj_matrix = data["Network"].toarray()
    attributes = data["Attributes"].toarray()
    labels = data["Label"].flatten()
    is_str_anomaly = data["str_anomaly_label"].flatten()
    is_attr_anomaly = data["attr_anomaly_label"].flatten()

    nodes = build_nodes(adj_matrix=adj_matrix,
                        attributes=attributes,
                        labels=labels,
                        is_str_anomaly=is_str_anomaly,
                        is_attr_anomaly=is_attr_anomaly)

    only_str = np.sum((is_str_anomaly == 1) & (is_attr_anomaly == 0))
    only_attr = np.sum((is_str_anomaly == 0) & (is_attr_anomaly == 1))
    both = np.sum((is_str_anomaly == 1) & (is_attr_anomaly == 1))

    logging.info(f"Anomalies: Str = {only_str}, Attr = {only_attr}, Str&Attr = {both}")

    build_edges(nodes=nodes,
                adj_matrix=adj_matrix)

    return labels.tolist(), Graph(nodes)
