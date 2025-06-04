import logging
from typing import List

import networkx as nx

from src.helpers.config.const import FEATURE_LABEL_ERROR2
from src.models.anomalydae.reconstruction_error_model_1 import normalize_node_features_via_minmax_and_remove_nan
from src.models.reconstruction_train import reconstruction_train

logging.basicConfig(level=logging.INFO)


def train(nx_graph: nx.Graph,
          labels: List[int],
          learning_rate: float,
          hid_dim: int,
          dataset: str):
    normalize_node_features_via_minmax_and_remove_nan(nx_graph)
    reconstruction_train(nx_graph=nx_graph,
                         labels=labels,
                         title_prefix=FEATURE_LABEL_ERROR2,
                         learning_rate=learning_rate,
                         hid_dim=hid_dim,
                         dataset=dataset)
