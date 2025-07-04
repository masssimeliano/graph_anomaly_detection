# -*- coding: utf-8 -*-
""" Structural Clustering Algorithm for Networks
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import math
import time
import warnings

import numpy as np
import torch

from . import Detector
from ..utils import logger


class SCAN(Detector):
    """
    Structural Clustering Algorithm for Networks

    SCAN is a clustering algorithm, which only takes the graph structure
    without the node features as the input. Note: This model will output
    detected clusters instead of "outliers" descibed in the original
    paper.

    .. note::
        This detector is transductive only. Using ``predict`` with
        unseen data will train the detector from scratch.

    See :cite:`xu2007scan` for details.

    Parameters
    ----------
    eps : float, optional
        Neighborhood threshold. Default: ``.5``.
    mu : int, optional
        Minimal size of clusters. Default: ``2``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.

    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    hub_score_ : torch.Tensor
        The binary hub scores of each node.
    scatter_score_ : torch.Tensor
        The binary scatter scores of each node, i.e., the "outlier"
        scores in the original paper.
    """

    def __init__(self,
                 eps=.5,
                 mu=2,
                 contamination=0.1,
                 verbose=0):
        super(SCAN, self).__init__(contamination=contamination,
                                   verbose=verbose)

        # model param
        self.eps = eps
        self.mu = mu

        self.hub_score_ = None
        self.scatter_score_ = None

    def process_graph(self, data):
        pass

    def fit(self, data, label=None):
        print("scan")

        c = 0
        self.edge_index = data.edge_index
        clusters = torch.zeros(data.x.shape[0])
        self.hub_score_ = torch.zeros(data.x.shape[0])
        self.scatter_score_ = torch.zeros(data.x.shape[0])
        non_member = []

        start_time = time.time()
        for n in range(data.num_nodes):
            if clusters[n]:
                continue
            else:
                queue = self._neighborhood(n).tolist()
                if len(queue) > self.mu:
                    c = c + 1
                    clusters[n] = c
                    while len(queue) != 0:
                        w = queue.pop(0)
                        r = self._neighborhood(w).tolist()
                        r.append(w)
                        for s in r:
                            if not clusters[s] or s in non_member:
                                clusters[s] = c
                            if not clusters[s]:
                                queue.append(s)
                else:
                    non_member.append(n)

        score = clusters.bool().float()

        if_hub = np.vectorize(self._if_hub)(non_member)
        self.hub_score_[non_member] = torch.Tensor(if_hub)
        self.scatter_score_[non_member] = torch.Tensor(1 - if_hub)

        logger(score=score,
               target=label,
               time=time.time() - start_time,
               verbose=self.verbose,
               deep=False)

        self.decision_score_ = score
        self._process_decision_score()
        return self

    def _similarity(self, u, v):
        u_set = torch.unique(self._neighbors(u))
        v_set = torch.unique(self._neighbors(v))
        inter = np.intersect1d(v_set, u_set)
        if len(inter) == 0:
            return 0
        # need to account for vertex itself, add 2(1 for each vertex)
        sim = (len(inter) + 2) / (
            math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
        return sim

    def _neighborhood(self, v):
        candidates = self._neighbors(v)
        if len(candidates) == 0:
            return torch.empty(0)
        sim = np.vectorize(self._similarity)(candidates, v)
        return candidates[sim > self.eps]

    def _neighbors(self, v):
        return self.edge_index[1][self.edge_index[0] == v]

    def _if_hub(self, v):
        neighbors = self._neighbors(v)
        return len(torch.unique(neighbors)) > 1

    def decision_function(self, data, label=None):
        if data is not None:
            warnings.warn("This detector is transductive only. "
                          "Training from scratch with the input data.")
            self.fit(data, label)
        return self.decision_score_
