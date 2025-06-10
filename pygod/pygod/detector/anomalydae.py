"""AnomalyDAE: Dual autoencoder for anomaly detection
on attributed networks"""

import time
import warnings

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

from src.helpers.config.training_config import *
from src.helpers.loaders.emd_file_getter import get_emd_file
from . import DeepDetector
from ..metric import eval_precision_at_k, eval_recall_at_k
from ..nn import AnomalyDAEBase


class AnomalyDAE(DeepDetector):

    def __init__(
        self,
        labels,
        title_prefix,
        data_set,
        emb_dim=64,
        hid_dim=64,
        num_layers=4,
        dropout=0.0,
        weight_decay=0.0,
        act=F.relu,
        backbone=None,
        alpha=0.5,
        theta=1.0,
        eta=1.0,
        contamination=0.1,
        lr=0.004,
        epoch=5,
        gpu=-1,
        batch_size=0,
        num_neigh=-1,
        verbose=0,
        save_emb=False,
        compile_model=False,
        **kwargs
    ):

        if backbone is not None or num_layers != 4:
            warnings.warn("Backbone and num_layers are not used in AnomalyDAE")

        super(AnomalyDAE, self).__init__(
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            weight_decay=weight_decay,
            act=act,
            backbone=backbone,
            contamination=contamination,
            lr=lr,
            epoch=epoch,
            gpu=gpu,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            save_emb=save_emb,
            compile_model=compile_model,
            **kwargs
        )

        self.emb_dim = emb_dim
        self.alpha = alpha
        self.theta = theta
        self.eta = eta
        self.title_prefix = title_prefix

        # new variables for quicker learning
        self.array_loss = []
        self.array_auc_roc = []
        self.array_recall_k = []
        self.array_precision_k = []
        self.labels = labels
        self.dataset = data_set
        self.loss_last = 0
        self.last_time = 0
        self.array_time = []
        self.stru_error_mean = []
        self.stru_error_std = []
        self.attr_error_mean = []
        self.attr_error_std = []

        self.save_emb = save_emb

    def process_graph(self, data):
        AnomalyDAEBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)

        return AnomalyDAEBase(
            in_dim=self.in_dim,
            num_nodes=self.num_nodes,
            emb_dim=self.emb_dim,
            hid_dim=self.hid_dim,
            dropout=self.dropout,
            act=self.act,
            **kwargs
        ).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, edge_index, batch_size)

        # positive weight conversion
        weight = 1 - self.alpha
        pos_weight_a = self.eta / (1 + self.eta)
        pos_weight_s = self.theta / (1 + self.theta)

        # custom loss function that contains
        # mean and std of both error arts
        (
            score,
            structural_error_mean,
            structural_error_std,
            attribute_error_mean,
            attribute_error_std,
        ) = self.model.loss_func(
            x[:batch_size],
            x_[:batch_size],
            s[:batch_size, node_idx],
            s_[:batch_size],
            weight,
            pos_weight_a,
            pos_weight_s,
        )

        loss = torch.mean(score)

        return (
            loss,
            score.detach().cpu(),
            structural_error_mean,
            structural_error_std,
            attribute_error_mean,
            attribute_error_std,
        )

    # custom fit() method that works with same epochs value
    # and saves resulting metrics inside for-cycle
    def fit(self, data, label=None):
        start_time = time.time()
        self.array_loss = []
        self.array_auc_roc = []
        self.array_recall_k = []
        self.array_precision_k = []
        self.array_time = []

        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data, self.num_neigh, batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            self.opt_in = torch.optim.Adam(
                self.model.inner.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            optimizer = torch.optim.Adam(
                self.model.outer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch + 1):
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                # structural and attribute reconstruction errors can be used here
                (
                    loss,
                    score,
                    stru_error_mean,
                    stru_error_std,
                    attr_error_mean,
                    attr_error_std,
                ) = self.forward_model(sampled_data)
                self.stru_error_mean = stru_error_mean
                self.stru_error_std = stru_error_std
                self.attr_error_mean = attr_error_mean
                self.attr_error_std = attr_error_std

                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = self.model.emb[0][
                            :batch_size
                        ].cpu()
                        self.emb[1][node_idx[:batch_size]] = self.model.emb[1][
                            :batch_size
                        ].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = self.model.emb[
                            :batch_size
                        ].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # saving loss value on last epoch
                loss_value = epoch_loss / data.x.shape[0]
                if self.gan:
                    loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
                self.loss_last = loss_value
                self.last_time = time.time() - start_time

                # calculating AUC-ROC through all epochs
                if epoch in EPOCHS:
                    self.array_time.append(time.time() - start_time)
                    self.array_loss.append(loss_value)
                    auc_roc = roc_auc_score(self.labels, self.decision_score_)

                    self_labels = self.labels
                    self_score = self.decision_score_
                    self_k = self.labels.count(1)
                    recall_k = eval_recall_at_k(
                        label=self_labels, score=self_score, k=self_k
                    )
                    precision_k = eval_precision_at_k(
                        label=self_labels, score=self_score, k=self_k
                    )
                    self.array_auc_roc.append(auc_roc)
                    self.array_recall_k.append(recall_k)
                    self.array_precision_k.append(precision_k)

                    # saving embedding if its needed
                    if self.save_emb:
                        dataset = self.dataset
                        title_prefix = self.title_prefix
                        learning_rate = self.lr
                        hid_dim = self.hid_dim
                        emd_file = get_emd_file(
                            dataset=dataset,
                            title_prefix=title_prefix,
                            learning_rate=learning_rate,
                            hid_dim=hid_dim,
                            epoch=epoch,
                        )
                        torch.save(obj=self.emb, f=emd_file)

        self._process_decision_score()
        return self

    # custom fit() method that works with same epochs value,
    # but calculates every resulting metric only once
    def fit_emd(self, data):
        start_time = time.time()
        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data, self.num_neigh, batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            self.opt_in = torch.optim.Adam(
                self.model.inner.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            optimizer = torch.optim.Adam(
                self.model.outer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch + 1):
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                # structural and attribute reconstruction errors can be used here
                (
                    loss,
                    score,
                    stru_error_mean,
                    stru_error_std,
                    attr_error_mean,
                    attr_error_std,
                ) = self.forward_model(sampled_data)
                self.stru_error_mean = stru_error_mean
                self.stru_error_std = stru_error_std
                self.attr_error_mean = attr_error_mean
                self.attr_error_std = attr_error_std

                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = self.model.emb[0][
                            :batch_size
                        ].cpu()
                        self.emb[1][node_idx[:batch_size]] = self.model.emb[1][
                            :batch_size
                        ].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = self.model.emb[
                            :batch_size
                        ].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # saving loss value on last epoch
            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
            self.loss_last = loss_value
            self.last_time = time.time() - start_time

        # saving embedding if its needed
        if self.save_emb:
            dataset = self.dataset
            title_prefix = self.title_prefix
            learning_rate = self.lr
            hid_dim = self.hid_dim
            epoch = self.epoch
            emd_file = get_emd_file(
                dataset=dataset,
                title_prefix=title_prefix,
                learning_rate=learning_rate,
                hid_dim=hid_dim,
                epoch=epoch,
            )
            torch.save(obj=self.emb, f=emd_file)

        self._process_decision_score()
        return self
