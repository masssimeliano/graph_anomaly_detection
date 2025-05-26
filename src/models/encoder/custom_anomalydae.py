import torch
from pygod.detector.anomalydae import AnomalyDAE
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

from src.helpers.config import EPOCHS, RESULTS_DIR


class CustomAnomalyDAE(AnomalyDAE):
    def __init__(self,
                 labels,
                 title_prefix,
                 data_set,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.array_loss = []
        self.array_auc_roc = []
        self.amount_of_epochs = max(EPOCHS)
        self.labels = labels
        self.title_prefix = title_prefix
        self.data_set = data_set
        self.loss_last = 0
        self.save_emb = True

    def fit(self, data, label=None):
        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_in = torch.optim.Adam(self.model.inner.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.outer.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.amount_of_epochs + 1):
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                loss, score = self.forward_model(sampled_data)
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.model.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.model.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # saving the loss value on the last epoch
            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
            self.loss_last = loss_value

            # calculating AUC-ROC through all epochs
            if (epoch in EPOCHS):
                self.array_loss.append(loss_value)
                auc_roc = roc_auc_score(self.labels, self.decision_score_)
                self.array_auc_roc.append(auc_roc)
                # saving embedding if its needed
                if (self.save_emb):
                    emd_file = self.get_emd_file(epoch)
                    torch.save(self.emb, emd_file)

        self._process_decision_score()
        return self

    def fit_emd(self, data, label=None):

        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_in = torch.optim.Adam(self.model.inner.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.outer.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id

                loss, score = self.forward_model(sampled_data)
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.model.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.model.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # saving the loss value on the last epoch
            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
            self.loss_last = loss_value

            # saving embedding if its needed
            if (epoch in EPOCHS):
                if (self.save_emb):
                    emd_file = self.get_emd_file(epoch)
                    torch.save(self.emb, emd_file)

        self._process_decision_score()
        return self

    def get_emd_file(self,
        current_epoch: int):
        return RESULTS_DIR / f"emd_{self.data_set}_{self.title_prefix}_{str(self.lr).replace('.', '')}_{self.hid_dim}_{current_epoch}.pt"
