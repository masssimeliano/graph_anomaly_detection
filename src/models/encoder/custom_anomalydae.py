import torch
from pygod.detector.anomalydae import AnomalyDAE

class CustomAnomalyDAE(AnomalyDAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_last = 0

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

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     weight,
                                     pos_weight_a,
                                     pos_weight_s)

        loss = torch.mean(score)
        self.loss_last = loss.item()

        return loss, score.detach().cpu()
