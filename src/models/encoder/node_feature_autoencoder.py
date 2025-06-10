"""
structure_and_feature_model_3.py
This file contains train wrapper for the model "Attr + Str3".
It also contains structural attribute extraction method.
"""

import torch.nn as nn
import torch.nn.functional as F


# simple autoencoder
class NodeFeatureAutoencoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, in_dim)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        return self.decoder(z)
