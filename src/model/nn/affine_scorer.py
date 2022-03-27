from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .affine import Biaffine
from .common import MLP


class BiaffineScorer(nn.Module):
    def __init__(self,
                 n_in,
                 hidden_dim,
                 out_dim,
                 mlp_dropout,
                 mlp_activate,
                 scale):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.mlp1 = MLP(n_in // 2, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(n_in // 2, hidden_dim, mlp_dropout, mlp_activate)
        self.affine = Biaffine(hidden_dim, out_dim, bias_x=True, bias_y=out_dim > 1)
        self.register_buffer('scale', 1 / torch.tensor(hidden_dim if scale else 1).pow(0.25))
        self.n_out = out_dim

    def reset_parameters(self):
        nn.init.zeros_(self.affine.weight)
        self.affine.weight.diagonal().one_()

    def forward(self, x, x2):
        h1 = self.mlp1(x) * self.scale
        h2 = self.mlp2(x2) * self.scale
        out = self.affine(h1, h2).permute(0, 2, 3, 1)
        return out
