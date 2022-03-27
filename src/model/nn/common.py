import torch.nn as nn
from torch import Tensor

from src.model.nn.dropout import SharedDropout


class ResLayer(nn.Module):
    def __init__(self, n_in, n_hidden, activate=True):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.n_out = n_hidden
        self.activation = nn.LeakyReLU() if activate else nn.Identity()

    def forward(self, x):
        return self.activation(self.linear(x)) + x


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, activate=True):
        super(MLP, self).__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU() if activate else nn.Identity()
        self.dropout = SharedDropout(p=dropout) if dropout > 0 else nn.Identity()
        self.n_out = n_hidden
        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_hidden}"
        if isinstance(self.dropout, SharedDropout):
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
