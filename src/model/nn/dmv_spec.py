import torch
import torch.nn as nn
from torch import Tensor


class DMVSkipConnectEncoder(nn.Module):
    def __init__(self, hidden_size, n_bottleneck=0, n_mid=0, dropout=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.n_out = hidden_size

        # To encode valence information
        if n_bottleneck == 0:
            self.HASCHILD_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.NOCHILD_linear = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.HASCHILD_linear = self.create_bottleneck(self.hidden_size, n_bottleneck)
            self.NOCHILD_linear = self.create_bottleneck(self.hidden_size, n_bottleneck)
        self.valence_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # To encode direction information
        if n_bottleneck == 0:
            self.LEFT_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.RIGHT_linear = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.LEFT_linear = self.create_bottleneck(self.hidden_size, n_bottleneck)
            self.RIGHT_linear = self.create_bottleneck(self.hidden_size, n_bottleneck)
        self.direction_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # To produce final hidden representation
        n_mid = n_mid if n_mid else hidden_size
        self.linear1 = nn.Linear(self.hidden_size, n_mid)
        self.linear2 = nn.Linear(n_mid, self.hidden_size)

    def forward(self, x: Tensor):
        # input:    ... x len x hidden1
        # output:   ... x len x dir x val x hidden2
        has_child = self.HASCHILD_linear(x) + x
        no_child = self.NOCHILD_linear(x) + x
        h = torch.cat([no_child.unsqueeze(-2), has_child.unsqueeze(-2)], dim=-2)
        h = self.activate(self.valence_linear(self.activate(h)))

        x = x.unsqueeze(-2)
        left_h = self.LEFT_linear(h) + x
        right_h = self.RIGHT_linear(h) + x
        h = torch.cat([left_h.unsqueeze(-3), right_h.unsqueeze(-3)], dim=-3)
        h = self.activate(self.direction_linear(self.activate(h)))

        h = self.dropout(h)
        return self.linear2(self.activate(self.linear1(h)))

    @staticmethod
    def create_bottleneck(n_in_out, n_bottleneck):
        return nn.Sequential(nn.Linear(n_in_out, n_bottleneck), nn.Linear(n_bottleneck, n_in_out))


class DMVFactorizedBilinear(nn.Module):
    def __init__(self, n_in, n_in2=None, r=64):
        super(DMVFactorizedBilinear, self).__init__()
        self.n_in = n_in
        self.n_in2 = n_in2 if n_in2 else n_in
        self.r = r
        self.project1 = nn.Linear(self.n_in, self.r)
        self.project2 = nn.Linear(self.n_in2, self.r)

    def forward(self, x1, x2):
        x1 = self.project1(x1)
        x2 = self.project2(x2)
        if len(x1.shape) == 5:
            return torch.einsum("bhdve, bcdve -> bhcdv", x1, x2)
        elif len(x1.shape) == 4:
            return torch.einsum("hdve, cdve -> hcdv", x1, x2)
        else:
            raise NotImplementedError
