import torch
import torch.nn as tnn

from src.model.vis_encoder import VisEncoderBase
from src.model.nn import MLP, BiaffineScorer


class VisBoxRelSimpleEncoder(VisEncoderBase):
    def __init__(self, n_in, n_hidden, dropout, activate, use_attr, use_img, img_feat):
        super().__init__()

        self.use_img = use_img
        if use_img:
            self.img_fc = MLP(n_in, n_hidden, dropout, activate)

        self.img_feat = img_feat
        if img_feat:
            n_in *= 2
        self.box_fc = MLP(n_in, n_hidden, dropout, activate)
        self.rel_fc = MLP(n_in, n_hidden, dropout, activate)
        # self.rel_fc = BiaffineScorer(n_in * 2, n_hidden, n_hidden, dropout, activate, 1)

        self.use_attr = use_attr
        if use_attr:
            self.attr_fc = MLP(n_in, n_hidden, dropout, activate)
        self.n_hidden = n_hidden
        self.dropout = dropout

    def forward(self, x, ctx):

        if self.img_feat:
            feat: torch.Tensor = x["vis_box_feat"]
            B, N, H = feat.shape
            box = feat
            inputs = torch.cat(
                [box, feat.mean(1, keepdim=True).expand(-1, feat.shape[1], -1)], dim=-1
            )
        else:
            inputs = x["vis_box_feat"]
            B, N, H = inputs.shape
            inputs = inputs.view(B, N, H)
        _rel_inp = (inputs.unsqueeze(1) + inputs.unsqueeze(2)) / 2
        x_rel = self.rel_fc(_rel_inp)
        # x_rel = self.rel_fc(inputs, inputs)
        rel = x_rel.view(len(x_rel), -1, self.n_hidden)

        out = {"box": self.box_fc(inputs), "rel": rel}
        if self.use_attr:
            out["attr"] = self.attr_fc(inputs)
        if self.use_img:
            out["img"] = self.img_fc(x["vis_box_feat"].mean(1, keepdim=True))
        return out

    def get_dim(self, field):
        return self.n_hidden

