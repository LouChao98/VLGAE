from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from src.model import ModelBase


class VisEncoderBase(nn.Module):
    bounded_model: ModelBase

    def __init__(self):
        super(VisEncoderBase, self).__init__()

    def forward(self, x, ctx):
        raise NotImplementedError

    def get_dim(self, field):
        raise NotImplementedError(f'Unrecognized {field=}')

