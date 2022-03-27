from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from src.model.embedding import Embedding
    from src.model import ModelBase


class EncoderBase(nn.Module):
    bounded_embedding: Embedding
    bounded_model: ModelBase

    def __init__(self, embedding: Embedding):
        super().__init__()
        self.__dict__['bounded_embedding'] = embedding

    def forward(self, x, ctx):
        raise NotImplementedError

    def get_dim(self, field):
        raise NotImplementedError(f'Unrecognized {field=}')

