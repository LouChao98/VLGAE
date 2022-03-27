from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from src.model.embedding import Embedding
from src.model.text_encoder.base import EncoderBase
from src.model.nn import SharedDropout
from src.utility.config import Config
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool

_warn, _info, _debug = get_logger_func('encoder')


@dataclass
class BlankEncoderConfig(Config):
    dropout: float
    shared_dropout: float


class BlankEncoder(EncoderBase):

    def __init__(self, embedding: Embedding, **cfg):
        super().__init__(embedding)
        self.cfg = cfg = BlankEncoderConfig.build(cfg)
        self.output_size = embedding.embed_size
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.shared_dropout = SharedDropout(cfg.dropout) if cfg.shared_dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, vp: VarPool, hiddens=None):
        x = self.dropout(x)
        x = self.shared_dropout(x)
        return {'x': x}

    def get_dim(self, field):
        return self.output_size
