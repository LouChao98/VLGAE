from typing import Union

import torch
import torch.nn as nn
from fastNLP.embeddings import StaticEmbedding, TokenEmbedding, CNNCharEmbedding, LSTMCharEmbedding
from torch import Tensor
from torch.nn import Parameter

from src.model.embedding.embedding import EmbeddingAdaptor
from src.model.nn.multivariate_kl import MultVariateKLD
from src.utility.var_pool import VarPool


class FastNLPEmbeddingAdaptor(EmbeddingAdaptor):

    def __init__(self, emb: TokenEmbedding):
        super().__init__(emb)
        self._embed_size = self.emb.embed_size
        self._word_dropout = emb.word_dropout
        self._dropout = emb.dropout_layer.p
        self._normalize_weight = None

    @property
    def embed_size(self):
        return self._embed_size

    def forward(self, field: Tensor, vp: VarPool):
        return self.emb(field)

    def normalize(self, method):
        emb: torch.nn.Embedding = self.emb.embedding
        if hasattr(self.emb, 'mapped_counts'):
            self.emb: StaticEmbedding
            if self._normalize_weight is None:
                self._normalize_weight = (self.emb.mapped_counts / self.emb.mapped_counts.sum()).unsqueeze(-1)
            mean = (emb.weight.data * self._normalize_weight).sum()
            if method == 'mean':
                emb.weight.data.sub_(mean)
            else:
                std = (((emb.weight.data - mean).pow(2.) * self._normalize_weight).sum() + 1e-6).sqrt()
                if method == 'mean+std':
                    emb.weight.data.sub_(mean)
                emb.weight.data.div_(std)
        else:
            padding_idx = self.emb.get_word_vocab().padding_idx
            start_idx = 1 if padding_idx == 0 else 0
            self._normalize(emb.weight.data[start_idx:], method)

class FastNLPEmbeddingVariationalAdaptor(FastNLPEmbeddingAdaptor):
    def __init__(self, emb: TokenEmbedding, mode: str, out_dim: int):
        # mode: vae or ib
        super(FastNLPEmbeddingVariationalAdaptor, self).__init__(emb)
        self.mode = mode
        if self.mode != 'basic':
            self._embed_size = out_dim
            self.enc = nn.Linear(emb.embed_size, 2 * out_dim)
            if self.mode == 'ib':
                self.gaussian_kl = MultVariateKLD('sum')
                self.target_mean = Parameter(torch.zeros(1, out_dim))
                self.target_lvar = Parameter(torch.zeros(1, out_dim))

    def forward(self, field: Tensor, vp: VarPool):
        if self.mode == 'basic':
            return super().forward(field, vp)

        mean, lvar = torch.chunk(self.enc(self.emb(field)), 2, dim=-1)
        if self.training:
            z = torch.empty_like(mean).normal_()
            z = (0.5 * lvar).exp() * z + mean
        else:
            z = mean
        vp.kl = self.kl(mean, lvar)
        return z

    def kl(self, mean, lvar):
        if self.mode == 'ib':
            _mean, _lvar = mean.view(-1, self.embed_size), lvar.view(-1, self.embed_size)
            _b = len(_mean)
            return self.gaussian_kl(_mean, self.target_mean.expand(_b, -1), _lvar, self.target_lvar.expand(_b, -1))
        else:
            return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1).sum()


class FastNLPCharEmbeddingAdaptor(FastNLPEmbeddingAdaptor):
    def normalize(self, method):
        self.emb: Union[CNNCharEmbedding, LSTMCharEmbedding]
        emb = self.emb.char_embedding
        start_idx = 1 if self.emb.char_pad_index == 0 else 0
        self._normalize(emb.weight.data[start_idx:], method)
