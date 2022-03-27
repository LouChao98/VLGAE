from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from src.model.nn import IndependentDropout
from src.utility.config import Config
from src.utility.logger import get_logger_func

if TYPE_CHECKING:
    from src.model import ModelBase
    from src.datamodule import DataModule
    from src.utility.var_pool import VarPool

AnyDict = Dict[str, Any]

_warn, _info, _debug = get_logger_func('embedding')


@dataclass
class EmbeddingItem:
    name: str
    field: str
    emb: EmbeddingAdaptor


@dataclass
class EmbeddingConfig(Config):
    use_word: bool
    use_tag: bool
    use_subword: bool  # I believe we need only one subwords field.'
    dropout: 0.  # when multi embedding, for each position, drop some entirely.
    # all other items are treated as EmbeddingItemConfig


@dataclass
class EmbeddingItemConfig(Config):
    args: AnyDict
    adaptor_args: AnyDict
    field: str
    requires_vocab: bool = True  # pass vocab to embedding
    normalize_word: bool = False  # pass the normalize_func(used by datamodule) to Embedding
    normalize_method: str = 'mean+std'  # mean+std, mean, std, none
    normalize_time: str = 'nowhere'  # when to normalize embedding, none, begin, epoch, batch


class Embedding(torch.nn.Module):
    """Embedding, plus apply to different fields."""
    bounded_model: ModelBase

    def __init__(self, dm: DataModule, **cfg):
        super().__init__()
        flags, emb_cfg = EmbeddingConfig.build(cfg, ignore_unknown=True)
        flags: EmbeddingConfig

        vocabs = dm.vocabs
        datasets = dm.datasets

        self.disabled_fields = set()
        if not flags.use_word:
            self.disabled_fields.add('word')
        if not flags.use_subword:
            self.disabled_fields.add('subword')
        if not flags.use_tag:
            self.disabled_fields.add('pos')

        # instantiate embeddings
        self.embeds: List[EmbeddingItem] = []
        self.normalize_dict = {'nowhere': [], 'begin': [], 'epoch': [], 'batch': []}
        for name, cfg in emb_cfg.items():
            if name.startswith('_') or cfg is None:
                continue
            cfg: EmbeddingItemConfig = EmbeddingItemConfig.build(cfg)
            if cfg.field in self.disabled_fields:
                continue
            instantiate_args = {}
            if cfg.requires_vocab:
                instantiate_args['vocab'] = vocabs[cfg.field]
            if cfg.normalize_word:
                instantiate_args['word_transform'] = dm.normalize_one_word_func
            emb = instantiate(cfg.args, **instantiate_args)
            emb = instantiate(cfg.adaptor_args, emb=emb)
            emb.process(vocabs, datasets)
            self.add_module(name, emb)
            self.embeds.append(EmbeddingItem(name, cfg.field, emb))
            self.normalize_dict[cfg.normalize_time].append((name, cfg.normalize_method))

        _info(f'Emb: {", ".join(e.name for e in self.embeds)}')
        _info(f'Normalize plan: {self.normalize_dict}')
        self.embed_size = sum(e.embed_size for e in self)

        if flags.dropout > 0:
            self.dropout_func = IndependentDropout(flags.dropout)
        else:
            self.dropout_func = lambda *x: x

    def forward(self, x, vp: VarPool):
        emb = list(self.dropout_func(*[item.emb(x[item.field], vp) for item in self.embeds]))
        seq_len = max(e.shape[1] for e in emb)
        assert all(e.shape[1] in (1, seq_len) for e in emb)
        for item, h in zip(self.embeds, emb):
            vp[item.name] = h
        for i in range(len(emb)):
            if emb[i].shape[1] == 1:
                emb[i] = emb[i].expand(-1, seq_len, -1)
        # from src.utility.fn import draw_att
        # draw_att(torch.cat(emb, dim=-1)[0])
        return torch.cat(emb, dim=-1)

    def normalize(self, now):
        for name, method in self.normalize_dict[now]:
            getattr(self, name).normalize(method)

    def __getitem__(self, key):
        return self.embeds[key].emb

    def __iter__(self):
        return map(lambda e: e.emb, self.embeds)

    def __len__(self):
        return len(self.embeds)


class EmbeddingAdaptor(nn.Module):
    device_indicator: Tensor
    singleton_emb = {}

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.register_buffer('device_indicator', torch.zeros(1))

        self._normalize_warned = False

    @property
    def embed_size(self):
        raise NotImplementedError

    @property
    def device(self):
        return self.device_indicator.device

    def process(self, vocabs, datasets):
        return

    def forward(self, inputs: List[Any], vp: VarPool):
        raise NotImplementedError

    def normalize(self, method: str):
        if not self._normalize_warned:
            _warn(f"{type(self)} didn't implement normalize.")
            self._normalize_warned = True

    @staticmethod
    def _normalize(data: Tensor, method: str):
        with torch.no_grad():
            if method == 'mean+std':
                std, mean = torch.std_mean(data, dim=0, keepdim=True)
                data.sub_(mean).divide_(std)
            elif method == 'mean':
                mean = torch.mean(data, dim=0, keepdim=True)
                data.sub_(mean)
            elif method == 'std':
                std = torch.std(data, dim=0, keepdim=True)
                data.divide_(std)
            else:
                raise ValueError(f'Unrecognized normalize method: {method}')

    @classmethod
    def get_singleton(cls, name, emb):
        if name in EmbeddingAdaptor.singleton_emb:
            return EmbeddingAdaptor.singleton_emb[name]
        EmbeddingAdaptor.singleton_emb[name] = emb = cls(emb)
        return emb
