from dataclasses import dataclass
from io import IOBase
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from fastNLP import DataSet, Vocabulary
from hydra.conf import MISSING
from torch import Tensor
from torch.optim import Optimizer

from src.datamodule.task import DepDataModule
from src.model import ModelBase
from src.model.dmv_helper import km_init, good_init
from src.model.torch_struct import DMV1o, DependencyCRF
from src.utility.config import Config
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]

_warn, _info, _debug = get_logger_func('model')


@dataclass
class DMVConfig(Config):
    viterbi_training: bool
    mbr_decoding: bool
    init_method: str  # km, good, random
    smooth: float

    # ============================= AUTO FIELDS =============================
    n_word: int = MISSING
    n_tag: int = MISSING
    n_token: int = MISSING


class DMV(ModelBase):
    _instance = None  # work around for DMVMStepOptimizer

    def __init__(self, **cfg):
        super().__init__()
        # noinspection PyTypeChecker
        self.cfg: DMVConfig = cfg
        self.root_param: Optional[nn.Parameter] = None
        self.trans_param: Optional[nn.Parameter] = None
        self.dec_param: Optional[nn.Parameter] = None
        self.optimizer: Optional[DMVMStepOptimizer] = None 

        if DMV._instance is not None:
            _warn('overwriting DMV._instance')
        DMV._instance = self

    def setup(self, dm: DepDataModule):
        self.datamodule = dm
        self.cfg = cfg = DMVConfig.build(self.cfg, allow_missing={'n_word', 'n_tag'})

        if cfg.init_method == 'km':
            d, t, r = km_init(dm.datasets['train'], cfg.n_token, cfg.smooth)
        elif cfg.init_method == 'good':
            d, t, r = good_init(dm.datasets['train'], cfg.n_token, cfg.smooth)
        else:
            d = np.random.randn(cfg.n_token, 2, 2, 2)
            r = np.random.randn(cfg.n_token)
            t = np.random.randn(cfg.n_token, cfg.n_token, 2, 2)

        self.root_param = nn.Parameter(torch.from_numpy(r))
        # head, child, dir, valence
        self.trans_param = nn.Parameter(torch.from_numpy(t))
        # head, dir, valence, decision
        self.dec_param = nn.Parameter(torch.from_numpy(d))

    def forward(self, inputs: InputDict, vp: VarPool, embed=None, encoded=None, return_all=False):
        assert embed is None
        assert encoded is None
        assert not return_all
        return self._forward(inputs, {}, vp)

    def _forward(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        b, l, n = vp.batch_size, vp.max_len, self.cfg.n_token
        token_array = inputs['token']

        t = self.trans_param.unsqueeze(0).expand(b, n, n, 2, 2)
        head_token_index = token_array.view(b, l, 1, 1, 1).expand(b, l, n, 2, 2)
        child_token_index = token_array.view(b, 1, l, 1, 1).expand(b, l, l, 2, 2)
        t = torch.gather(torch.gather(t, 1, head_token_index), 2, child_token_index)
        index = torch.triu(torch.ones(l, l, dtype=torch.long, device=t.device)) \
            .view(1, l, l, 1, 1).expand(b, l, l, 1, 2)
        t = torch.gather(t, 3, index).squeeze(3)

        d = self.dec_param.unsqueeze(0).expand(b, n, 2, 2, 2)
        head_pos_index = token_array.view(b, l, 1, 1, 1).expand(b, l, 2, 2, 2)
        d = torch.gather(d, 1, head_pos_index)

        r = self.root_param.unsqueeze(0).expand(b, n)
        r = torch.gather(r, 1, token_array)

        merged_d, merged_t = DMV1o.merge(d, t, r)
        return {'merged_dec': merged_d, 'merged_attach': merged_t}

    def loss(self, x: TensorDict, gold: InputDict, vp: VarPool) -> Tuple[Tensor, TensorDict]:
        dist = DMV1o([x['merged_dec'], x['merged_attach']], vp.seq_len)
        if self.cfg.viterbi_training:
            ll = dist.max.sum()
        else:
            ll = dist.partition.sum()
        return -ll, {'ll': ll}

    # noinspection DuplicatedCode
    @torch.enable_grad()
    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        if self.optimizer:
            self.optimizer.apply()
        mdec = x['merged_dec'].detach().requires_grad_()
        mattach = x['merged_attach'].detach().requires_grad_()
        dist = DMV1o([mdec, mattach], vp.seq_len)
        if self.cfg.mbr_decoding:
            arc = torch.autograd.grad(dist.partition.sum(), mattach)[0].sum(-1)
            dist = DependencyCRF(arc, vp.seq_len)
            arc = dist.argmax.nonzero()
            predicted = vp.seq_len.new_zeros(vp.batch_size, vp.max_len)
            predicted[arc[:, 0], arc[:, 2] - 1] = arc[:, 1]
        else:
            arc = dist.argmax.sum(-1).nonzero()
            predicted = vp.seq_len.new_zeros(vp.batch_size, vp.max_len)
            predicted[arc[:, 0], arc[:, 2] - 1] = arc[:, 1]
        return {'arc': predicted}

    def normalize_embedding(self, now):
        pass

    # noinspection DuplicatedCode
    def write_prediction(self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]) -> IOBase:
        for i, length in enumerate(dataset['seq_len'].content):
            word, arc = dataset[i]['raw_word'], predicts['arc'][i]
            for line_id, (word, arc) in enumerate(zip(word, arc), start=1):
                line = '\t'.join([str(line_id), word, '-', str(arc)])
                s.write(f'{line}\n')
            s.write('\n')
        return s


class DMVMStepOptimizer(Optimizer):
    def __init__(self, params, smooth: float):
        self.dmv = DMV._instance
        self.dmv.optimizer = self

        self._root, self._dec, self._trans = None, None, None
        self.smooth = smooth
        self.can_apply = False
        super().__init__(self.dmv.parameters(), {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._root is None:
            self._root = torch.zeros_like(self.dmv.root_param)
            self._dec = torch.zeros_like(self.dmv.dec_param)
            self._trans = torch.zeros_like(self.dmv.trans_param)

        self._root -= self.dmv.root_param.grad
        self._dec -= self.dmv.dec_param.grad
        self._trans -= self.dmv.trans_param.grad
        self.can_apply = True

    def apply(self):
        if self.can_apply:
            self.dmv.root_param.data, self._root = \
                torch.log(self._root + self.smooth).log_softmax(0), self.dmv.root_param.data
            self.dmv.dec_param.data, self._dec = \
                torch.log(self._dec + self.smooth).log_softmax(3), self.dmv.dec_param.data
            self.dmv.trans_param.data, self._trans = \
                torch.log(self._trans + self.smooth).log_softmax(1), self.dmv.trans_param.data
            self.reset()

    def reset(self):
        self._root.zero_()
        self._dec.zero_()
        self._trans.zero_()
        self.can_apply = False
