from dataclasses import dataclass
from io import IOBase
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from fastNLP import DataSet, Vocabulary
from hydra.conf import MISSING
from hydra.utils import instantiate
from torch import Tensor
from torch.nn import Module

import src
from src.datamodule.task import DepDataModule
from src.model.embedding import EmbeddingAdaptor
from src.model.base import ModelBase
from src.model.dmv import DMV
from src.model.dmv_helper import generate_rule_1o, LinearPadder, SquarePadder
from src.model.nn import DMVFactorizedBilinear
from src.model.nn.multivariate_kl import MultVariateKLD
from src.model.torch_struct import DMV1o, DependencyCRF
from src.model.torch_struct.dmv import LEFT, RIGHT, NOCHILD
from src.utility.config import Config
from src.utility.fn import dict_apply
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]
_warn, _info, _debug = get_logger_func('model')


@dataclass
class LDNDMVConfig(Config):
    context_mode: str  # 'hx', 'mean', 'token', 'none'
    # 'y' or '<path to pretrained dmv>'
    # y means using the recoveried counts instead of the output of dmv
    # path means using the output of a fixed dmv
    init_method: str
    init_epoch: int
    viterbi_training: bool
    mbr_decoding: bool
    extended_valence: bool
    function_mask: bool

    # 'none': no variational, 'all:*' and 'tag:*' apply variational methods
    # '*' can be vae and ib. 'all' and 'tag' represent the non-variational part of embedding.
    variational_mode: str
    z_dim: int  # the output dim of variational methods. ignored if variational_mode=none

    mid_ff: AnyDict
    head_ff: AnyDict
    child_ff: AnyDict
    root_ff: AnyDict
    dec_ff: AnyDict

    attach_rank: int
    dec_rank: int
    root_rank: int

    root_emb_dim: int
    dec_emb_dim: int

    # ============================= AUTO FIELDS =============================
    n_word: int = MISSING
    n_tag: int = MISSING
    n_token: int = MISSING


class DiscriminativeNDMV(ModelBase):
    def __init__(self, **cfg):
        super().__init__()

        # noinspection PyTypeChecker
        self.cfg: LDNDMVConfig = cfg
        self.variational_enc: Optional[Module] = None
        self.target_mean: Optional[nn.Parameter] = None
        self.target_lvar: Optional[nn.Parameter] = None
        self.head_ff: Optional[Module] = None
        self.mid_ff: Optional[Module] = None
        self.child_ff: Optional[Module] = None
        self.root_ff: Optional[Module] = None
        self.dec_ff: Optional[Module] = None
        self.attach_scorer: Optional[Module] = None
        self.dec_scorer: Optional[Module] = None
        self.root_scorer: Optional[Module] = None
        self.dropout: Optional[Module] = None
        self.root_emb: Optional[Module] = None
        self.dec_emb: Optional[Module] = None
        self.token_mode: Optional[str] = None
        self.word_embedding: Optional[EmbeddingAdaptor] = None
        self.tag_embedding: Optional[EmbeddingAdaptor] = None
        self.gaussian_kl = MultVariateKLD('sum')
        # self.function_mask: Optional[Tensor] = None # set by register_buffer

        self.dmv: Optional[DMV] = None

    def setup(self, dm: DepDataModule):
        super().setup(dm)
        self.cfg = cfg = LDNDMVConfig.build(self.cfg, allow_missing={'n_word', 'n_tag'},
                                            ignore_unknown=self.__class__ is not DiscriminativeNDMV)

        # prepare token embedding
        self.token_mode = dm.token_mode
        if dm.token2word is not None:
            self.register_buffer('word_indexer', torch.tensor(dm.token2word))
            self.__dict__['word_embedding'] = self.embedding.word_embedding
        if dm.token2tag is not None:
            self.register_buffer('tag_indexer', torch.tensor(dm.token2tag))
            self.__dict__['tag_embedding'] = self.embedding.tag_embedding

        # prepare variational methods
        if cfg.variational_mode != 'none':
            assert cfg.context_mode != 'none'
            self.variational_enc = nn.Linear(self.encoder.get_dim('x'), cfg.z_dim * 2)
            if cfg.variational_mode.endswith('ib'):
                self.target_mean = nn.Parameter(torch.zeros(1, cfg.z_dim))
                self.target_lvar = nn.Parameter(torch.zeros(1, cfg.z_dim))
            if cfg.variational_mode.startswith('tag'):
                assert isinstance(cfg.n_tag, int), "no tag can be used"
                n_in = self.embedding.tag_embedding.embed_size + cfg.z_dim
            else:
                n_in = self.embedding.embed_size + cfg.z_dim
        else:
            n_in = (self.encoder.get_dim('x') if cfg.context_mode not in ('none', 'passthrough') else 0) \
                   + self.embedding.embed_size

        # prepare scorer
        self.head_ff = instantiate(cfg.head_ff, n_in=n_in)  # MLP
        self.child_ff = instantiate(cfg.child_ff, n_in=self.token_emb_size)  # MLP
        self.root_ff = instantiate(cfg.root_ff, n_in=cfg.root_emb_dim)  # MLP
        self.dec_ff = instantiate(cfg.dec_ff, n_in=cfg.dec_emb_dim)  # MLP

        assert self.head_ff.n_out == self.child_ff.n_out == self.root_ff.n_out == self.dec_ff.n_out
        self.mid_ff = instantiate(cfg.mid_ff, hidden_size=self.head_ff.n_out)  # DMVSkipConnectEncoder

        self.attach_scorer = DMVFactorizedBilinear(n_in=self.mid_ff.n_out, r=self.cfg.attach_rank)
        self.dec_scorer = DMVFactorizedBilinear(n_in=self.mid_ff.n_out, r=self.cfg.dec_rank)
        self.root_scorer = DMVFactorizedBilinear(n_in=self.mid_ff.n_out, r=self.cfg.root_rank)

        self.root_emb = nn.Parameter(torch.randn(1, cfg.root_emb_dim))
        self.dec_emb = nn.Parameter(torch.randn(2, cfg.dec_emb_dim))

        # setup function_mask
        if cfg.function_mask:
            masked_pos = 'ADP AUX CCONJ SCONJ CONJ DET PART'.split()
            assert dm.vocabs['tag'].unknown not in masked_pos
            masked_pos = [dm.vocabs['tag'][p] for p in masked_pos]
            self.register_buffer('function_mask', torch.tensor(masked_pos))

        # prepare initializer
        if cfg.init_method == 'y':
            train_ds = dm.datasets['train_init']
            train_ds.apply_more(lambda x: generate_rule_1o(x['arc']))
            train_ds.set_target('dec_rule', 'attach_rule', 'root_rule')
            train_ds['dec_rule'].set_padder(LinearPadder(0))
            train_ds['root_rule'].set_padder(LinearPadder(0))
            train_ds['attach_rule'].set_padder(SquarePadder(0))
        elif isinstance(cfg.init_method, str):
            assert cfg.extended_valence, "KM init only support extended_valence=true"
            self.dmv = DMV(viterbi_training=cfg.viterbi_training, mbr_decoding=cfg.mbr_decoding, init_method='random',
                           smooth=0.1, n_word=cfg.n_word, n_tag=cfg.n_tag, n_token=cfg.n_token)
            self.dmv.setup(dm)
            self.dmv.load_state_dict(
                dict_apply(torch.load(cfg.init_method)['state_dict'], key_func=lambda x: x.split('.')[1]))
        else:
            _info("No initialization is set.")

    # noinspection PyDictCreation
    def _forward(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        out = {}
        b, n = vp.batch_size, vp.max_len

        context, out['kl'] = self.extract_sent_repr(encoded)
        h = self.construct_token_repr(encoded['emb'], context, vp)

        # batch x len x dir x val x hidden
        h_parent = self.mid_ff(self.head_ff(h))
        h_child = self.mid_ff(self.child_ff(self.token_emb)).unsqueeze(0)
        h_root = self.mid_ff(self.root_ff(self.root_emb)).unsqueeze(0)
        h_dec = self.mid_ff(self.dec_ff(self.dec_emb)).unsqueeze(0)

        # [batch x len x dir x val x hidden] [1, state, dir, val, hidden] => [batch, len, state, dir, val]
        attach_rule = self.attach_scorer(h_parent, h_child).log_softmax(2)
        if not self.cfg.extended_valence:
            attach_rule = torch.stack([attach_rule[..., 0], attach_rule[..., 0]], dim=-1)
        target_size = torch.Size([b, n, n, 2, 2])
        attach_prob = attach_rule.gather(2, inputs['token'].reshape(b, 1, n, 1, 1).expand(target_size))
        left_mask = torch.tril(torch.ones(n, n, device=attach_prob.device), diagonal=-1)
        right_mask = torch.triu(torch.ones(n, n, device=attach_prob.device), diagonal=1)
        attach_prob = attach_prob[..., LEFT, :] * left_mask.unsqueeze(0).unsqueeze(-1) \
                      + attach_prob[..., RIGHT, :] * right_mask.unsqueeze(0).unsqueeze(-1)
        if self.cfg.function_mask:
            tag_array = inputs['tag'].unsqueeze(-1).unsqueeze(-1)
            mask_set = self.function_mask.view(1, 1, 1, -1)
            in_mask = tag_array.eq(mask_set).any(dim=-1, keepdims=True)
            attach_prob.masked_fill_(in_mask, -src.INF)
        out = {**out, 'attach': attach_prob, 'attach_rule': attach_rule}

        # permute: [batch x len x dec x dir x val] => [batch x len x dir x val x dec]
        dec_prob = self.dec_scorer(h_parent, h_dec).permute(0, 1, 3, 4, 2).log_softmax(-1)
        out = {**out, 'dec': dec_prob, 'dec_rule': dec_prob}

        # sum: root does not require dir and val. (always RIGHT, NOCHILD)
        root_prob = self.root_scorer(h_root, h_child).sum([-1, -2]).log_softmax(-1).squeeze(1).expand(b, -1)
        out: TensorDict = {**out, 'root': torch.gather(root_prob, 1, inputs['token']), 'root_rule': root_prob}

        out['merged_dec'], out['merged_attach'] = DMV1o.merge(out['dec'], out['attach'], out['root'])

        if src.trainer.current_epoch < self.cfg.init_epoch:
            if self.dmv is not None:
                # init using a pretrained dmv
                dmv_out = self.dmv._forward(inputs, {}, vp)
                out['dmv_merged_dec'], out['dmv_merged_attach'] = dmv_out['merged_dec'], dmv_out['merged_attach']
        return out

    def extract_sent_repr(self, encoded):
        if self.cfg.context_mode == 'none':
            return None, None

        b, l, *_ = encoded['x'].shape
        if self.cfg.context_mode == 'hx':
            context = encoded['hiddens'][-2:].permute(1, 0, 2).reshape(b, 1, -1)
        elif self.cfg.context_mode == 'mean':
            context = encoded['x'].mean(1, keepdim=True)
        elif self.cfg.context_mode == 'max':
            context = encoded['x'].max(1, keepdim=True)[0]
        else:
            context = encoded['x']

        if self.variational_enc is not None:
            mean, lvar = torch.chunk(self.variational_enc(context), 2, dim=-1)
            if self.cfg.variational_mode.endswith('ib'):
                _mean, _lvar = mean.view(-1, self.cfg.z_dim), lvar.view(-1, self.cfg.z_dim)
                _b = len(_mean)
                kl = self.gaussian_kl(_mean, self.target_mean.expand(_b, -1), _lvar, self.target_lvar.expand(_b, -1))
            else:
                kl = -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1).sum()
            if self.training:
                context = torch.empty_like(mean).normal_()
                context = (0.5 * lvar).exp() * context + mean
            else:
                context = mean
        else:
            kl = None

        if context.shape[1] == 1 and l > 1:
            context = context.expand(-1, l, -1)
        return context, kl

    def construct_token_repr(self, emb, sent, vp):
        if sent is None or (self.cfg.context_mode == 'passthrough' and self.cfg.variational_mode == 'none'):
            return emb
        if self.cfg.variational_mode.startswith('tag'):
            # vp contains all unconcated embedding
            return torch.cat([vp.tag_embedding, sent], dim=-1)
        return torch.cat([emb, sent], dim=-1)

    def loss(self, x: TensorDict, gold: InputDict, vp: VarPool) -> Tuple[Tensor, TensorDict]:
        out = {}
        if src.trainer.current_epoch < self.cfg.init_epoch and self.training:
            # _warn("Model: Initializing")
            if self.dmv is not None:
                # init using a pretrained dmv
                dec = x['dmv_merged_dec'].detach().requires_grad_()
                attach = x['dmv_merged_attach'].detach().requires_grad_()
                dist = DMV1o([dec, attach], vp.seq_len)
                expected_count = torch.autograd.grad(dist.partition.sum(), [dec, attach])
                out['enll'] = -(expected_count[0] * x['merged_dec']).sum() \
                              - (expected_count[1] * x['merged_attach']).sum()
            else:
                out['enll'] = -(gold['dec_rule'] * x['dec']).sum() \
                              - (gold['attach_rule'] * x['attach']).sum() \
                              - (gold['root_rule'] * x['root']).sum()
        else:
            dist = DMV1o([x['merged_dec'], x['merged_attach']], vp.seq_len)
            if self.cfg.viterbi_training:
                out['nll'] = -dist.max.sum()
            else:
                out['nll'] = -dist.partition.sum()
        if x['kl'] is not None:
            out['lstm_kl'] = x['kl']
        if 'kl' in vp:
            out['emb_kl'] = vp.kl
        return sum(out.values()), out

    # noinspection DuplicatedCode
    @torch.enable_grad()
    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        mdec = x['merged_dec'].detach().requires_grad_()
        mattach = x['merged_attach'].detach().requires_grad_()
        dist = DMV1o([mdec, mattach], vp.seq_len)
        if self.cfg.mbr_decoding:
            arc = torch.autograd.grad(dist.partition.sum(), mattach)[0].sum(-1)
            dist = DependencyCRF(arc, vp.seq_len)
            arc = dist.argmax.nonzero()  # b x n x n -> M x 3
            predicted = vp.seq_len.new_zeros(vp.batch_size, vp.max_len)
            predicted[arc[:, 0], arc[:, 2] - 1] = arc[:, 1]
        else:
            arc = dist.argmax.sum(-1).nonzero()
            predicted = vp.seq_len.new_zeros(vp.batch_size, vp.max_len)
            predicted[arc[:, 0], arc[:, 2] - 1] = arc[:, 1]
        return {'arc': predicted}

    # noinspection DuplicatedCode
    def write_prediction(self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]) -> IOBase:
        tag_vocab = vocabs['tag']
        for i, length in enumerate(dataset['seq_len'].content):
            word, tag, arc = dataset[i]['raw_word'], dataset[i]['tag'], predicts['arc'][i]
            for line_id, (word, tag, arc) in enumerate(zip(word, tag, arc), start=1):
                line = '\t'.join([str(line_id), word, tag_vocab.to_word(tag), str(arc)])
                s.write(f'{line}\n')
            s.write('\n')
        return s

    @property
    def token_emb(self):
        emb = []
        if self.word_embedding is not None:
            emb.append(self.word_embedding(self.word_indexer, None))
        if self.tag_embedding is not None:
            emb.append(self.tag_embedding(self.tag_indexer, None))
        return torch.cat(emb, dim=-1)

    @property
    def token_emb_size(self):
        emb_size = 0
        if self.word_embedding is not None:
            emb_size += self.word_embedding.embed_size
        if self.tag_embedding is not None:
            emb_size += self.tag_embedding.embed_size
        return emb_size
