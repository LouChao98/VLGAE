from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn as nn
from omegaconf import MISSING
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from src.model.embedding import Embedding
from src.model.text_encoder.base import EncoderBase
from src.model.nn import ScalarMix, SharedDropout, VariationalLSTM
from src.utility.config import Config
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool

_warn, _info, _debug = get_logger_func('encoder')
RNN_TYPE_DICT = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
RNNCELL_TYPE_DICT = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}


@dataclass
class LSTMEncoderConfig(Config):
    reproject_emb: int = 0  # reproject layer before lstm
    # ============================= dropout ==============================
    pre_shared_dropout: float = 0.
    pre_dropout: float = 0.
    post_shared_dropout: float = 0.
    post_dropout: float = 0.

    # =============================== lstm ===============================
    rnn_type: str = 'lstm'  # lstm, gru or rnn
    hidden_size: Union[int, List[int]] = MISSING  # hidden size for each layer
    proj_size: int = 0  # projective size
    num_layers: int = MISSING  # total layers
    output_layers: Union[int, List[int]] = -1  # which layers are return, start from 0
    init_version: str = 'biased'
    shared_dropout: bool = True
    lstm_dropout: float = 0.33  # only between layers, unlike zhangyu.
    no_eos: bool = False  # simulate no <eos>
    sorted: bool = True

    # ============================== output ==============================
    mix: bool = False  # whether to use a ScaleMix when multiple outputs
    reproject_out: int = 0
    cat_emb: bool = False


class RNNEncoder(EncoderBase):

    def __init__(self, embedding: Embedding, **cfg):
        super().__init__(embedding)
        self.cfg = cfg = LSTMEncoderConfig.build(cfg)

        # check output_layers
        output_layers: List[int] = [cfg.output_layers] if isinstance(cfg.output_layers, int) else cfg.output_layers
        output_layers = sorted(cfg.num_layers + o if o < 0 else o for o in output_layers)
        assert output_layers[0] >= 0 and output_layers[-1] < cfg.num_layers
        if output_layers[-1] < cfg.num_layers - 1:
            cfg.num_layers = output_layers[-1] + 1
            _warn(f'max index of output_layers is smaller to n_layers, n_layers is set to {cfg.num_layers}')
        self.output_layers = output_layers

        self.embedding2nn = nn.Linear(embedding.embed_size, cfg.reproject_emb) if cfg.reproject_emb else nn.Identity()

        # ============================= dropout ==============================

        self.pre_shared_dropout = SharedDropout(cfg.pre_shared_dropout) if cfg.pre_shared_dropout else nn.Identity()
        self.pre_dropout = nn.Dropout(cfg.pre_dropout) if cfg.pre_dropout else nn.Identity()
        self.post_shared_dropout = SharedDropout(cfg.post_shared_dropout) if cfg.post_shared_dropout else nn.Identity()
        self.post_dropout = nn.Dropout(cfg.post_dropout) if cfg.post_dropout else nn.Identity()

        # =============================== lstm ===============================

        input_size = cfg.reproject_emb if cfg.reproject_emb > 0 else embedding.embed_size
        if cfg.shared_dropout:
            assert isinstance(cfg.hidden_size, int), 'Not supported'
            assert cfg.proj_size == 0, 'Not supported'
            self.lstm = VariationalLSTM(input_size, cfg.hidden_size, cfg.num_layers, cfg.lstm_dropout,
                                        RNNCELL_TYPE_DICT[cfg.rnn_type])
            self.output_size = 2 * cfg.hidden_size
        else:
            # figure out how many layers in each sub modules
            layer_for_each_rnn = [x - y for x, y in zip(output_layers, [-1] + output_layers[:-1])]

            # check hiddens
            if isinstance(cfg.hidden_size, int):
                hiddens = [cfg.hidden_size for _ in layer_for_each_rnn]
            else:
                hiddens = cfg.hidden_size
                assert len(hiddens) == len(layer_for_each_rnn)

            # construct nn
            self.lstm_dropout = nn.Dropout(cfg.lstm_dropout)
            self.lstm = nn.ModuleList()
            rnn_type = RNN_TYPE_DICT[cfg.rnn_type]
            for n_layer, hidden in zip(layer_for_each_rnn, hiddens):
                sub_lstm = rnn_type(input_size,
                                    hidden,
                                    n_layer,
                                    dropout=cfg.lstm_dropout if n_layer > 1 else 0,
                                    bidirectional=True,
                                    proj_size=cfg.proj_size if hidden > cfg.proj_size > 0 else 0)
                self.lstm.append(sub_lstm)
                input_size = 2 * cfg.proj_size if cfg.proj_size else 2 * hidden
            self.output_size = 2 * cfg.proj_size if cfg.proj_size else 2 * hiddens[-1]

        if cfg.mix:
            assert isinstance(cfg.hidden_size, int) or all(h == cfg.hidden_size[0] for h in cfg.hidden_size), \
                'Only if has same dim for all layers, mix can be used.'
            self.mix = ScalarMix(len(output_layers))
        else:
            self.output_size *= len(output_layers)

        if cfg.reproject_out:
            self.nn2out = nn.Linear(self.output_size, cfg.reproject_out)
            self.output_size = cfg.reproject_out
        else:
            self.nn2out = nn.Identity()

        if cfg.cat_emb:
            self.output_size += embedding.embed_size

        self.reset_parameters(cfg.init_version)

    def reset_parameters(self, init_method):
        if init_method == 'zy':
            for name, param in self.named_parameters():
                if name.startswith('lstm'):
                    # apply orthogonal_ to weight
                    if len(param.shape) > 1:
                        nn.init.orthogonal_(param)
                    # apply zeros_ to bias
                    else:
                        nn.init.zeros_(param)
        elif init_method == 'biased':
            for name, param in self.named_parameters():
                if name.startswith('lstm'):
                    # apply orthogonal_ to weight
                    if len(param.shape) > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        # based on https://github.com/pytorch/pytorch/issues/750#issuecomment-280671871
                        param.data.fill_(0.)
                        n = param.shape[0]
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
        # else:
        #     raise ValueError(f'Bad init_version, {self.cfg.init_version=}')

    def forward(self, x: Tensor, vp: VarPool, hiddens=None):
        """
        :param x: output of embedding
        :param vp: the varpool
        :param hiddens: ttbp
        :return: a dict contains
            x, Tensor: the concated or mixed representation.
            all: List[Tensor], a list contains all outputs specified in self.output_layers.
            hx: the output state for all layers.
        """
        if isinstance(x, list): x = torch.cat(x, dim=-1)

        emb = x
        x = self.embedding2nn(x)
        x = self.pre_shared_dropout(x)
        x = self.pre_dropout(x)
        xs, hx = self.lstm_forward(x, vp, hiddens)
        if self.cfg.mix:
            x = self.mix(xs)
        else:
            x = torch.cat(xs, dim=-1)
        x = self.post_dropout(x)
        x = self.post_shared_dropout(x)
        if self.cfg.no_eos:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)], dim=1)
        x = self.nn2out(x)

        if self.cfg.cat_emb:
            x = torch.cat([x,  emb], dim=-1)

        # from src.utility.fn import draw_att
        # draw_att(x[0])

        return {'x': x, 'all': xs, 'hiddens': hx}

    def lstm_forward(self, x: Tensor, vp: VarPool, hiddens=None):
        if self.cfg.no_eos:
            x = x[:, :-1]
            x = pack_padded_sequence(x, vp.seq_len_cpu - 1, True, enforce_sorted=self.cfg.sorted)
        else:
            x = pack_padded_sequence(x, vp.seq_len_cpu, True, enforce_sorted=self.cfg.sorted)

        if self.cfg.shared_dropout:
            outputs, (hx, _) = self.lstm(x, hiddens)
            outputs = [outputs[i] for i in self.output_layers]
            outputs = [pad_packed_sequence(o, True)[0] for o in outputs]
        else:
            layer_count = -1
            outputs = []
            output_layers = self.output_layers.copy()
            hx = []
            hiddens = hiddens if hiddens is not None else [None] * len(self.lstm)

            for layer, hidden in zip(self.lstm, hiddens):
                output: PackedSequence
                output, (hx_, _) = layer(x, hidden)
                hx.append(hx_)

                layer_count += layer.num_layers
                if layer_count == output_layers[0]:
                    output_layers.pop(0)
                    outputs.append(pad_packed_sequence(output, True)[0])

                data = self.lstm_dropout(output.data)
                x = PackedSequence(data, output.batch_sizes, output.sorted_indices, output.unsorted_indices)
            hx = torch.cat(hx, 0)
        return outputs, hx

    def get_dim(self, field):
        if field == 'x' or field == 'all':
            return self.output_size
        return super().get_dim(field)
