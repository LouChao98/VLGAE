import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from src.model.nn.dropout import SharedDropout


class VariationalLSTM(nn.Module):
    r"""
    VariationalLSTM :cite:`yarin-etal-2016-dropout` is an variant of the vanilla bidirectional LSTM
    adopted by Biaffine Parser with the only difference of the dropout strategy.
    It drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.
    APIs are roughly the same as :class:`~torch.nn.LSTM` except that we only allows
    :class:`~torch.nn.utils.rnn.PackedSequence` as input.
    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state `h`.
        num_layers (int):
            The number of recurrent layers. Default: 1.
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer (except last).
            Default: 0.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, cell=nn.LSTMCell, init='zy'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.init = init

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(cell(input_size=input_size, hidden_size=hidden_size))
            self.b_cells.append(cell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        s = f'{self.input_size}, {2 * self.hidden_size}'
        if self.num_layers > 1:
            s += f', num_layers={self.num_layers}'
        if self.dropout > 0:
            s += f', dropout={self.dropout}'

        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self):
        if self.init == 'zy':
            for name, param in self.named_parameters():
                if name.startswith('lstm'):
                    # apply orthogonal_ to weight
                    if len(param.shape) > 1:
                        nn.init.orthogonal_(param)
                    # apply zeros_ to bias
                    else:
                        nn.init.zeros_(param)
        elif self.init == 'biased':
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
        else:
            raise ValueError(f'Bad init_version, {self.cfg.init_version=}')

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence: PackedSequence, hx=None):
        r"""
        Args:
            sequence (~torch.nn.utils.rnn.PackedSequence):
                A packed variable length sequence.
            hx (~torch.Tensor, ~torch.Tensor):
                A tuple composed of two tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial hidden state
                for each element in the batch.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial cell state
                for each element in the batch.
                If `hx` is not provided, both `h` and `c` default to zero.
                Default: ``None``.
        Returns:
            ~torch.nn.utils.rnn.PackedSequence, (~List[torch.Tensor], ~torch.Tensor):
                The first is a list of packed variable length sequence for each layer.
                The second is a tuple of tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]``
                holds the hidden state for `t=seq_len`.
                Like output, the layers can be separated using
                ``h.view(num_layers, num_directions, batch_size, hidden_size)``
                and similarly for c.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]``
                holds the cell state for `t=seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n, hiddens = [], [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = hx
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training and i > 0:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_i, (h_i, c_i) = self.layer_forward(x, (h[i, 0], c[i, 0]), self.f_cells[i], batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x, (h[i, 1], c[i, 1]), self.b_cells[i], batch_sizes, True)
            x_i = torch.cat((x_i, x_b), -1)
            h_i = torch.stack((h_i, h_b))
            c_i = torch.stack((c_i, c_b))
            x = x_i
            h_n.append(h_i)
            c_n.append(c_i)
            hiddens.append(
                PackedSequence(x_i, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices))

        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        return hiddens, hx
