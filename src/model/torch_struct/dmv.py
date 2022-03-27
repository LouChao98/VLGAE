import torch
from torch import Tensor

from .helpers import _Struct
from .semirings import Semiring

NOCHILD = 1
HASCHILD = 0
LEFT = 0
RIGHT = 1
GO = 0
STOP = 1
DIR_NUM = 2
VAL_NUM = 2
DEC_NUM = 2


class DMV1oStruct(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False, cache=False):
        # dec, attach
        s: Semiring = self.semiring

        if isinstance(scores[0], torch.Tensor):
            # attach_score: batch, N, N, valence
            # dec_score:    batch, N, direction, valence, decision
            attach: Tensor = s.convert(scores[1])
            dec: Tensor = s.convert(scores[0])
        else:
            attach: Tensor = s.convert([scores[0][1], scores[1][1]])
            dec: Tensor = s.convert([scores[0][0], scores[1][0]])

        _, batch, N, *_ = dec.shape
        # diagonal for left, diagonal(1) for right.
        I = s.zero_(attach.new_empty((s.size(), batch, N + 1, N + 1, VAL_NUM)))
        C = s.zero_(attach.new_empty((s.size(), batch, N + 1, N + 1, VAL_NUM)))
        attach_left = s.mul(attach, dec[:, :, :, None, LEFT, :, GO])
        attach_right = s.mul(attach, dec[:, :, :, None, RIGHT, :, GO])

        diag_minus1(C, 0, 2, 3).copy_(dec[:, :, :, LEFT, :, STOP].transpose(-2, -1))
        C.diagonal(1, 2, 3).copy_(dec[:, :, :, RIGHT, :, STOP].transpose(-2, -1))
        _zero = C.new_tensor(s.zero)
        if _zero.ndim == 0:
            _zero = _zero.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            _zero = _zero.unsqueeze(-1).unsqueeze(-1)

        for w in range(1, N):
            n = N - w

            x = s.sum(s.mul(stripe_val(C, n, w, (0, 1, NOCHILD)), stripe_val(C, n, w, (w, 1, HASCHILD))))
            x = s.times(x.unsqueeze(-2), attach_left.diagonal(-w, -3, -2))
            diag_minus1(I, -w, -3, -2).copy_(x)

            x = s.sum(s.mul(stripe_val(C, n, w, (0, 1, HASCHILD)), stripe_val(C, n, w, (w, 1, NOCHILD))))
            x = s.times(x.unsqueeze(-2), attach_right.diagonal(w, -3, -2))
            I.diagonal(w + 1, -3, -2).copy_(x)

            x = s.sum(s.mul(stripe_val(C, n, w, (0, 0, NOCHILD), 0, True), stripe_noval(I, n, w, (w, 0))), -2)
            diag_minus1(C, -w, -3, -2).copy_(x.transpose(-2, -1))

            x = s.sum(s.mul(stripe_noval(I, n, w, (0, 2)), stripe_val(C, n, w, (1, w + 1, NOCHILD), 0, True)), -2)
            C.diagonal(w + 1, -3, -2).copy_(x.transpose(-2, -1))
            C[:, lengths.ne(w), 0, w + 1] = _zero

        v = torch.gather(C[:, :, 0, :, NOCHILD], -1, (lengths[None, ..., None] + 1).expand(s.size(), -1, -1))
        return v, [dec, attach], [C, I]

    def _arrange_marginals(self, marg):
        return marg[1]  # return attach


def stripe_val(x: Tensor, n, w, offset=(0, 0, 0), dim=1, keep_val=False):
    # x: s x b x N x N x valence
    # on the last three dim, N x N x valence
    # n and w are for N x N
    assert x.shape[-1] == 2
    assert x.is_contiguous(), 'x must be contiguous, or write on new view will lost.'
    seq_len = x.shape[-2]
    if keep_val:
        size = (*x.shape[:-3], n, w, 1)
        stride = list(x.stride())
        stride[-3] = (seq_len + 1) * 2
        stride[-2] = (1 if dim == 1 else seq_len) * 2
    else:
        stride = list(x.stride())[:-1]
        stride[-2] = (seq_len + 1) * 2
        stride[-1] = (1 if dim == 1 else seq_len) * 2
        size = (*x.shape[:-3], n, w)
    return x.as_strided(size=size,
                        stride=stride,
                        storage_offset=x.storage_offset() + (offset[0] * seq_len * 2 + offset[1] * 2 + offset[2]))


def stripe_noval(x: Tensor, n, w, offset=(0, 0), dim=1):
    # x: s x b x N x N x valence
    # on the last three dim, N x N x valence
    # n and w are for N x N
    assert x.shape[-1] == 2
    assert x.is_contiguous(), 'x must be contiguous, or write on new view will lost.'
    seq_len = x.shape[-2]
    stride = list(x.stride())
    stride[-3] = (seq_len + 1) * 2
    stride[-2] = (1 if dim == 1 else seq_len) * 2
    return x.as_strided(size=(*x.shape[:-3], n, w, 2),
                        stride=stride,
                        storage_offset=x.storage_offset() + (offset[0] * seq_len * 2 + offset[1] * 2))


def diag_minus1(x: Tensor, offset, dim1, dim2) -> Tensor:
    # assume a[..., dim1, ..., dim2, ...]
    stride = list(x.stride())
    if offset > 0:
        storage_offset = stride[dim2] * offset
    else:
        storage_offset = stride[dim1] * abs(offset)
    to_append = stride[dim1] + stride[dim2]
    if dim2 < 0:
        stride.pop(dim1)
        stride.pop(dim2)
    else:
        stride.pop(dim2)
        stride.pop(dim1)  # todo handle +/- or -/+ (now only support +/+ ans -/-)
    stride.append(to_append)
    size = list(x.size())
    to_append = size[dim1] - 1 - abs(offset)
    if dim2 < 0:
        size.pop(dim1)
        size.pop(dim2)
    else:
        size.pop(dim2)
        size.pop(dim1)
    size.append(to_append)
    return x.as_strided(size, stride, storage_offset=storage_offset)
