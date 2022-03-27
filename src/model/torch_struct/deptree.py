import itertools

import torch
from torch import Tensor

from .helpers import Chart, _Struct

# Constants
# A, B are right-facing and left-facing respectly. (see 6.b vectorized parsing)
# L, R, C, I are flags in eisner algorithm.
A, B, R, C, L, I = 0, 1, 1, 1, 0, 0


class DepTree(_Struct):
    """
    A projective dependency CRF.

    Parameters:
        arc_scores_in: Arc scores of shape (B, N + 1, N + 1) or (B, N + 1, N + 1, L) with root at index 0.
            parent to child, or arc_scores_in[0, i, j] is the score of arc i to j.

    Note: For single-root case, cache is forced to False
    """

    def _dp(self, arc_scores_in, lengths=None, force_grad=False, cache=True, max_width=None):
        multiroot = getattr(self, 'multiroot', False)
        assert not multiroot

        anchor = arc_scores_in if isinstance(arc_scores_in, Tensor) else arc_scores_in[0]
        if anchor.dim() not in (3, 4):
            raise ValueError('potentials must have dim of 3 (unlabeled) or 4 (labeled)')

        labeled = anchor.dim() == 4

        # 1. Check shape, length
        # 2. Call requires_grad_
        # 3. Mask out scores on invalid position
        arc_scores_in, batch, N, lengths = self._check_potentials(arc_scores_in, lengths)

        s = self.semiring
        arc_scores = s.sum(arc_scores_in) if labeled else arc_scores_in
        I = s.zero_(arc_scores_in.new_empty(*(s.size(), batch, N, N)))
        C = s.zero_(arc_scores_in.new_empty(*(s.size(), batch, N, N)))
        s.one_(C.diagonal(dim1=-2, dim2=-1))
        _zero = C.new_tensor(s.zero).unsqueeze(-1)

        bound = N if max_width is None else max_width
        for w in range(1, bound):
            n = N - w
            # two complete span form a incomplete span, also add an arc
            # ilr = C(i->r) + C(j->r+1)
            # [semiring, batch, n, w]
            ilr = s.mul(stripe(C, n, w), stripe(C, n, w, (w, 1)))
            il = ir = s.sum(ilr)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i
            # with I(j->i) of n spans
            I.diagonal(-w, -2, -1).copy_(s.mul(il, arc_scores.diagonal(-w, -2, -1)))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i
            # with I(i->j) of n spans
            I.diagonal(w, -2, -1).copy_(s.mul(ir, arc_scores.diagonal(w, -2, -1)))

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = s.mul(stripe(C, n, w, (0, 0), 0), stripe(I, n, w, (w, 0)))
            C.diagonal(-w, -2, -1).copy_(s.sum(cl))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = s.mul(stripe(I, n, w, (0, 1)), stripe(C, n, w, (1, w), 0))
            C.diagonal(w, -2, -1).copy_(s.sum(cr))
            # disable multi words to modify the root
            if not multiroot:
                C[:, lengths.ne(w), 0, w] = _zero

        final = C[:, :, 0]
        v = torch.gather(final, -1, lengths.view(1, -1, 1).expand(s.size(), -1, 1)).squeeze(-1)
        return v, [arc_scores_in], [I, C]

    def _dp_orig(self, arc_scores_in, lengths=None, force_grad=False, cache=True):

        multiroot = getattr(self, 'multiroot', False)
        cache = False if not multiroot else cache

        # When KLDiv or CrossEntropy, arc_scores_in=[score_p, score_q]
        anchor = arc_scores_in if isinstance(arc_scores_in, Tensor) else arc_scores_in[0]
        if anchor.dim() not in (3, 4):
            raise ValueError('potentials must have dim of 3 (unlabeled) or 4 (labeled)')
        labeled = anchor.dim() == 4

        # 1. Check shape, length
        # 2. Call requires_grad_
        # 3. Mask out scores on invalid position
        arc_scores_in, batch, N, lengths = self._check_potentials(arc_scores_in, lengths)

        # Init, every chart has shape (semiring_size, batch, N[parent], N[span length])
        # Chart A,B have different direction, for example, for A, 0 is length=1, while for B, -1 is length=1.
        # This is because Length_A[i] + Legnth_B[i] = k (the k in the following for-loop)

        semiring = self.semiring
        arc_scores = semiring.sum(arc_scores_in) if labeled else arc_scores_in
        alpha = [[[Chart((batch, N, N), arc_scores, semiring, cache=cache) for _ in range(2)] for _ in range(2)]
                 for _ in range(2)]
        semiring.one_(alpha[A][C][L].data[:, :, :, 0].data)
        semiring.one_(alpha[A][C][R].data[:, :, :, 0].data)
        semiring.one_(alpha[B][C][L].data[:, :, :, -1].data)
        semiring.one_(alpha[B][C][R].data[:, :, :, -1].data)

        start_idx = 0 if multiroot else 1
        for k in range(1, N - start_idx):
            # two complete span form a incomplete span, also add an arc
            ACR = alpha[A][C][R][start_idx:N - k, :k]
            BCL = alpha[B][C][L][k + start_idx:, N - k:]
            x = semiring.dot(ACR, BCL)
            arcs_l = semiring.times(x, torch.diagonal(arc_scores, -k, dim1=-2, dim2=-1)[..., start_idx:])
            alpha[A][I][L][start_idx:N - k, k] = arcs_l
            alpha[B][I][L][k + start_idx:N, N - k - 1] = arcs_l
            arcs_r = semiring.times(x, torch.diagonal(arc_scores, k, dim1=-2, dim2=-1)[..., start_idx:])
            alpha[A][I][R][start_idx:N - k, k] = arcs_r
            alpha[B][I][R][k + start_idx:N, N - k - 1] = arcs_r

            # one complete span and one incomplete span form a complete span
            # there are two direction: c->i or i->c
            ACL = alpha[A][C][L][start_idx:N - k, :k]
            BIL = alpha[B][I][L][k + start_idx:, N - k - 1:N - 1]
            new = semiring.dot(ACL, BIL)
            alpha[A][C][L][start_idx:N - k, k] = new
            alpha[B][C][L][k + start_idx:N, N - k - 1] = new

            AIR = alpha[A][I][R][start_idx:N - k, 1:k + 1]
            BCR = alpha[B][C][R][k + start_idx:, N - k:]
            new = semiring.dot(AIR, BCR)
            alpha[A][C][R][start_idx:N - k, k] = new
            alpha[B][C][R][k + start_idx:N, N - k - 1] = new

        if not multiroot:
            # if not multiroot, there are one extra arc from ROOT to a word.
            root_incomplete_span = semiring.times(alpha[A][C][L][1, :N - 1], arc_scores[:, :, 0, 1:])
            for k in range(1, N):
                AIR = root_incomplete_span[:, :, :k]
                BCR = alpha[B][C][R][k, N - k:]
                alpha[A][C][R][0, k] = semiring.dot(AIR, BCR)

        final = alpha[A][C][R][(0, )]
        v = torch.stack([final[:, i, l] for i, l in enumerate(lengths)], dim=1)
        return v, [arc_scores_in], alpha

    def _check_potentials(self, arc_scores, lengths=None):
        semiring = self.semiring
        batch, N, N2, *_ = self._get_dimension_and_requires_grad(arc_scores)
        assert N == N2, 'Non-square potentials'

        if lengths is None:
            lengths = torch.LongTensor([N - 1] * batch).to(arc_scores.device)
        else:
            assert max(lengths) <= N, 'Length longer than N'

        arc_scores = semiring.convert(arc_scores)
        arc_scores = arc_scores.clone()  # avoid leaf error when backward

        for b in range(batch):
            semiring.zero_(arc_scores[:, b, lengths[b] + 1:, :])
            semiring.zero_(arc_scores[:, b, :, lengths[b] + 1:])
        return arc_scores, batch, N, lengths

    def _arrange_marginals(self, grads):
        return self.semiring.convert(self.semiring.unconvert(grads[0]))

    @staticmethod
    def to_parts(sequence: Tensor, extra=None, lengths=None):
        """
        Convert a sequence representation to arcs

        Parameters:
            sequence : b x (N+1) long tensor in [0, N] (indexing is +1), where 0 is root (and its value is ignored).
            index: seq_len without root
        Returns:
            arcs : b x (N+1) x (N+1) arc indicators
        """
        batch, N1 = sequence.shape
        if lengths is None:
            lengths = torch.LongTensor([N1 - 1] * batch)
        labels = torch.zeros(batch, N1, N1).long()
        for n in range(1, N1):
            labels[torch.arange(batch), sequence[:, n], n] = 1
        for b in range(batch):
            labels[b, lengths[b] + 1:, :] = 0
            labels[b, :, lengths[b] + 1:] = 0
        return labels

    @staticmethod
    def from_parts(arcs):
        """
        Convert a arc representation to sequence

        Parameters:
            arcs : b x (N+1) x (N+1) arc indicators
        Returns:
            sequence : b x (N+1) long tensor in [0, N] (indexing is +1), where 0 is root (and its value is always 0).
        """
        batch, N, _ = arcs.shape
        labels = torch.zeros(batch, N).long()
        on = arcs.nonzero(as_tuple=False)
        for i in range(on.shape[0]):
            labels[on[i][0], on[i][2]] = on[i][1]
        labels[:, 0] = 0
        return labels, None

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1, )).item()
        N = torch.randint(2, 4, (1, )).item()
        return torch.rand(b, N, N), (b, N)

    def enumerate(self, arc_scores, non_proj=False, multi_root=True):
        semiring = self.semiring
        parses = []
        q = []
        batch, N, _ = arc_scores.shape
        for mid in itertools.product(range(N + 1), repeat=N - 1):
            parse = [-1] + list(mid)
            if not _is_spanning(parse):
                continue
            if not non_proj and not _is_projective(parse):
                continue
            if not multi_root and _is_multi_root(parse):
                continue
            q.append(parse)
            parses.append(semiring.times(*[arc_scores[:, parse[i], i] for i in range(1, N, 1)]))
        return semiring.sum(torch.stack(parses, dim=-1)), None


def stripe(x, n, w, offset=(0, 0), dim=1):
    # based on yzhangcs's supar/utils/fn.py:stripe
    # https://github.com/yzhangcs/parser
    # MODIFIED: on the last two dim
    # ORIG: on the first two dim
    r"""
    Returns a diagonal stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.

    Returns:
        a diagonal stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """
    assert x.is_contiguous(), 'x must be contiguous, or write on new view will lost.'
    seq_len = x.size(-1)
    stride = list(x.stride())
    stride[-2] = seq_len + 1
    stride[-1] = 1 if dim == 1 else seq_len
    return x.as_strided(size=(*x.shape[:-2], n, w),
                        stride=stride,
                        storage_offset=x.storage_offset() + (offset[0] * seq_len + offset[1]))


def deptree_nonproj(arc_scores, eps=1e-5):
    """
    Compute the marginals of a non-projective dependency tree using the
    matrix-tree theorem.

    Allows for overlapping arcs.

    Much faster, but cannot provide a semiring.

    Parameters:
         arc_scores : b x N x N arc scores with root scores on diagonal.
         semiring

    Returns:
         arc_marginals : b x N x N.
    """

    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    lap[:, 0] = torch.diagonal(input, 0, -2, -1).exp()
    inv_laplacian = lap.inverse()
    factor = (torch.diagonal(inv_laplacian, 0, -2, -1).unsqueeze(2).expand_as(input).transpose(1, 2))
    term1 = input.exp().mul(factor).clone()
    term2 = input.exp().mul(inv_laplacian.transpose(1, 2)).clone()
    term1[:, :, 0] = 0
    term2[:, 0] = 0
    output = term1 - term2
    roots_output = (torch.diagonal(input, 0, -2, -1).exp().mul(inv_laplacian.transpose(1, 2)[:, 0]))
    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    return output


# Tests


def deptree_part(arc_scores, eps=1e-5):
    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    lap[:, 0] = torch.diagonal(input, 0, -2, -1).exp()
    return lap.logdet()


def _is_spanning(parse):
    """
    Is the parse tree a valid spanning tree?
    Returns
    --------
    spanning : bool
    True if a valid spanning tree.
    """
    d = {}
    for m, h in enumerate(parse):
        if m == h:
            return False
        d.setdefault(h, [])
        d[h].append(m)
    stack = [0]
    seen = set()
    while stack:
        cur = stack[0]
        if cur in seen:
            return False
        seen.add(cur)
        stack = d.get(cur, []) + stack[1:]
    if len(seen) != len(parse) - len([1 for p in parse if p is None]):
        return False
    return True


def _is_multi_root(parse):
    root_count = 0
    for m, h in enumerate(parse):
        if h == 0:
            root_count += 1
    return root_count > 1


def _is_projective(parse):
    """
    Is the parse tree projective?
    Returns
    --------
    projective : bool
       True if a projective tree.
    """
    for m, h in enumerate(parse):
        for m2, h2 in enumerate(parse):
            if m2 == m:
                continue
            if m < h:
                if m < m2 < h < h2 or m < h2 < h < m2 or m2 < m < h2 < h or h2 < m < m2 < h:
                    return False
            if h < m:
                if h < m2 < m < h2 or h < h2 < m < m2 or m2 < h < h2 < m or h2 < h < m2 < m:
                    return False
    return True
