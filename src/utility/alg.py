from typing import List, Tuple

import torch

from src import INF
from src.utility.fn import pad


def eisner(scores, seq_len):
    batch_size, max_len, _ = scores.shape
    scores = scores.permute(2, 1, 0).contiguous()
    s_i = torch.full_like(scores, -INF)
    s_c = torch.full_like(scores, -INF)
    p_i = scores.new_zeros(max_len, max_len, batch_size).long()
    p_c = scores.new_zeros(max_len, max_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, max_len):
        n = max_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i, r) + C(j, r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j, i) = max(C(i, r) + C(j, r+1) + S(j, i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i, j) = max(C(i, r) + C(j, r+1) + S(i, j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j, i) = max(C(r, i) + I(j, r)), i <= r < j
        cl = stripe(s_c, n, w, dim=0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i, j) = max(I(i, r) + C(r, j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][seq_len.ne(w)] = -INF
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    for i, length in enumerate(seq_len.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(scores.device))
    return pad(predicts, total_length=max_len)


def eisner2o(scores, seq_len):
    r"""
    Second-order Eisner algorithm for projective decoding.
    This is an extension of the first-order one that further incorporates sibling scores into tree scoring.

    References:
        - Ryan McDonald and Fernando Pereira. 2006.
          `Online Learning of Approximate Dependency Parsing Algorithms`_.

    Args:
        scores (~torch.Tensor, ~torch.Tensor):
            A tuple of two tensors representing the first-order and second-order scores repectively.
            The first (``[batch_size, seq_len, seq_len]``) holds scores of all dependent-head pairs.
            The second (``[batch_size, seq_len, seq_len, seq_len]``) holds scores of all dependent-head-sibling triples.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> s_arc = torch.tensor([[[ -2.8092,  -7.9104,  -0.9414,  -5.4360],
                                   [-10.3494,  -7.9298,  -3.6929,  -7.3985],
                                   [  1.1815,  -3.8291,   2.3166,  -2.7183],
                                   [ -3.9776,  -3.9063,  -1.6762,  -3.1861]]])
        >>> s_sib = torch.tensor([[[[ 0.4719,  0.4154,  1.1333,  0.6946],
                                    [ 1.1252,  1.3043,  2.1128,  1.4621],
                                    [ 0.5974,  0.5635,  1.0115,  0.7550],
                                    [ 1.1174,  1.3794,  2.2567,  1.4043]],
                                   [[-2.1480, -4.1830, -2.5519, -1.8020],
                                    [-1.2496, -1.7859, -0.0665, -0.4938],
                                    [-2.6171, -4.0142, -2.9428, -2.2121],
                                    [-0.5166, -1.0925,  0.5190,  0.1371]],
                                   [[ 0.5827, -1.2499, -0.0648, -0.0497],
                                    [ 1.4695,  0.3522,  1.5614,  1.0236],
                                    [ 0.4647, -0.7996, -0.3801,  0.0046],
                                    [ 1.5611,  0.3875,  1.8285,  1.0766]],
                                   [[-1.3053, -2.9423, -1.5779, -1.2142],
                                    [-0.1908, -0.9699,  0.3085,  0.1061],
                                    [-1.6783, -2.8199, -1.8853, -1.5653],
                                    [ 0.3629, -0.3488,  0.9011,  0.5674]]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner2o((s_arc, s_sib), mask)
        tensor([[0, 2, 0, 2]])

    .. _Online Learning of Approximate Dependency Parsing Algorithms:
        https://www.aclweb.org/anthology/E06-1011/
    """

    # the end position of each sentence in a batch
    s_arc, s_sib = scores
    batch_size, max_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0).contiguous()
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0).contiguous()
    s_i = torch.full_like(s_arc, -INF)
    s_s = torch.full_like(s_arc, -INF)
    s_c = torch.full_like(s_arc, -INF)
    p_i = s_arc.new_zeros(max_len, max_len, batch_size).long()
    p_s = s_arc.new_zeros(max_len, max_len, batch_size).long()
    p_c = s_arc.new_zeros(max_len, max_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, max_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = max_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # I(j->i) = max(I(j->r) + S(j->r, i)), i < r < j |
        #               C(j->j) + C(i->j-1))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n + w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, seq_len.new_tensor(0), 0).squeeze(1)
        il_span, il_path = il.permute(2, 0, 1).max(-1)
        s_i.diagonal(-w).copy_(il_span + s_arc.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts + 1)
        # I(i->j) = max(I(i->r) + S(i->r, j), i < r < j |
        #               C(i->i) + C(j->i+1))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n + w)], n, w)
        ir[0] = -INF
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        ir_span, ir_path = ir.permute(2, 0, 1).max(-1)
        s_i.diagonal(w).copy_(ir_span + s_arc.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        slr_span, slr_path = slr.permute(2, 0, 1).max(-1)
        # S(j, i) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr_span)
        p_s.diagonal(-w).copy_(slr_path + starts)
        # S(i, j) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr_span)
        p_s.diagonal(w).copy_(slr_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        # disable multi words to modify the root
        s_c[0, w][seq_len.ne(w)] = -INF
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_s, p_c, heads, i, j, flag):
        if i == j:
            return
        if flag == 'c':
            r = p_c[i, j]
            backtrack(p_i, p_s, p_c, heads, i, r, 'i')
            backtrack(p_i, p_s, p_c, heads, r, j, 'c')
        elif flag == 's':
            r = p_s[i, j]
            i, j = sorted((i, j))
            backtrack(p_i, p_s, p_c, heads, i, r, 'c')
            backtrack(p_i, p_s, p_c, heads, j, r + 1, 'c')
        elif flag == 'i':
            r, heads[j] = p_i[i, j], i
            if r == i:
                r = i + 1 if i < j else i - 1
                backtrack(p_i, p_s, p_c, heads, j, r, 'c')
            else:
                backtrack(p_i, p_s, p_c, heads, i, r, 'i')
                backtrack(p_i, p_s, p_c, heads, r, j, 's')

    preds = []
    p_i = p_i.permute(2, 0, 1).cpu()
    p_s = p_s.permute(2, 0, 1).cpu()
    p_c = p_c.permute(2, 0, 1).cpu()
    for i, length in enumerate(seq_len.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_s[i], p_c[i], heads, 0, length, 'c')
        preds.append(heads.to(seq_len.device))

    return pad(preds, total_length=max_len)


def eisner2o_grand(scores, lens):
    # lens should be without ROOT.

    def stripe_02(x, n, w, offset=(0, 0, 0), dim=1):
        # x: [seq_len, seq_len, seq_len, ...]
        assert x.is_contiguous(), 'x must be contiguous, or write on new view will lost.'
        x, seq_len = x.contiguous(), x.size(1)
        stride, numel1, numel2 = list(x.stride()), x[0, 0].numel(), x[0, 0, 0].numel()
        stride[0] = (seq_len + 1) * numel1 + numel2
        stride[2] = (1 if dim == 1 else seq_len) * numel2
        del stride[1]
        return x.as_strided(size=(n, w, *x.shape[3:]),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel1 + offset[2] * numel2)

    # the end position of each sentence in a batch
    s_arc, s_grand = scores
    batch_size, seq_len, _ = s_arc.shape

    # [seq_len, seq_len, batch_size], [head, child, batch]
    s_a = s_arc.permute(2, 1, 0).contiguous()
    # [seq_len, seq_len, seq_len, batch_size], [grand, parent, child, batch]
    s_g = s_grand.permute(1, 2, 3, 0).contiguous()

    s_i = torch.full_like(s_a, -INF)
    s_c = torch.full_like(s_g, -INF)  # head, end, dep, batch
    s_c.diagonal().permute(2, 0, 1).diagonal().fill_(0)
    s_i = s_i.contiguous()
    s_c = s_c.contiguous()

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w

        # i(i,j) = arc(i,j) MUL
        #               SUM_r  c(i,r) MUL
        #                   SUM_k c(r+1,j,k) MUL g(i,j,k)
        ilr = stripe(s_c, n, w).clone().logsumexp(2) \
              + (s_g.diagonal(w).permute(2, 0, 1).unsqueeze(1) + stripe(s_c, n, w, (w, 1))).logsumexp(dim=2)
        s_i.diagonal(w).copy_(ilr.permute(2, 0, 1).logsumexp(-1) + s_a.diagonal(w))

        ilr = (s_g.diagonal(-w).permute(2, 0, 1).unsqueeze(1) + stripe(s_c, n, w)).logsumexp(dim=2) \
              + stripe(s_c, n, w, (w, 1)).clone().logsumexp(2)
        s_i.diagonal(-w).copy_(ilr.permute(2, 0, 1).logsumexp(-1) + s_a.diagonal(-w))

        # c(i,j,r) = i(i,r) MUL
        #               SUM_k c(r,j,k) MUL g(i,r,k)
        cl = stripe(s_i, n, w, (0, 1)) + (stripe(s_g, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)).logsumexp(dim=2)
        stripe_02(s_c, n, w, offset=(0, w, 1)).copy_(cl)

        cr = (stripe(s_g, n, w, (w, 0)) + stripe(s_c, n, w, (0, 0), 0)).logsumexp(dim=2) + stripe(s_i, n, w, (w, 0))
        stripe_02(s_c, n, w, (w, 0, 0)).copy_(cr)

        s_c[0, w][:, lens.ne(w)] = -INF

    s_c = s_c.logsumexp(2)
    # return s_c[0].gather(0, lens.unsqueeze(0)).sum()
    return s_c[0].gather(0, lens.unsqueeze(0)).squeeze(0)


def isprojective(sequence):
    r"""
    Checks if a dependency tree is projective.
    This also works for partial annotation.

    Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
    which are hard to detect in the scenario of partial annotation.

    Args:
        sequence (list[int]):
            A list of head indices.

    Returns:
        ``True`` if the tree is projective, ``False`` otherwise.

    Examples:
        >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
        False
        >>> CoNLL.isprojective([3, -1, 2])
        False
    """

    pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True


def istree(sequence, proj=False, multiroot=False):
    r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

    if proj and not isprojective(sequence):
        return False
    n_roots = sum(head == 0 for head in sequence)
    if n_roots == 0:
        return False
    if not multiroot and n_roots > 1:
        return False
    if any(i == head for i, head in enumerate(sequence, 1)):
        return False
    return next(tarjan(sequence), None) is None


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""Returns a diagonal stripe of the tensor.
    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.
    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    seq_len = x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def find_dep_boundary(heads: List[int], head_in_span) -> List[Tuple[int, int, int]]:
    left_bd = [i for i in range(len(heads))]
    right_bd = [i + 1 for i in range(len(heads))]

    for child_idx, head_idx in enumerate(heads):
        if head_idx > 0:
            if left_bd[child_idx] < left_bd[head_idx - 1]:
                left_bd[head_idx - 1] = left_bd[child_idx]

            elif child_idx > right_bd[head_idx - 1] - 1:
                right_bd[head_idx - 1] = child_idx + 1
                while head_idx != 0:
                    if heads[head_idx - 1] > 0 and child_idx + 1 > right_bd[heads[head_idx - 1] - 1]:
                        right_bd[heads[head_idx - 1] - 1] = child_idx + 1
                        head_idx = heads[head_idx - 1]
                    else:
                        break

    # (head_word_idx, left_bd_idx, right_bd_idx)
    triplet = []
    # head index should add1, as the root token would be the first token. But not here.
    # [ )  left bdr, right bdr.
    for (parent, left_bdr, right_bdr) in (zip(heads, left_bd, right_bd)):
        if parent != 0:
            if head_in_span:
                triplet.append((left_bdr, right_bdr, parent - 1))
            else:
                triplet.append((left_bdr, right_bdr, heads[parent - 1]))

    return triplet


def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.
    Args:
        sequence (list):
            List of head indices.
    Yields:
        A list of indices making up a SCC. All self-loops are ignored.
    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = -INF(low[i], low[j])
            elif onstack[j]:
                low[i] = -INF(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s):
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding :cite:`mcdonald-etal-2005-non`.
    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in :cite:`mcdonald-etal-2005-non`.
    Notes:
        The algorithm does not guarantee to parse a single-root tree.
    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.
    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.
    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    """

    s[0, 1:] = -INF
    # prevent self-loops
    s.diagonal()[1:].fill_(-INF)
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads co-INFg from the cycle
    # len(y) denotes heads co-INFg from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads co-INFg from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree


def mst(scores, mask, multiroot=False):
    r"""
    MST algorithm for decoding non-projective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.
    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.
    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.
    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.
    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = -INF
        >>> scores.diagonal(0, 1, 2)[1:].fill_(-INF)
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length + 1, :length + 1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = -INF
            s = s.index_fill(1, torch.tensor(0), -INF)
            for root in roots:
                s[:, 0] = -INF
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)
