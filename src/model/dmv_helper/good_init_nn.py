# unlike good_init.py, this file contains helpers to initialize nn without dmv.

from typing import List

import numpy as np
from fastNLP.core.field import Padder

from src.model.torch_struct.dmv import LEFT, RIGHT, HASCHILD, NOCHILD, GO, STOP


class LinearPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(r.shape[0] for r in contents)
        batch_size = len(contents)
        out = np.full((batch_size, max_sent_length, *contents[0].shape[1:]), fill_value=self.pad_val, dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            out[b_idx, :sent_len] = rule
        return out


class SquarePadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(r.shape[0] for r in contents)
        batch_size = len(contents)
        out = np.full((batch_size, max_sent_length, max_sent_length, *contents[0].shape[2:]), fill_value=self.pad_val,
                      dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            out[b_idx, :sent_len, :sent_len] = rule
        return out


def generate_rule_1o(heads: List[int]):
    """
    First-order DMV, generate the grammar rules used in the "predicted" parse tree from other parser.
    :param heads: the head of each position
    :return: decision rule
    """
    seq_len = len(heads)
    decision = np.zeros(shape=(seq_len, 2, 2, 2))
    attach = np.zeros(shape=(seq_len, seq_len, 2))
    root = np.zeros(shape=(seq_len,))
    root[heads.index(0)] = 1

    left_most_child = list(range(seq_len))
    right_most_child = list(range(seq_len))
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        elif child < head:
            if child < left_most_child[head]:
                left_most_child[head] = child
        else:
            if child > right_most_child[head]:
                right_most_child[head] = child

    for child, head in enumerate(heads):
        head = head - 1

        if child < head:
            most_child, d = left_most_child, LEFT
        else:
            most_child, d = right_most_child, RIGHT

        valence = NOCHILD if most_child[head] == child else HASCHILD
        decision[head][d][valence][GO] += 1
        if head != -1:
            attach[head][child][valence] += 1

        valence = NOCHILD if left_most_child[child] == child else HASCHILD
        decision[child][LEFT][valence][STOP] += 1

        valence = NOCHILD if right_most_child[child] == child else HASCHILD
        decision[child][RIGHT][valence][STOP] += 1

    return {'dec_rule': decision, 'attach_rule': attach, 'root_rule': root}
