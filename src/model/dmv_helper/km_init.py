import numpy as np
from fastNLP import DataSet, DataSetIter
from numpy import ndarray

from src.datamodule.sampler import ConstantTokenNumSampler
from src.model.torch_struct.dmv import HASCHILD, NOCHILD, STOP, GO

harmonic_sum = [0., 1.]


def get_harmonic_sum(n: int):
    global harmonic_sum
    while n >= len(harmonic_sum):
        harmonic_sum.append(harmonic_sum[-1] + 1 / len(harmonic_sum))
    return harmonic_sum[n]


def update_decision(change: ndarray, norm_counter: ndarray, token_array: ndarray, dec_param: ndarray):
    for i in range(token_array.shape[1]):
        pos = token_array[:, i]
        for _direction in (0, 1):
            if change[i, _direction] > 0:
                np.add.at(norm_counter, (pos, _direction, NOCHILD, GO), 1.)
                np.add.at(norm_counter, (pos, _direction, HASCHILD, GO), -1.)
                np.add.at(dec_param, (pos, _direction, HASCHILD, GO), change[i, _direction])
                np.add.at(norm_counter, (pos, _direction, NOCHILD, STOP), -1.)
                np.add.at(norm_counter, (pos, _direction, HASCHILD, STOP), 1.)
                np.add.at(dec_param, (pos, _direction, NOCHILD, STOP), 1.)
            else:
                np.add.at(dec_param, (pos, _direction, NOCHILD, STOP), 1.)


def first_child_update(norm_counter: ndarray, dec_param: ndarray):
    all_param = dec_param.flatten()
    all_norm = norm_counter.flatten()
    mask = (all_param <= 0) | (0 <= all_norm)
    ratio = -all_param / all_norm
    ratio[mask] = 1.
    return np.min(ratio)


def km_init(dataset: DataSet, n_token: int, smooth: float):
    # do not ask why? I do not know more than you.
    dec_param = np.zeros((n_token, 2, 2, 2))
    root_param = np.zeros((n_token,))
    trans_param = np.zeros((n_token, n_token, 2, 2))

    norm_counter = np.full(dec_param.shape, smooth)
    change = np.zeros((max(dataset['seq_len'].content), 2))
    sampler = ConstantTokenNumSampler(dataset['seq_len'].content, 1000000, -1, 0, force_same_len=True)
    data_iter = DataSetIter(dataset, batch_sampler=sampler, as_numpy=True)
    for x, y in data_iter:
        token_array = x['token']
        batch_size, word_num = token_array.shape
        change.fill(0.)
        np.add.at(root_param, (token_array, ), 1. / word_num)
        if word_num > 1:
            for child_i in range(word_num):
                child_sum = get_harmonic_sum(child_i - 0) + get_harmonic_sum(word_num - child_i - 1)
                scale = (word_num - 1) / word_num / child_sum
                for head_i in range(word_num):
                    if child_i == head_i:
                        continue
                    direction = 1 if head_i <= child_i else 0
                    head_pos = token_array[:, head_i]
                    child_pos = token_array[:, child_i]
                    diff = scale / abs(head_i - child_i)
                    np.add.at(trans_param, (head_pos, child_pos, direction), diff)
                    change[head_i, direction] += diff
        update_decision(change, norm_counter, token_array, dec_param)

    trans_param += smooth
    dec_param += smooth
    root_param += smooth

    es = first_child_update(norm_counter, dec_param)
    norm_counter *= 0.9 * es
    dec_param += norm_counter

    root_param_sum = root_param.sum()
    trans_param_sum = trans_param.sum(1, keepdims=True)
    decision_param_sum = dec_param.sum(3, keepdims=True)

    root_param /= root_param_sum
    trans_param /= trans_param_sum
    dec_param /= decision_param_sum

    return np.log(dec_param), np.log(trans_param), np.log(root_param)
