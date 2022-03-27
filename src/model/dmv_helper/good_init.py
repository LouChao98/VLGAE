import numpy as np
from fastNLP import DataSet, AutoPadder


from src.model.torch_struct.dmv import HASCHILD, NOCHILD, STOP, GO


def recovery_one(heads):
    left_most = np.arange(len(heads))
    right_most = np.arange(len(heads))
    for idx, each_head in enumerate(heads):
        if each_head in (0, len(heads) + 1):  # skip head is ROOT
            continue
        each_head -= 1
        if idx < left_most[each_head]:
            left_most[each_head] = idx
        if idx > right_most[each_head]:
            right_most[each_head] = idx

    valences = np.empty((len(heads), 2), dtype=np.int)
    head_valences = np.empty(len(heads), dtype=np.int)

    for idx, each_head in enumerate(heads):
        each_head -= 1
        valences[idx, 0] = NOCHILD if left_most[idx] == idx else HASCHILD
        valences[idx, 1] = NOCHILD if right_most[idx] == idx else HASCHILD
        if each_head > idx:  # each_head = -1 `s head_valence is never used
            head_valences[idx] = NOCHILD if left_most[each_head] == idx else HASCHILD
        else:
            head_valences[idx] = NOCHILD if right_most[each_head] == idx else HASCHILD
    return valences, head_valences


def good_init(dataset: DataSet, n_token: int, smooth: float):
    """process all sentences in one batch."""
    max_len = max(dataset['seq_len'].content)
    heads = np.zeros((len(dataset), max_len + 1), dtype=np.int)
    valences = np.zeros((len(dataset), max_len + 1, 2), dtype=np.int)
    head_valences = np.zeros((len(dataset), max_len + 1), dtype=np.int)
    root_counter = np.zeros((n_token,))

    for idx, instance in enumerate(dataset):
        one_heads = np.asarray(instance['arc'])
        one_valences, one_head_valences = recovery_one(one_heads)
        heads[idx, 1:instance['seq_len'] + 1] = one_heads
        valences[idx, 1:instance['seq_len'] + 1] = one_valences
        head_valences[idx, 1:instance['seq_len'] + 1] = one_head_valences

    batch_size, sentence_len = heads.shape
    len_array = np.asarray(dataset['seq_len'].content)
    token_array = AutoPadder()(dataset['token'].content, 'token', np.int, 1)
    batch_arange = np.arange(batch_size)

    batch_trans_trace = np.zeros((batch_size, max_len, max_len, 2, 2))
    batch_dec_trace = np.zeros((batch_size, max_len, max_len, 2, 2, 2))

    for m in range(1, sentence_len):
        h = heads[:, m]
        direction = (h <= m).astype(np.long)
        h_valence = head_valences[:, m]
        m_valence = valences[:, m]
        m_child_valence = h_valence

        len_mask = ((h <= len_array) & (m <= len_array))

        batch_dec_trace[batch_arange, m - 1, m - 1, 0, m_valence[:, 0], STOP] = len_mask
        batch_dec_trace[batch_arange, m - 1, m - 1, 1, m_valence[:, 1], STOP] = len_mask

        head_mask = h == 0
        mask = head_mask * len_mask
        if mask.any():
            np.add.at(root_counter, token_array[:, m - 1], mask)

        head_mask = ~head_mask
        mask = head_mask * len_mask
        if mask.any():
            batch_trans_trace[batch_arange, h - 1, m - 1, direction, m_child_valence] = mask
            batch_dec_trace[batch_arange, h - 1, m - 1, direction, h_valence, GO] = mask

    dec_post_dim = (2, 2, 2)
    dec_counter = np.zeros((n_token, *dec_post_dim))
    index = (token_array.flatten(),)
    np.add.at(dec_counter, index, np.sum(batch_dec_trace, 2).reshape(-1, *dec_post_dim))

    trans_post_dim = (2, 2)
    head_ids = np.tile(np.expand_dims(token_array, 2), (1, 1, max_len))
    child_ids = np.tile(np.expand_dims(token_array, 1), (1, max_len, 1))
    trans_counter = np.zeros((n_token, n_token, *trans_post_dim))
    index = (head_ids.flatten(), child_ids.flatten())
    np.add.at(trans_counter, index, batch_trans_trace.reshape(-1, *trans_post_dim))

    root_counter += smooth
    root_sum = root_counter.sum()
    root_param = np.log(root_counter / root_sum)

    trans_counter += smooth
    trans_sum = trans_counter.sum(axis=1, keepdims=True)
    trans_param = np.log(trans_counter / trans_sum)

    dec_counter += smooth
    dec_sum = dec_counter.sum(axis=3, keepdims=True)
    dec_param = np.log(dec_counter / dec_sum)
    return dec_param, trans_param, root_param
