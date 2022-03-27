import errno
import logging
import os
from functools import wraps
from typing import Any, Dict, Callable, Optional, Iterator

from hydra.utils import instantiate
from omegaconf import ListConfig, DictConfig
from pytorch_lightning import Trainer
from torch import Tensor


def not_distributed_guard():
    import torch.distributed as dist
    assert not dist.is_initialized()


def endless_iter(i: Iterator, shuffle: Optional[Callable] = None, inplace_shuffle: Optional[Callable] = None):
    while True:
        if shuffle is not None:
            i = shuffle(i)
        if inplace_shuffle is not None:
            inplace_shuffle(i)
        for x in i:
            yield x


def dict_apply(d: Dict[Any, Any], func=None, key_func=None):
    assert func or key_func
    if func is None:
        return {key_func(key): value for key, value in d.items()}
    elif key_func is None:
        return {key: func(value) for key, value in d.items()}
    return {key_func(key): func(value) for key, value in d.items()}


def hydra_instantiate_func_helper(func):
    """convert func() to func()()"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        def mid():
            return func(*args, **kwargs)

        return mid

    return wrapper


def reduce_loss(mode, loss, num_token, num_sentence) -> Tensor:
    if not isinstance(loss, list):
        loss, num_token, num_sentence = [loss], [num_token], [num_sentence]
    assert len(loss) >= 1, 'Nothing to reduce. You should handle this error outside this function.'
    if mode == 'token':
        # average over tokens in a batch
        return sum(loss) / (sum(num_token) + 1e-12)
    elif mode == 'sentence':
        # first average over tokens in a sentence.
        # then average sentences over a batch
        # return sum((l / s).sum() for l, s in zip(loss, seq_len)) / (sum(len(s) for s in seq_len))
        raise NotImplementedError('Deprecated')
    elif mode == 'batch':
        # average over sentences in a batch
        return sum(loss) / (sum(num_sentence) + 1e-12)
    elif mode == 'sum':
        return sum(loss)
    raise ValueError


def split_list(raw, size):
    out = []
    offset = 0
    for s in size:
        out.append(raw[offset: offset + s])
        offset += s
    assert offset == len(raw)
    return out


def instantiate_no_recursive(*args, **kwargs):
    return instantiate(*args, **kwargs, _recursive_=False)


def get_coeff_iter(command, idx_getter=None, validator=None):
    # 1. not (list, tuple, ListConfig): constant alpha
    # 2. List[str]: str should be [value]@[epoch]. eg "[0@0, 0.5@100]". Linearly to value at epoch.
    #               the first term must be @0 (from the beginning)
    if not isinstance(command, (list, tuple, ListConfig)):
        # -123456789 is never reached, so it is endless
        assert command != -123456789
        return iter(lambda: command, -123456789)

    if idx_getter is None:
        _i = 0

        def auto_inc():
            nonlocal _i
            i, _i = _i, _i + 1
            return i

        idx_getter = auto_inc

    def calculate_alpha(value_and_step):
        prev_v, prev_s = value_and_step[0].split('@')
        prev_v, prev_s = float(prev_v), int(prev_s)
        assert prev_s == 0, 'the first step must be 0'
        idx = idx_getter()
        for i in range(1, len(value_and_step)):
            next_v, next_s = value_and_step[i].split('@')
            next_v, next_s = float(next_v), int(next_s)
            rate = (next_v - prev_v) / (next_s - prev_s)
            while idx <= next_s:
                value = prev_v + rate * (idx - prev_s)
                if validator is not None:
                    assert validator(value), f'Bad value in coeff_iter. Get {value}.'
                yield value
                idx = idx_getter()
            prev_v, prev_s = next_v, next_s
        while True:
            yield prev_v

    return iter(calculate_alpha(command))


def instantiate_trainer(callbacks=None, **kwargs):
    if callbacks is not None:
        NoneType = type(None)
        callbacks = list(filter(lambda x: not isinstance(x, (dict, DictConfig, NoneType)), callbacks.values()))
    return Trainer(callbacks=callbacks, **kwargs)


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors) for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def filter_list(data, mask):
    if isinstance(mask[0], list):
        out = []
        for subdata, submask in zip(data, mask):
            out.append(filter_list(subdata, submask))
        return out
    elif isinstance(mask[0], int):
        return [subdata for subdata, submask in zip(data, mask) if submask]
    raise ValueError(f'Bad mask value: {mask}')


def draw_att(data: Tensor, path=None):
    assert data.ndim == 2
    import seaborn as sns
    import matplotlib.pyplot as plt
    data = data.detach().cpu().numpy()
    sns.heatmap(data=data, center=0, mask=data < -100)
    if path:
        plt.savefig(path)
    else:
        plt.show()


def merge_outputs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        adata, bdata = a[key], b[key]
        if len(adata) > len(bdata):
            bdata.extend([None] * (len(adata) - len(bdata)))
        else:
            adata.extend([None] * (len(bdata) - len(adata)))
        a[key] = [ai if ai is not None else bi for ai, bi in zip(a[key], b[key])]
    return a


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def listloggers():
    rootlogger = logging.getLogger()
    print(rootlogger)
    for h in rootlogger.handlers:
        print('     %s' % h)

    for nm, lgr in logging.Logger.manager.loggerDict.items():
        print('+ [%-20s] %s ' % (nm, lgr))
        if not isinstance(lgr, logging.PlaceHolder):
            for h in lgr.handlers:
                print('     %s' % h)