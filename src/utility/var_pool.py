from typing import Callable, List, Union

from fastNLP.core.utils import seq_len_to_mask
from torch import Tensor


class VarPool:
    def __init__(self, **kwargs):
        self._pool = {}
        self._lazy_func = {}
        self._circle_trace = []

        for key, value in kwargs.items():
            self._pool[key] = value

        self.add_lazy('seq_len', 'batch_size', lambda x: len(x))
        self.add_lazy('seq_len', 'max_len', lambda x: max(x))
        self.add_lazy('seq_len', 'num_token', lambda x: sum(x))
        self.add_lazy(['seq_len', 'max_len'], 'mask', lambda x, y: seq_len_to_mask(x, y))

    def add_lazy(self, source: Union[str, List[str]], target: str, func: Callable, overwrite=False):
        assert overwrite or target not in self._lazy_func, f'{target=}'
        if isinstance(source, str):
            source = [source]
        self._lazy_func[target] = (source, func)

    def select(self, mask):
        new_vp = VarPool()
        for key, value in self._pool.items():
            if key in ('batch_size', 'max_len'):
                continue
            if key.endswith('_cpu') or key.endswith('_cuda'):
                continue
            if not isinstance(value, Tensor):
                continue
            new_vp.add_lazy([], key, lambda v=value: v[mask], overwrite=True)
        for key, value in self._lazy_func.items():
            if key not in new_vp._lazy_func and not key.endswith('cuda') and not key.endswith('cpu'):
                new_vp.add_lazy(value[0], key, value[1], overwrite=True)
        return new_vp

    def __getitem__(self, item):
        if item in self._pool:
            return self._pool[item]
        if item in self._lazy_func:
            source, func = self._lazy_func[item]
            self._circle_trace.append(item)
            assert not any(map(lambda s: s in self._circle_trace, source))
            source = [self[s] for s in source]
            self._circle_trace.pop()
            target = func(*source)
            self[item] = target
            return target
        name, device = item.rsplit('_', 1)
        if device in ('cuda', 'cpu'):
            value = self[name].to(device)
            self._pool[item] = value
            return value
        raise KeyError(f'No {item}.')

    def __setitem__(self, key, value):
        self._pool[key] = value
        if isinstance(value, Tensor):
            self.add_lazy(key, key + '_cuda', lambda x: x if x.device.type == 'cuda' else x.cuda())
            self.add_lazy(key, key + '_cpu', lambda x: x if x.device.type == 'cpu' else x.cpu())

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._pool[key] = value

    def __contains__(self, key):
        return key in self._pool or key in self._lazy_func
