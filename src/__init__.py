import logging
import os
from typing import Optional, Mapping

import numpy as np
import pytorch_lightning
import torch
from easydict import EasyDict
from hydra._internal.utils import is_under_debugger as _is_under_debugger
from hydra.utils import HydraConfig
from omegaconf import ListConfig, OmegaConf

from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func('src')

g_cfg = EasyDict({
    'MANUAL': 1,
})  # globel configuration obj
trainer: Optional[pytorch_lightning.Trainer] = None
debugging = False

# >>> setup logger

pl_logger = logging.getLogger('lightning')
pl_logger.propagate = False

fastnlp_logger = logging.getLogger('fastNLP')
fastnlp_logger.propagate = False

wandb_logger = logging.getLogger('wandb')
# wandb_logger.propagate = False

# >>> setup OmegaConf

# OmegaConf.register_new_resolver('in', lambda x, y: x in y)
OmegaConf.register_new_resolver('lang', lambda x: x.split('_')[0])
OmegaConf.register_new_resolver('last', lambda x: x.split('/')[-1])
OmegaConf.register_new_resolver('div2', lambda x: x // 2)
# OmegaConf.register_new_resolver('cat', lambda x, y: x + y)

_hit_debug = True


def is_under_debugger():
    if os.environ.get('DEBUG_MODE', '').lower() in ('true', 't', '1', 'yes', 'y'):
        result = True
    else:
        result = _is_under_debugger()
    global _hit_debug, debugging
    if result and _hit_debug:
        _warn("Debug mode.")
        _hit_debug = False
        debugging = True
    return result


OmegaConf.register_new_resolver('in_debugger', lambda x, default=None: x if is_under_debugger() else default)


def path_guard(x: str):
    x = x.split(',')
    x.sort()
    x = '_'.join(x)
    x = x.replace('/', '-')
    x = x.replace('=', '-')
    return x[:240]


OmegaConf.register_new_resolver('path_guard', path_guard)


def half_int(x):
    assert x % 2 == 0
    return x // 2


OmegaConf.register_new_resolver('half_int', half_int)


def name_guard(fallback):
    try:
        return HydraConfig.get().job.override_dirname
    except ValueError as v:
        if 'HydraConfig was not set' in str(v):
            return fallback
        raise v


OmegaConf.register_new_resolver('name_guard', name_guard)


def choose_accelerator(gpus):
    if isinstance(gpus, int):
        return 'ddp' if gpus > 1 else None
    elif isinstance(gpus, str):
        return 'ddp' if len(gpus.split(',')) > 1 else None
    elif isinstance(gpus, (list, ListConfig)):
        return 'ddp' if len(gpus) > 1 else None
    elif gpus is None:
        return None
    raise ValueError(f'Unrecognized {gpus=} ({type(gpus)})')


OmegaConf.register_new_resolver('accelerator', choose_accelerator)


# >>> setup inf

INF = 1e20


def setup_inf(v):
    global INF
    import src.model.torch_struct as stt
    INF = v
    stt.semirings.semirings.NEGINF = -INF


setup_inf(1e20)


# pl patch

def _extract_batch_size(batch):
    if isinstance(batch, torch.Tensor):
        yield batch.shape[0]
    elif isinstance(batch, np.ndarray):
        yield batch.shape[0]
    elif isinstance(batch, str):
        yield len(batch)
    elif isinstance(batch, Mapping):
        for sample in batch:
            yield from _extract_batch_size(sample)
    else:
        x, y = batch
        yield len(x['id'])


from pytorch_lightning.utilities import data as pludata

pludata._extract_batch_size = _extract_batch_size
