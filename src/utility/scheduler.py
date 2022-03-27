# noinspection PyUnresolvedReferences
import logging
# noinspection PyUnresolvedReferences
import math

# noinspection PyUnresolvedReferences
import numpy as np
from torch.optim import lr_scheduler
# noinspection PyUnresolvedReferences
from transformers import (get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup)

from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func('scheduler')


def get_exponential_lr_scheduler(optimizer, gamma, **kwargs):
    if isinstance(gamma, str):
        gamma = eval(gamma)
        _debug(f'gamma is converted to {gamma} {type(gamma)}')
    kwargs['gamma'] = gamma
    return lr_scheduler.ExponentialLR(optimizer, **kwargs)


def get_reduce_lr_on_plateau_scheduler(optimizer, **kwargs):
    return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


def get_lr_lambda_scheduler(optimizer, lr_lambda, **kwargs):
    if isinstance(lr_lambda, str):
        lr_lambda = eval(lr_lambda)
        _debug(f'lr_lambda is converted to {lr_lambda} {type(lr_lambda)}')
    kwargs['lr_lambda'] = lr_lambda
    return lr_scheduler.LambdaLR(optimizer, **kwargs)
