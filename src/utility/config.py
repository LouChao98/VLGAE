import inspect
from dataclasses import dataclass

from omegaconf import MISSING, DictConfig

from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func('config')


@dataclass
class Config:
    @classmethod
    def build(cls, env, ignore_unknown=False, allow_missing=None):
        if isinstance(env, (dict, DictConfig)):
            if 'cfg' in env and isinstance(env['cfg'], cls):
                breakpoint()
                return env['cfg']

            matched = {k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
            unmatched = {k: env[k]
                         for k in env.keys() - matched.keys()
                         if not k.startswith('n_')}  # n_* will be set automatically
            if unmatched and not ignore_unknown:
                raise ValueError(f'Unrecognized cfg: {unmatched}')
            # noinspection PyArgumentList
            cfg = cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})

            allow_missing = allow_missing or set()
            for key, value in cfg.__dict__.items():
                if not key.startswith('_') and key not in allow_missing:
                    assert value is not MISSING, f'{key} is MISSING.'

            if ignore_unknown:
                return cfg, unmatched
            return cfg
        elif isinstance(env, cls):
            return env
        raise TypeError

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            _warn(f"Adding new key: {key}")
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)