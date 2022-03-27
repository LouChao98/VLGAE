from __future__ import annotations

import json
import os
import os.path
import random
import string
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra import compose
from hydra.utils import HydraConfig, instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import src
from src.datamodule import DataModule
from src.pipeline import Pipeline
from src.utility.fn import instantiate_no_recursive
from src.utility.fn import symlink_force
from src.utility.logger import get_logger_func
from src.utility.pl_callback import BestWatcherCallback
from src.utility.pl_callback import NNICallback

_warn, _info, _debug = get_logger_func('main')


@hydra.main('config', 'config_train')
def train(cfg: DictConfig):
    src.g_cfg = cfg
    _info(f'Working directory: {os.getcwd()}')

    outputs_root = os.path.join(cfg.root, 'outputs')
    if os.path.exists(outputs_root):
        symlink_force(os.getcwd(), os.path.join(outputs_root, '0_latest_run'))

    if cfg.name == '@@@AUTO@@@':
        # In the case we can not set name={hydra:job.override_dirname} in config.yaml, e.g., multirun
        cfg.name = HydraConfig.get().job.override_dirname

    # init multirun
    if (num := HydraConfig.get().job.get('num')) is not None and num > 1:
        # set group in wandb, if use joblib, this will be set from joblib.
        if 'MULTIRUN_ID' not in os.environ:
            os.environ['MULTIRUN_ID'] = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
        if 'logger' in cfg.trainer and 'tags' in cfg.trainer.logger:
            cfg.trainer.logger.tags.append(os.environ['MULTIRUN_ID'])

    if (config_folder := cfg.load_cfg_from_checkpoint) is not None:
        # Load saved config.
        # Note that this only load overrides. Inconsistency happens if you change sub-config's file.
        # From Hydra's author:
        # https://stackoverflow.com/questions/67170653/how-to-load-hydra-parameters-from-previous-jobs-without-having-to-use-argparse/67172466?noredirect=1
        _info('Loading saved overrides')
        config_folder = Path(config_folder)
        original_overrides = OmegaConf.load(config_folder / 'overrides.yaml')
        current_overrides = HydraConfig.get().overrides.task
        # hydra_config = OmegaConf.load(config_folder / 'hydra.yaml')
        config_name = 'conf'  # hydra_config.hydra.job.config_name
        overrides = original_overrides + current_overrides
        # noinspection PyTypeChecker
        cfg = compose(config_name, overrides=overrides)
        if os.path.exists(config_folder / 'nni.json'):
            with open(config_folder / 'nni.json') as f:
                nni_overrides = json.load(f)
                NNICallback.setup_cfg(nni_overrides, cfg)
        _info(OmegaConf.to_yaml(cfg))
        src.g_cfg = cfg

    if (seed := cfg.seed) is not None:
        pl.seed_everything(seed)
        # torch.use_deterministic_algorithms(True)
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    assert not (cfg.pipeline.load_from_checkpoint is not None and cfg.trainer.resume_from_checkpoint is not None), \
        'You should not use load_from_checkpoint and resume_from_checkpoint at the same time.'
    assert not cfg.watch_field.startswith('test/'), 'You should not use test set to tune hparams.'

    trainer: pl.Trainer = instantiate(cfg.trainer)
    src.trainer = trainer
    if 'optimized_metric' in cfg:
        assert any(isinstance(c, BestWatcherCallback) for c in trainer.callbacks)

    datamodule: DataModule = instantiate_no_recursive(cfg.datamodule)
    pipeline: Pipeline = instantiate_no_recursive(cfg.pipeline, dm=datamodule)
    trainer.fit(pipeline, datamodule)

    ckpt_path = "best"
    trainer.test(model=pipeline, datamodule=datamodule, ckpt_path=ckpt_path)

    _info(f'Working directory: {os.getcwd()}')

    # Return metric score for hyperparameter optimization
    callbacks = trainer.callbacks
    for c in callbacks:
        if isinstance(c, BestWatcherCallback):
            if c.best_model_path:
                _info(f'Best ckpt: {c.best_model_path}')
            if 'optimized_metric' in cfg:
                return c.best_model_metric[cfg.optimized_metric]
            break


if __name__ == '__main__':
    train()
