import json
import logging
import os.path
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra import compose
from hydra.utils import HydraConfig, instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import src
from src import datamodule
from src.datamodule import DataModule
from src.pipeline import Pipeline
from src.utility.fn import instantiate_no_recursive
from src.utility.pl_callback import NNICallback

log = logging.getLogger(__name__)


@hydra.main('config', 'config_test')
def test(cfg: DictConfig):
    if (seed := cfg.seed) is not None:
        pl.seed_everything(seed)

    if cfg.pipeline.load_from_checkpoint is None:
        log.warning('Testing a random-initialized model.')

    if (p := cfg.pipeline.load_from_checkpoint) is not None:
        p = Path(p)
        if len(p.parts) >= 2 and p.parts[-2] == 'checkpoint':
            config_folder = p.parents[1] / 'config'
        else:
            config_folder = p.parent / 'config'
        if config_folder.exists():
            # Load saved config.
            # Note that this only load overrides. Inconsistency happens if you change sub-config's file.
            # From Hydra's author:
            # https://stackoverflow.com/questions/67170653/how-to-load-hydra-parameters-from-previous-jobs-without-having-to-use-argparse/67172466?noredirect=1
            log.info('Loading saved overrides')
            original_overrides = OmegaConf.load(config_folder / 'overrides.yaml')
            current_overrides = HydraConfig.get().overrides.task
            # hydra_config = OmegaConf.load(config_folder / 'hydra.yaml')
            config_name = 'config_test'  # hydra_config.hydra.job.config_name
            overrides = original_overrides + current_overrides
            # noinspection PyTypeChecker
            cfg = compose(config_name, overrides=overrides)
            if os.path.exists(config_folder / 'nni.json'):
                with open(config_folder / 'nni.json') as f:
                    nni_overrides = json.load(f)
                    NNICallback.setup_cfg(nni_overrides, cfg)
            log.info(OmegaConf.to_yaml(cfg))

    src.g_cfg = cfg

    trainer: pl.Trainer = instantiate(cfg.trainer)
    src.trainer = trainer

    datamodule: DataModule = instantiate_no_recursive(cfg.datamodule)
    pipeline: Pipeline = instantiate_no_recursive(cfg.pipeline, dm=datamodule)
    output_name = cfg.get('output_name', 'predict')
    datamodule.setup('test')

    trainer.test(pipeline, dataloaders=datamodule.dataloader('train'))
    pipeline.write_prediction(output_name + '_train.conll', 'train', pipeline._test_outputs[0])
    trainer.test(pipeline, dataloaders=datamodule.dataloader('dev'))
    pipeline.write_prediction(output_name + '_dev.conll', 'dev', pipeline._test_outputs[0])
    trainer.test(pipeline, dataloaders=datamodule.dataloader('test'))
    pipeline.write_prediction(output_name + '_test.conll', 'test', pipeline._test_outputs[0])


if __name__ == '__main__':
    test()
