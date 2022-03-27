from __future__ import annotations

import os
import pprint
import sys
from typing import Any, Dict, Optional, Union, Iterable, TYPE_CHECKING

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.overrides.distributed import LightningDistributedModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from tqdm import tqdm

import src
from src.utility.fn import dict_apply, listloggers, symlink_force
from src.utility.logger import get_logger_func

if TYPE_CHECKING:
    from src.runner import BasicRunner

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]
GenDict = (dict, DictConfig)
GenList = (list, ListConfig)

_warn, _info, _debug = get_logger_func('callback')


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = 'gradients', log_freq: int = 100):
        self.log_mode = log
        self.log_freq = log_freq

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.is_global_zero:
            logger = WatchModelWithWandb.get_wandb_logger(trainer=trainer)
            if logger is not None and self.log_mode != 'none':
                if isinstance(trainer.model, LightningDistributedModule):
                    model = trainer.model.module
                else:
                    model = trainer.model
                logger.watch(model=model, log=self.log_mode, log_freq=self.log_freq)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            if os.path.exists('config/config.yaml'):
                wandb.save('config/config.yaml')
            wandb.finish()

    @staticmethod
    def get_wandb_logger(trainer: pl.Trainer) -> Optional[WandbLogger]:
        if trainer.logger is None:
            return None
        logger = trainer.logger
        if isinstance(logger, Iterable):
            for lg in logger:
                if isinstance(lg, WandbLogger):
                    logger = lg
            return logger
        return logger if isinstance(logger, WandbLogger) else None


class MyProgressBar(TQDMProgressBar):
    """Only one, short, ascii"""

    def __init__(self, refresh_rate: int, process_position: int):
        refresh_rate = 0 if _detect_nni() is not None else refresh_rate
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(desc='Validation sanity check',
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=False,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(desc='Training',
                   initial=self.train_batch_idx,
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def init_predict_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = tqdm(desc="Predicting",
                   initial=self.train_batch_idx,
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = tqdm(desc='Testing',
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'[{trainer.current_epoch + 1}] train')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'[{trainer.current_epoch + 1}]   val')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.main_progress_bar.set_description(f'[{trainer.current_epoch + 1}] train')

    def print(
            self, *args, sep: str = " ", end: str = os.linesep, file=None, nolock: bool = False
    ):
        _info(sep.join(map(str, args)))

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        metrics = super().get_metrics(trainer, pl_module)
        if 'v_num' in metrics:
            del metrics['v_num']
        return metrics


class LearningRateMonitorWithEarlyStop(LearningRateMonitor):
    """earlystop if lr is too small"""

    def __init__(self, logging_interval=None, log_momentum=False, minimum_lr=None):
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)
        self.minimum_lr = minimum_lr
        self.fully_initialized = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule, unused=None):
        main_lr = max(self._extract_stats(trainer, 'any').values())
        if main_lr < self.minimum_lr and self.fully_initialized:
            trainer.should_stop = True
        elif main_lr >= self.minimum_lr:
            # skip the increasing stage
            self.fully_initialized = True


class BestWatcherCallback(ModelCheckpoint):
    """A model_checkpoint with more info about the best."""

    def __init__(
            self,
            monitor: str,  # the metric to monitor
            mode: str,  # min or max
            hint: bool = True,  # show a hint in stdout when a new best
            save=False,  # save best model
            write: str = 'none',  # write predictions
            report: bool = True,  # report the best at the end of training
    ):
        assert save is False or isinstance(save, GenDict)
        assert write in ('none', 'new', 'always')
        if save is False:
            save = {}
        super().__init__(monitor=monitor, mode=mode, dirpath=save.get('dirpath'), filename=save.get('filename'))
        self.hint = hint  # hit in logging (not logger).
        self.save = save  # save model checkpoint. dict or False.
        self.write = write  # write predict.
        self.report = report  # report when end or quit to logger and console.

        self.best_model_metric = None
        self.best_model_score = self.kth_value

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: BasicRunner):
        self.report = self.report and trainer.logger is not None

    def on_validation_end(self, trainer: Trainer, pl_module: BasicRunner):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if (trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.sanity_checking  # don't save anything during sanity check
                or self._last_global_step_saved == global_step  # already saved at the last step
        ):  # noqa
            return

        self._validate_monitor_key(trainer)
        metric = self._monitor_candidates(trainer, epoch, global_step)
        current = metric.get(self.monitor)
        is_best = self.check_metric(trainer, current)

        if self.write == 'always' or (self.write == 'new' and is_best):
            self.write_prediction(pl_module, epoch)

        if self.save:
            super().on_validation_end(trainer, pl_module)

        if is_best:
            self.best_model_metric = metric
            self.best_model_score = current
            if self.hint:
                self.do_hint(epoch)
            if self.report:
                trainer.logger.log_metrics({f'best/{k[5:]}': v
                                            for k, v in metric.items() if k.startswith('test/')}, global_step)

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        # this will de called even when KeyboardInterrupt
        if self.report:
            metric = dict_apply(self.best_model_metric, lambda x: x.item() if isinstance(x, Tensor) else x)
            _info(f'Best: {pprint.pformat(metric)}')

    # def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    #     outputs = pl_module._predict_outputs
    #     pl_module.write_prediction('predict', outputs[0])

    def check_metric(self, trainer, current):
        if current is None:
            return False

        if not isinstance(current, torch.Tensor):
            _warn(
                f'{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly.'
                f' HINT: check the value of {self.monitor} in your validation loop',
                RuntimeWarning,
            )
            current = torch.tensor(current)

        if current.isnan():
            raise RuntimeError('Moniter is Nan.')

        monitor_op = {'min': torch.lt, 'max': torch.gt}[self.mode]
        is_best = monitor_op(current, self.best_model_score)
        is_best = trainer.training_type_plugin.reduce_boolean_decision(is_best)
        return is_best

    def do_hint(self, epoch):
        _info(f'[{epoch + 1}] New best.')

    def save_checkpoint(self, trainer: "pl.Trainer"):
        if self.save and (trainer.current_epoch >= self.save['start_patience']):
            super().save_checkpoint(trainer)
            if self.best_model_path:
                symlink_force(self.best_model_path, os.getcwd() + '/checkpoint/best.ckpt')

    def write_prediction(self, pl_module: BasicRunner, epoch):
        if not hasattr(pl_module, '_val_outputs'):
            raise MisconfigurationException('Can not find _val_outputs.'
                                            'This is required because lightning prevent me from getting outputs.')
        outputs = pl_module._val_outputs
        assert len(outputs) in (1, 2)
        # pl_module.write_prediction(f'dev.predict.{epoch}.txt', 'dev', outputs[0])
        # if len(outputs) == 2:
        #     pl_module.write_prediction(f'test.predict.{epoch}.txt', 'test', outputs[1])
        pl_module.write_prediction(f'dev.predict.txt', 'dev', outputs[0])
        if len(outputs) == 2:
            pl_module.write_prediction(f'test.predict.txt', 'test', outputs[1])

    @classmethod
    def _format_checkpoint_name(
            cls,
            filename,
            metrics,
            prefix="",
            auto_insert_metric_name=True,
    ) -> str:
        filename = ModelCheckpoint._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)
        filename = filename.replace('/', '_')
        return filename


def _detect_nni():
    try:
        import logging
        n_handlers = len(logging.getLogger('').handlers)
        import nni
        if nni.get_experiment_id() != 'STANDALONE':
            return nni
        # listloggers()
        logging.getLogger('').handlers = logging.getLogger('').handlers[:n_handlers]
    except ImportError:
        return None
    return None


class NNICallback(Callback):
    # intermediate_result = test_{watch_field} at each step
    # final_result = test_{watch_field} at best val_{watch_field}
    def __init__(self, monitor, init_patience=0):
        super().__init__()
        self.is_activated = False
        self.init_patience = init_patience
        if (nni := _detect_nni()) is not None:
            params = nni.get_next_parameter()
            self.setup_cfg(params, src.g_cfg)
            self.is_activated = True
            self.nni = nni
            self.best = (-1, -1)
            self.watch_field = monitor

    @staticmethod
    def setup_cfg(params, target):
        for key, value in params.items():
            s = target
            key = key.split('.')
            for k in key[:-1]:
                if isinstance(s, GenList):
                    k = int(k)
                s = s[k]
            s[key[-1]] = value

    def on_save_checkpoint(self, trainer, pl_module, callback_state):
        return {}

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        raise NotImplementedError('You should not reload this callback')

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.is_activated or trainer.current_epoch < self.init_patience:
            return
        current_val, current_test = self._get_metric(trainer)
        if current_val > self.best[0]:
            self.best = (current_val, current_test)
        self.nni.report_intermediate_result(current_val)

    def on_fit_end(self, trainer, pl_module):
        if self.is_activated:
            self.nni.report_final_result(self.best[0])

    def _get_metric(self, trainer):
        logs = trainer.logger_connector.callback_metrics
        current_val, current_test = logs[self.watch_field], logs[self.test_watch_field]
        return current_val.item(), current_test.item()
