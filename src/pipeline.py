import math
import re
from copy import deepcopy
from functools import reduce
from typing import TYPE_CHECKING, Any, Optional, List, Union, Dict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import Tensor

import src
from src.datamodule.datamodule import DataModule
from src.utility.fn import dict_apply, get_coeff_iter, instantiate_no_recursive, reduce_loss, merge_outputs
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool

if TYPE_CHECKING:
    from src.model import ModelBase

AnyDict = Dict[str, Any]
_warn, _info, _debug = get_logger_func('runner')


class Pipeline(pl.LightningModule):

    def __init__(self,
                 dm: DataModule,
                 loss_reduction_mode: str = 'token',
                 load_from_checkpoint: str = None):
        self.trainer: pl.Trainer
        self.dm: DataModule = dm
        super().__init__()

        assert loss_reduction_mode in ('token', 'batch', 'sum')
        self.loss_reduction_mode = loss_reduction_mode
        self.checkpoint_path = load_from_checkpoint

        self.model = None
        self.metric = None
        self._has_setup = False
        self._init_scheduler_when_running = None
        self._val_outputs = None  # we can not pass outputs to callbacks directly
        self._test_outputs = None
        self.save_hyperparameters(OmegaConf.to_container(src.g_cfg))

    def setup(self, stage: Optional[str] = None) -> None:
        if self._has_setup: return
        if self.trainer.precision == 16:
            src.setup_inf(1e4)

        with open_dict(src.g_cfg.model):  # setup n_words, n_tag, n_rel ...
            src.g_cfg.model = OmegaConf.merge(src.g_cfg.model, self.dm.get_vocab_count())

        self.model: ModelBase = instantiate(src.g_cfg.model)
        self.model.setup(self.dm)
        self.metric = torch.nn.ModuleList([
            instantiate(src.g_cfg.metric, extra_vocab=self.dm.vocabs),
            instantiate(src.g_cfg.metric, extra_vocab=self.dm.vocabs)
        ])
        if self.__class__ is Pipeline:
            # workaround of loading when use setup.
            # setup is called by trainer automatically, and before setup(), there is no model.
            self._has_setup = True
            if self.checkpoint_path:
                _info(f'Loading model from {self.checkpoint_path}')
                self.load_model_inplace(self.checkpoint_path)
                if src.trainer.training:
                    # src.trainer.test(self, datamodule=self.dm)
                    src.trainer.state.fn = TrainerFn.FITTING
                    src.trainer.state.status = TrainerStatus.RUNNING
                    src.trainer.training = True
        # src.trainer.test(self, datamodule=self.dm)
        # exit(0)

    def forward(self, x, seq_len):
        score = self.model(x, seq_len)
        predict = self.model.decode(score, seq_len)
        return predict

    def on_train_start(self):
        self.model.normalize_embedding('begin')
        if (scheduler_cfg := self._init_scheduler_when_running) is not None:
            # setup scheduler according to len(dataloader)
            n_batches = math.ceil(len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches)
            resolved_scheduler_args = {}
            for key, value in scheduler_cfg.args.items():
                if isinstance(value, str) and value.endswith(' epoch'):
                    value = int(value.split()[0]) * n_batches
                resolved_scheduler_args[key] = value
            scheduler = instantiate_no_recursive(resolved_scheduler_args, optimizer=self.trainer.optimizers[0])
            scheduler = {
                'scheduler': scheduler,
                'interval': scheduler_cfg.interval,
                'frequency': scheduler_cfg.frequency,
                'monitor': src.g_cfg.watch_field,
                'strict': True
            }
            self.trainer.lr_schedulers = self.trainer.configure_schedulers([scheduler], None, False)

    def on_train_epoch_start(self):
        self.model.normalize_embedding('epoch')

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.model.normalize_embedding('batch')

    def training_step(self, batch, batch_idx):
        x, y = batch['train']
        vp = self.model.set_varpool(VarPool(**x, **y))
        self.log('bs', float(len(x['id'])), prog_bar=True, logger=False)

        score = self.model(x, vp)
        loss = self.model.loss(score, y, vp)

        with torch.no_grad():
            detailed_loss = dict_apply(loss[1], lambda l: self.reduce_loss(l, vp))
            self.log_dict(detailed_loss, prog_bar=True, logger=False)
            self.log_dict({f'train/{k}': v for k, v in detailed_loss.items()})
        loss = self.reduce_loss(loss[0], vp)
        self.log('train/sup_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.metric[0].reset()
        self.metric[1].reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        vp = self.model.set_varpool(VarPool(**{**x, **y}))
        score = self.model(x, vp)
        predict = self.model.decode(score, vp)
        loss = self.model.loss(score, y, vp)
        loss = self.reduce_loss(loss[0], vp).item()
        mask = vp.mask
        if 'punct_mask' in x:
            mask = x['punct_mask']
        self.metric[dataloader_idx].update(predict, y, mask)
        return {'loss': loss, 'id': x['id'], 'seq_len': x['seq_len'], 'predict': predict}

    def validation_epoch_end(self, outputs: Union[List[AnyDict], List[List[AnyDict]]]):
        epoch = self.current_epoch + (0 if self.trainer.sanity_checking else 1)

        val_result = self.metric[0].compute()
        val_result['loss'] = sum(batch['loss'] for batch in outputs) / (len(outputs) + 1e-9)
        self.log_dict({'val/' + k: v for k, v in val_result.items()})
        self.print(f'[{epoch}] VAL \t' + '\t'.join(f'{k}={f"{v:.3f}":.6}' for k, v in val_result.items()))

        self._val_outputs = [outputs]

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_result = self.metric[0].compute()
        test_result['loss'] = sum(item['loss'] for item in outputs) / len(outputs)
        if self.logger is not None:
            self.logger.log_metrics(dict_apply(test_result, lambda x: x.item() if isinstance(x, Tensor) else x), 0)
        self.log_dict(test_result)
        self._test_outputs = [outputs]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        x, y = batch
        vp = self.model.set_varpool(VarPool(**x))
        score = self.model(x, vp)
        predict = self.model.decode(score, vp)
        return {'predict': predict}

    def configure_optimizers(self):
        optimizer_cfg = src.g_cfg.optimizer
        if optimizer_cfg.get('groups') is None or len(optimizer_cfg.groups) == 0:
            params = self.model.parameters()
        else:
            """ process groups, config should at least contains `pattern: <str>` for each group for filtering params.
            pattern is a regex, start from childrens of the model. 
            e.g., ^embedding.bert will select all params in the bert.
            fallback params are grouped into the least group.
            """
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.model.named_parameters():
                matches = [i for i, g in enumerate(optimizer_cfg.groups) if re.match(g.pattern, name)]
                if len(matches) > 1:
                    _warn(f'{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}')
                if len(matches) > 0:
                    _debug(f'{name} match {optimizer_cfg.groups[matches[0]].pattern}.')
                    params[matches[0]].append(p)
                else:
                    _debug(f'{name} match defaults.')
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    _warn(f'Nothing matches {optimizer_cfg.groups[i].pattern}')
            params = [{'params': p, **optimizer_cfg.groups[i]} for i, p in enumerate(params) if len(p) > 0]
            params.append({'params': default_group})

        optimizer = instantiate(optimizer_cfg.args, params=params, _convert_='all')

        if (scheduler_cfg := src.g_cfg.get('scheduler')) is None:
            return optimizer

        if scheduler_cfg.get('init_when_running'):
            self._init_scheduler_when_running = scheduler_cfg
            scheduler_cfg = deepcopy(scheduler_cfg)  # create a fake scheduler to make lr_monitor work
            for key in scheduler_cfg.args:
                if isinstance(scheduler_cfg.args[key], str) and scheduler_cfg.args[key].endswith(' epoch'):
                    scheduler_cfg.args[key] = 1

        scheduler = instantiate_no_recursive(scheduler_cfg.args, optimizer=optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': scheduler_cfg.interval,
                'frequency': scheduler_cfg.frequency,
                'monitor': src.g_cfg.watch_field,
                'strict': True
            }
        }

    def write_prediction(self, filename, mode, output=None):
        if output is None:
            output = self._val_outputs[0] if mode == 'val' else self._test_outputs[0]
        output = self.model.preprocess_write(output)

        if dist.is_initialized() and (ws := dist.get_world_size()) > 1:
            holder = [None] * ws
            dist.all_gather_object(holder, output)
            if dist.get_rank() > 0:
                return
            else:
                output = reduce(merge_outputs, holder)

        ds = self.dm.datasets[mode]
        with open(filename, 'w') as f:
            self.model.write_prediction(f, output, ds, self.dm.vocabs)

    def init_alpha_scheduler(self, command):
        return get_coeff_iter(command, lambda: self.current_epoch)

    def reduce_loss(self, loss, vp):
        return reduce_loss(self.loss_reduction_mode, loss, vp.num_token, vp.batch_size)

    def load_model_inplace(self, path):
        checkpoint = pl_load(path, map_location=lambda storage, loc: storage)
        checkpoint = self.model.process_checkpoint(checkpoint)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        if src.debugging:
            model_state = self.state_dict()
            for key, value in checkpoint['state_dict'].items():
                assert torch.allclose(model_state[key], value)

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()

    def predict_dataloader(self):
        return self.dm.predict_dataloader()
