from __future__ import annotations

import functools
from collections import defaultdict
from io import IOBase
from typing import Any, Dict, List, Tuple

import torch.nn as nn
from fastNLP import DataSet, Vocabulary
from hydra.utils import instantiate
from torch import Tensor

import src
from src.datamodule import DataModule
from src.model.embedding import Embedding
from src.model.text_encoder import EncoderBase
from src.utility.defaultlist import defaultlist
from src.utility.fn import get_coeff_iter
from src.utility.logger import get_logger_func
from src.utility.var_pool import VarPool
from abc import ABC
from typing import Dict, Any, Type, Tuple

from src.utility.config import Config
from hydra.utils import instantiate
from omegaconf import open_dict, OmegaConf
from torch import Tensor


from src.model.vis_encoder.base import VisEncoderBase

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]

_warn, _info, _debug = get_logger_func("model")


class ModelBase(nn.Module):
    datamodule: DataModule
    embedding: Embedding
    encoder: EncoderBase
    _function_group = {}

    def __init__(self):
        super(ModelBase, self).__init__()
        self._dynamic_cfg = {}

    def setup(self, dm: DataModule):
        self.datamodule = dm
        self.embedding = Embedding(**src.g_cfg.embedding, dm=dm)
        self.encoder = instantiate(src.g_cfg.encoder, embedding=self.embedding)
        self.embedding.__dict__["bounded_model"] = self
        self.encoder.__dict__["bounded_model"] = self

    def forward(
        self, inputs: InputDict, vp: VarPool, embed=None, encoded=None, return_all=False
    ):
        dyn_cfg = self.apply_dynamic_cfg()
        src.trainer.lightning_module.log_dict(dyn_cfg)
        if embed is None:
            embed = self.embedding(inputs, vp)
        if encoded is None or encoded["__need_encode"]:
            if encoded is None:
                encoded = {}
            else:
                del encoded["__need_encode"]
            encoded |= self.encoder(embed, vp)
        encoded["emb"] = embed
        score = self._forward(inputs, encoded, vp)
        if return_all:
            return embed, encoded, score
        return score

    def _forward(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        raise NotImplementedError

    def loss(
        self, x: TensorDict, gold: InputDict, vp: VarPool
    ) -> Tuple[Tensor, TensorDict]:
        raise NotImplementedError

    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        raise NotImplementedError

    def normalize_embedding(self, now):
        self.embedding.normalize(now)

    def preprocess_write(self, output: List[Dict[str, Any]]):
        batch_size = len(output[0]["id"])  # check one batch
        safe_to_sort = all(
            (len(p) == batch_size) for p in output[0]["predict"].values()
        )

        if safe_to_sort:
            # I will put all predicts in the order of idx, but you have to remove padding by yourself.
            sorted_predicts = defaultdict(defaultlist)
            for batch in output:
                id_, predict = batch["id"], batch["predict"]
                for key, value in predict.items():
                    if isinstance(value, Tensor):
                        value = value.detach().cpu().numpy()
                    for one_id, one_value in zip(id_, value):
                        sorted_predicts[key][one_id] = one_value
            return sorted_predicts
        else:
            raise NotImplementedError("Can not preprocess automatically.")

    def write_prediction(
        self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]
    ) -> IOBase:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def set_varpool(self, vp: VarPool) -> VarPool:
        return vp

    @classmethod
    def add_impl_to_group(cls, group, spec, pre_hook=None):
        def decorator(func):
            if group not in cls._function_group:
                cls._function_group[group] = {}
            assert spec not in cls._function_group[group], spec
            cls._function_group[group][spec] = (func, pre_hook)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def set_impl_in_group(self, group, spec):
        try:
            impl, pre_hook = self._function_group[group][spec]
        except Exception as e:
            _warn(f"Failed to load {group}: {spec}")
            raise e
        if pre_hook is not None:
            getattr(self, pre_hook)()
        setattr(self, group, functools.partial(impl, self))

    def add_dynamic_cfg(self, name, command):
        """name: <obj nevigation>|<cfg nevigation>"""
        if name in self._dynamic_cfg:
            _warn(f"Overwriting {name} with {command}")
        self._dynamic_cfg[name] = get_coeff_iter(
            command, idx_getter=lambda: src.trainer.current_epoch
        )

    def apply_dynamic_cfg(self):
        params = {key: next(value) for key, value in self._dynamic_cfg.items()}
        for key, value in params.items():
            obj_nev, cfg_nev = key.split("|")
            o = self
            for attr_name in obj_nev.split("."):
                o = getattr(o, attr_name)
            s = o
            cfg_nev = cfg_nev.split(".")
            for k in cfg_nev[:-1]:
                s = s[k]
            s[cfg_nev[-1]] = value
        return params

    def process_checkpoint(self, ckpt):
        return ckpt


class JointModelBase(ModelBase, ABC):
    # assume only one datamodule
    # assume image does not require embedding
    # assume all visual-side module/parameter are named with 'vis_' prefix.

    # I prefer not seperate the joint model into a language-side model and a visual-side model
    # because it is hard to foresee possible interaction between two sides and
    # for now the visual-side model is very simple.

    # language part, inherit from ModelBase
    # datamodule: DataModule
    # embedding: Embedding
    # encoder: EncoderBase

    # visual part
    vis_encoder: VisEncoderBase

    def setup(self, dm: DataModule):
        if getattr(self, "__setup_handled") is not True:
            _warn("You call setup() directly. Consider to use _setup()")
        self.datamodule = dm
        # self.embedding = Embedding(**src.g_cfg.embedding, dm=dm)
        # self.embedding.__dict__['bounded_model'] = self
        self.encoder = instantiate(src.g_cfg.encoder, embedding=self.embedding)
        self.encoder.__dict__["bounded_model"] = self
        self.vis_encoder = instantiate(src.g_cfg.vis_encoder)
        if self.vis_encoder is None:
            _warn("vis_encoder is disabled.")
        else:
            self.vis_encoder.__dict__["bounded_model"] = self

    def _setup(self, dm: DataModule, cfg_class: Type[Config], allow_missing=None):
        setattr(self, "__setup_handled", True)
        self.cfg = cfg = cfg_class.build(self.cfg, allow_missing=allow_missing)
        with open_dict(cfg.dep_model_cfg):
            cfg.dep_model_cfg = OmegaConf.merge(cfg.dep_model_cfg, dm.get_vocab_count())
        self.dependency = instantiate(cfg.dep_model_cfg)
        self.dependency.setup(dm)
        JointModelBase.setup(self, dm)
        return cfg

    @property
    def embedding(self):
        return self.dependency.embedding

    def forward(
        self,
        inputs: InputDict,
        vp: VarPool,
        embed=None,
        encoded=None,
        vis_encoded=None,
        return_all=False,
    ):
        if vis_encoded is None:
            vis_input = {
                key: value for key, value in inputs.items() if key.startswith("vis_")
            }
            if len(vis_input) > 0:
                vis_encoded = self.vis_encoder(vis_input, vp)
            else:
                vis_encoded = {}
        encoded = encoded if encoded is not None else {"__need_encode": True}
        for key, value in vis_encoded.items():
            encoded[f"vis_{key}"] = value
        embed, encoded, score = super().forward(inputs, vp, embed, encoded, True)
        vis_score = self._vis_forward(inputs, vis_encoded, encoded, score, vp)
        score = {**score, **vis_score}
        if return_all:
            return embed, encoded, score
        else:
            return score

    def _forward(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        return self.dependency._forward(inputs, encoded, vp)

    def _vis_forward(
        self,
        inputs: InputDict,
        encoded: TensorDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        raise NotImplementedError

