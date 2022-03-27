import re
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional

import pytorch_lightning as pl
from fastNLP import DataSet, Vocabulary
from fastNLP import DataSetIter
from omegaconf import OmegaConf

import src
from src.datamodule.sampler import BasicSampler, ConstantTokenNumSampler
from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func("datamodule")


class DataModule(pl.LightningDataModule):
    INPUTS = ("id", "word", "seq_len")
    TARGETS = ("target",)
    EXTRA_VOCAB = ()  # fields need to build extra_vocab
    LOADER = None

    def __init__(
        self,
        train_path,
        train_init_path,
        train_dataloader,
        dev_path,
        dev_dataloader,
        test_path,
        test_dataloader,
        loader,
        normalize_word=True,
        build_no_create_entry=True,
        max_len=None,
    ):
        """

        :param loader:
        :param normalize_word: clean word.
        :param punct_func: the punct judger
        :param build_no_create_entry: build word vocab using dev, test. See more in fastNLP's doc.
        :param build_word_for_others: build word vocab using unlabeled, augumented ....
            By default, train/dev/test will be used.
        :param max_len: a dict, {name: len}
        :param distributed: be set automatically.
        :param suffix: suffix in name.
        """
        super(DataModule, self).__init__()
        self.train_path = train_path
        self.train_init_path = train_init_path
        self.train_dataloader_cfg = train_dataloader
        self.dev_path = dev_path
        self.dev_dataloader_cfg = dev_dataloader
        self.test_path = test_path
        self.test_dataloader_cfg = test_dataloader

        self._loader = loader
        self.loader = loader  # this field will be set automatically.
        self.normalize_word = normalize_word
        self.build_no_create_entry = build_no_create_entry
        self.max_len = max_len or {}

        self.datasets: Dict[str, DataSet] = {}
        self.ds_should_set_target = {"train", "dev", "test"}
        self.vocabs: Dict[str, Optional[Vocabulary]] = {}
        self._persistent_variable = []
        self._has_setup = False

    # ========================================================================
    # Override these functions to customize.

    def _load(self, path, name) -> DataSet:
        raise NotImplementedError

    def post_init_vocab(self, datasets: Dict[str, DataSet]):
        pass  # after init_vocab, before apply_vocab

    def on_load_extra_data(self, dataset: DataSet):
        pass  # must inplace

    # ========================================================================

    def setup(self, stage=None):
        if self._has_setup:
            return

        _info("Loading data.")

        self.datasets["train"] = self.load(self.train_path, name="train")
        self.datasets["train_init"] = self.load(self.train_init_path, name="train_init")
        self.datasets["test"] = self.load(self.test_path, name="test")
        self.datasets["dev"] = self.load(self.dev_path, name="dev")

        self.init_vocab(stage)

        self.apply_max_len()
        self.setup_fields()
        for name, ds in self.datasets.items():
            _info(
                f'{name} contains {len(ds)} instances and {sum(ds["seq_len"].content) - 2 * len(ds)} tokens.'
            )

        self._has_setup = True
        return self

    def setup_fields(self):
        # This should match the settings when 'can_load=False'
        for ds in self.datasets.values():
            inputs = list(self.INPUTS)
            while True:
                try:
                    ds.set_input(*inputs)
                except KeyError as e:
                    invalid_field = e.args[0][:-27]
                    inputs.remove(invalid_field)
                    _warn(
                        f'Can not find field "{invalid_field}" when setting input fields.'
                    )
                else:
                    break

        for name, ds in self.datasets.items():
            if name in self.ds_should_set_target:
                ds.set_target(*self.TARGETS)
        return self

    def load(self, path, name=None, loader=None):
        self.loader = loader if loader is not None else self._loader
        ds = self._load(path, name)

        # backup input fields
        for field in self.INPUTS:
            if field in ("id", "word", "seq_len"):
                continue
            if field.startswith("vis_"):
                continue
            ds.copy_field(field, f"raw_{field}")

        # process word
        if "word" not in ds:
            ds.copy_field("raw_word", "word")
            if self.normalize_word:
                _debug("Normalizing word.")
                ds.apply_field(self.normalize_word_func, "word", "word")
        elif self.normalize_word:
            _warn("normalize_word is skipped because 'word' exists.")

        if "id" not in ds:
            ds.add_field("id", list(range(len(ds))), padder=None)
        else:
            _warn(
                '"id" is created before the default pipeline. Make sure the padder is set correctly.'
            )

        if "seq_len" not in ds:
            ds.add_seq_len("word")
        else:
            _warn(
                '"seq_len" is created before the default pipeline. Make sure the padder is set correctly.'
            )

        return ds

    def get_create_entry_ds(self):
        return [self.datasets["train"]]

    def get_no_create_entry_ds(self):
        if self.build_no_create_entry:
            no_create_entry_ds = [self.datasets["dev"], self.datasets["test"]]
            no_create_entry_ds = list(
                filter(lambda x: isinstance(x, DataSet), no_create_entry_ds)
            )
        else:
            no_create_entry_ds = []
        return no_create_entry_ds

    def init_vocab(self, stage):
        # set self.vocabs[XXX] = None to skip auto init.

        # init vocab
        if "word" not in self.vocabs:
            self.vocabs["word"] = Vocabulary()
        else:
            assert self.vocabs["word"] is None, "Must be None to skip auto init: word"
        for field in self.EXTRA_VOCAB:
            if field in self.vocabs:
                assert (
                    self.vocabs[field] is None
                ), f"Must be None to skip auto init: {field}"
                continue
            if field in self.INPUTS:
                self.vocabs[field] = Vocabulary()
            else:
                self.vocabs[field] = Vocabulary(padding=None, unknown="<unk>")

        initialized = set()

        create_entry_ds = self.get_create_entry_ds()
        no_create_entry_ds = self.get_no_create_entry_ds()

        # auto build
        if self.vocabs["word"] is not None and "word" not in initialized:
            initialized.add("word")
            self.vocabs["word"].from_dataset(
                *create_entry_ds,
                field_name="word",
                no_create_entry_dataset=no_create_entry_ds,
            )
        for field in self.EXTRA_VOCAB:
            if self.vocabs[field] is not None and field not in initialized:
                initialized.add(field)
                self.vocabs[field].from_dataset(
                    self.datasets["train"], field_name=field
                )

        self.post_init_vocab(self.datasets)
        self._check_all_vocab_initialized()
        self.apply_vocab()

        if stage != "test":
            for vname, vocab in self.vocabs.items():
                vocab.save(f"vocab_{vname}.txt")

    def apply_vocab(self, ds=None):
        if ds is None:
            to_be_indexed = self.datasets.values()
        elif isinstance(ds, (list, tuple)):
            to_be_indexed = ds
        else:
            to_be_indexed = [ds]
        for ds in to_be_indexed:
            if not isinstance(ds, DataSet):
                continue
            for field, vocab in self.vocabs.items():
                if field in ds:
                    vocab.index_dataset(ds, field_name=field)

    def _check_all_vocab_initialized(self):
        for name, vocab in self.vocabs.items():
            if vocab is None:
                raise ValueError(f"Vocab {name} is set to manual setup, but not.")

    def apply_max_len(self):
        for name, ds in self.datasets.items():
            if (max_len := self.max_len.get(name)) is not None:
                ds.drop(lambda i: i["seq_len"] > max_len)

    def add_persistent_variable(self, name):
        assert name in self.__dict__
        self._persistent_variable.append(name)

    def dataloader(self, name):
        if name in ("train", "train_init"):
            return get_dataset_iter(self.datasets[name], **self.train_dataloader_cfg)
        elif name == "dev":
            return get_dataset_iter(
                self.datasets[name], **self.dev_dataloader_cfg, shuffle=False
            )
        elif name == "test":
            return get_dataset_iter(
                self.datasets[name], **self.test_dataloader_cfg, shuffle=False
            )
        raise ValueError

    def train_dataloader(self):
        loaders = {"train": self.dataloader("train")}
        for key in self.datasets:
            if key in ("train", "dev", "test"):
                continue
            loaders[key] = self.dataloader(key)
        _info(f"Returning {len(loaders)} loader(s) as train_dataloader.")
        return loaders

    def val_dataloader(self):
        return self.dataloader("dev")

    def test_dataloader(self):
        return self.dataloader("test")

    def predict_dataloader(self):
        return self.dataloader("test")

    @staticmethod
    def normalize_chars(w: str):
        if w == "-LRB-":
            return "("
        elif w == "-RRB-":
            return ")"
        elif w == "-LCB-":
            return "{"
        elif w == "-RCB-":
            return "}"
        elif w == "-LSB-":
            return "["
        elif w == "-RSB-":
            return "]"
        return w.replace(r"\/", "/").replace(r"\*", "*")

    def normalize_one_word_func(self, w):
        return re.sub(r"\d", "0", self.normalize_chars(w))

    def normalize_word_func(self, ws: List[str]):
        return [re.sub(r"\d", "0", self.normalize_chars(w)) for w in ws]

    def get_vocab_count(self):
        return OmegaConf.create(
            {f"n_{name}": len(vocab) for name, vocab in self.vocabs.items()}
        )

    @staticmethod
    @contextmanager
    def tolerant_exception(allowed, current):
        try:
            yield
        except Exception as e:
            if current not in allowed:
                raise e
            else:
                _warn(str(e))


def get_dataset_iter(
    ds: DataSet,
    token_size,
    num_bucket,
    batch_size=-1,
    single_sent_threshold=-1,
    shuffle=True,
    sort_in_batch=True,
    force_same_len=False,
    **kwargs,
):

    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("pin_memory", False)

    if num_bucket > 1 and len(ds) > num_bucket:
        get_sampler = partial(
            ConstantTokenNumSampler,
            max_token=token_size,
            max_sentence=batch_size,
            num_bucket=num_bucket,
            single_sent_threshold=single_sent_threshold,
            sort_in_batch=sort_in_batch,
            shuffle=shuffle,
            force_same_len=force_same_len,
        )
    else:
        assert batch_size > 0
        get_sampler = partial(
            BasicSampler,
            batch_size=batch_size,
            sort_in_batch=sort_in_batch,
            shuffle=shuffle,
        )

    return DataSetIter(
        ds, batch_sampler=get_sampler([i for i in ds["seq_len"].content]), **kwargs
    )
