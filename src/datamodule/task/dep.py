from collections import Counter

from fastNLP import DataSet
from fastNLP.io import ConllLoader
from nltk.corpus import stopwords

import src

from src.datamodule.datamodule import DataModule
from src.datamodule.vocabulary import Vocabulary
from src.utility.alg import isprojective
from src.utility.logger import get_logger_func
import omegaconf

_warn, _info, _debug = get_logger_func('runner')


class DepDataModule(DataModule):
    INPUTS = ('id', 'word', 'token', 'seq_len')  # word for encoder, token for dmv
    TARGETS = ('arc', )
    LOADER = ConllLoader

    def __init__(
            self,
            use_tag=True,
            num_lex=0,  # limit word in token. not consider tag.
            num_token=99999,  # limit total token. consider (lex, tag) pair.
            ignore_stop_word=False,
            headers=None,
            indexes=None,
            **kwargs):
        assert num_lex > 0 or use_tag, 'Nothing to build token'

        headers = headers or ['raw_word', 'tag', 'arc']
        indexes = indexes or [1, 2, 3]
        loader = self.LOADER(headers, indexes=indexes, dropna=False, sep='\t')

        self.use_tag = use_tag
        if use_tag:
            assert 'tag' in headers
            self.INPUTS = self.INPUTS + ('tag', )
            self.EXTRA_VOCAB = self.EXTRA_VOCAB + ('tag', )

        self.num_lex = num_lex
        self.num_token = num_token
        self.ignore_stop_word = ignore_stop_word
        super().__init__(loader=loader, **kwargs)
        self.vocabs['token'] = None  # set to manual init

        self.token2word = None
        self.token2tag = None
        if self.use_tag and self.num_lex > 0:
            self.token_mode = 'joint'
        elif self.use_tag:
            self.token_mode = 'tag'
        else:
            self.token_mode = 'word'

    def _load(self, path, name):
        ds: DataSet = self.loader._load(path)

        if self.token_mode == 'joint':
            ds.apply(lambda x: [f'{w.lower()}:{p}' for w, p in zip(x['raw_word'], x['tag'])], new_field_name='token')
        elif self.token_mode == 'tag':
            ds.apply(lambda x: x['tag'], new_field_name='token')
        else:
            ds.apply(lambda x: list(map(str.lower, x['raw_word'])), new_field_name='token')

        if name in ('train', 'train_init', 'dev', 'val', 'test'):
            ds['arc'].int()
            orig_len = len(ds)
            ds.drop(lambda i: not isprojective(i['arc']), inplace=False)
            cleaned_len = len(ds)
            if cleaned_len < orig_len:
                _warn(f'Data contains nonprojective trees. {path}')
        else:
            raise NotImplementedError

        return ds

    def post_init_vocab(self, datasets):
        count = Counter()
        word_count = Counter()

        if self.token_mode == 'tag':
            self.vocabs['token'] = self.vocabs['tag']
            self.token2tag = list(range(len(self.vocabs['token'])))
            return

        for ds in self.get_create_entry_ds():
            for inst in ds:
                word_count.update(map(str.lower, inst['word']))
                if self.token_mode == 'joint':
                    count.update(zip(map(str.lower, inst['word']), inst['tag']))

        if self.ignore_stop_word:
            sw = set(stopwords.words('english'))
            used_word = [w for w, i in word_count.most_common(self.num_lex + len(sw)) if w not in sw]
            used_word = set(used_word[:self.num_lex])
        else:
            used_word = set(w for w, i in word_count.most_common(self.num_lex))

        processed_count = {}
        if self.token_mode == 'joint':
            for (w, p), c in count.most_common():
                if w in used_word:
                    processed_count[f'{w}:{p}'] = c
                    if len(processed_count) == self.num_token:
                        break
            for p in self.vocabs['tag'].word2idx:
                if p in ('<pad>', '<unk>'): continue
                processed_count[f'<unk>:{p}'] = 100000
        else:
            for w, c in word_count.most_common():
                if w in used_word:
                    processed_count[w] = c
                    if len(processed_count) == self.num_token:
                        break

        token_vocab = Vocabulary()
        token_vocab.word_count = Counter(processed_count)
        token_vocab.build_vocab()
        self.vocabs['token'] = token_vocab

        if self.token_mode == 'joint':
            w, t = zip(*[token_vocab.idx2word[i].rsplit(':', 1) for i in range(2, len(token_vocab))])
            w = ['<pad>', '<unk>'] + list(w)
            t = ['<pad>', '<unk>'] + list(t)
            self.token2word = [self.vocabs['word'][i] for i in w]
            self.token2tag = [self.vocabs['tag'][i] for i in t]
        else:
            self.token2word = [self.vocabs['word'][token_vocab.idx2word[i]] for i in range(len(token_vocab))]

    def train_dataloader(self):
        loaders = {'train': self.dataloader('train')}
        for key in self.datasets:
            if key in ('train', 'dev', 'test'):
                continue
            if key == 'train_init':
                try:
                    n_init = src.g_cfg.model.init_epoch
                    do_init = src.g_cfg.model.init_method == 'y' and n_init > 0
                except (KeyError, omegaconf.errors.ConfigAttributeError):
                    _warn('ignoring train_init due to missing cfg.')
                    continue
                if do_init:
                    loaders['train'] = _TrainInitLoader(self.dataloader('train_init'), loaders['train'], n_init)
            loaders[key] = self.dataloader(key)
        _info(f'Returning {len(loaders)} loader(s) as train_dataloader.')
        return loaders


class _TrainInitLoader:
    def __init__(self, init_loader, normal_loader, n_init) -> None:
        self.init_loader = init_loader
        self.normal_loader = normal_loader
        self.n_init = n_init
        self.current = 1

    def __iter__(self):
        if self.current <= self.n_init:
            self.current += 1
            _warn('Initializing')
            yield from self.init_loader
        else:
            yield from self.normal_loader
