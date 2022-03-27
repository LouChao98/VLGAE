import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from queue import Empty, Queue
from threading import Thread

import regex
import torch
import torch.distributed as dist
from fastNLP import SpanFPreRecMetric
from torchmetrics import Metric

from src.utility.logger import get_logger_func
from src.utility.meta import Singleton

_warn, _info, _debug = get_logger_func('metric')
EPS = 1e-12


class SequenceTaggingSpanMetric(Metric):
    def __init__(self, extra_vocab, encoding_type='bioes', use_confidence=True, cvt_style=False, *args, **kwargs):
        # use_confidence: consider probality, see flair.SequenceTagger._obtain_labels
        super().__init__(*args, **kwargs)

        vocab = extra_vocab['target']
        self.use_confidence = use_confidence
        self._metric = SpanFPreRecMetric(vocab,
                                         use_confidence=use_confidence,
                                         encoding_type=encoding_type,
                                         tag_to_span_func=self.get_span_labels if cvt_style else None)
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('unlabeled_correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('predict', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('n', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, predict, gold, mask):
        self.n += len(predict['target'])
        if self.use_confidence:
            predict = (predict['target'], predict['score'])
        else:
            predict = predict['target']
        self._metric.evaluate(predict, gold['target'], mask.sum(1))
        true_positive = sum(self._metric._true_positives.values())
        false_negative = sum(self._metric._false_negatives.values())
        false_positive = sum(self._metric._false_positives.values())
        self.correct += true_positive
        self.total += false_negative + true_positive
        self.predict += false_positive + true_positive
        self.unlabeled_correct += self._metric._unlabeled_true_positives

        self._metric._true_positives = defaultdict(int)
        self._metric._false_positives = defaultdict(int)
        self._metric._false_negatives = defaultdict(int)
        self._metric._unlabeled_true_positives = 0

    def compute(self):
        precision = self.correct / (self.predict + EPS)
        recall = self.correct / (self.total + EPS)
        unlabeled_recall = self.unlabeled_correct / (self.total + EPS)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)
        _debug(f'sent: {self.n}')
        return {'f1': 100 * f1, 'pre': 100 * precision, 'rec': 100 * recall, 'urec': 100 * unlabeled_recall}

    @staticmethod
    def get_span_labels(sentence_tags, ignore_labels=None):
        # copy from cvt.
        # If ['B-a', 'I-b'], this will return [('b', (0, 2))]
        # ignore_labels is no use
        """Go from token-level labels to list of entities (start, end, class)."""

        span_labels = []
        last = 'o'
        start = -1
        for i, tag in enumerate(sentence_tags):
            tag = tag.lower()
            pos, _ = (None, 'o') if tag == 'o' else tag.split('-')
            if (pos == 's' or pos == 'b' or tag == 'o') and last != 'o':
                span_labels.append((last.split('-')[-1], (start, i)))
            if pos == 'b' or pos == 's' or last == 'o':
                start = i
            last = tag
        if sentence_tags[-1].lower() != 'o':
            span_labels.append((sentence_tags[-1].split('-')[-1].lower(), (start, len(sentence_tags))))
        return span_labels


class _PseudoProjDepExternalMetric(metaclass=Singleton):
    def __init__(self, data) -> None:
        ON_POSIX = 'posix' in sys.builtin_module_names
        _warn('Puncatuations are NOT ignored when evaluation.')
        command = [
            'python', '/home/louchao/code/struct_vat2/scripts/eval_dep_with_proj_convert.py',
            os.getcwd(), '--gval', data.dev, '--gtest', data.test, '--maltdir',
            '/home/louchao/code/struct_vat2/data/maltparser-1.9.2', '--cfg',
            data.ud_name.split('_')[0], '--delete'
        ]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, close_fds=ON_POSIX, bufsize=1)
        q = Queue()
        t = Thread(target=self._enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()
        self.p, self.q, self.t = p, q, t

    def get(self):
        try:
            line = self.q.get_nowait()
        except Empty:
            line = None
        return line

    def _enqueue_output(self, out, queue):
        for line in iter(out.readline, ''):
            if line.strip() == '':
                continue
            queue.put(line)
        out.close()

    def __del__(self):
        self.p.terminate()


class PseudoProjDepExternalMetric(Metric):
    def __init__(self, extra_vocab, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not dist.is_initialized(), 'ddp is not supported'
        self.core = _PseudoProjDepExternalMetric(data)
        self.n, self.total = 0, 0  # for compatibility
        self.last_score = (0, 0, 0)  # [uas, las, conll18]
        self.last_epoch = -1
        self.mode = None

    def update(self, predict, gold, mask):
        return

    def compute(self):
        line = self.core.get()
        if line is None:
            _warn(f'No evalution, reporting results on epoch {self.last_epoch}')
            uas, las, las_conll18 = self.last_score
        else:
            if line.startswith('Failed'):
                _warn(line)
                uas, las, las_conll18 = self.last_score
            else:
                epoch, uas, las, las_conll18 = line.strip().split()
                mode, uas = uas.split('/')
                if self.mode is None:
                    self.mode = mode
                else:
                    assert mode == self.mode, f'Mismatch. Expect {self.mode} but get {mode}'
                _debug(f'Reporting results on epoch {epoch}')
                uas = float(uas.split('=')[1])
                las = float(las.split('=')[1])
                las_conll18 = float(las_conll18.split('=')[1])
                self.last_epoch = int(epoch)
                self.last_score = (uas, las, las_conll18)
        return {'uas': uas, 'las': las, 'conll18': las_conll18}


class _PseudoProjDepMetric(metaclass=Singleton):
    def __init__(self, data, ignore_punct) -> None:
        self.ignore_punct = ignore_punct
        assert self.ignore_punct is False, 'Not implemented'
        self.init = 2  # handle sanity check
        self.epoch = 0
        self.dev_set = self._load(data['dev'])
        self.test_set = self._load(data['test'])

        self.tmpdir = '/dev/shm' if os.path.exists('/dev/shm') else None
        self.maltdir = '/home/louchao/data/maltparser-1.9.2'
        self.cfgname = data.ud_name.split('_')[0]
        self.pattern = '{mode}_predict_{epoch}.txt'

        self.test_result = None

    def get(self):
        if (r := self.test_result) is not None:
            self.test_result = None
            return 'test', r
        if self.init > 0:
            self.init -= 1
            if not os.path.exists(self.pattern.format(epoch=self.epoch, mode='dev')):
                self.test_result = (0, 0, 0)
                return 'dev', (0, 0, 0)
            else:
                self.init = 0

        predict_val = self._load_predict(self.pattern.format(epoch=self.epoch, mode='dev'))
        predict_test = self._load_predict(self.pattern.format(epoch=self.epoch, mode='test'))
        self.epoch += 1

        self.test_result = self._eval(predict_test, self.test_set)
        return 'dev', self._eval(predict_val, self.dev_set)

    def _eval(self, predicts, golds):
        total = 0
        u_correct = 0
        l_correct = 0
        l_conll18_correct = 0
        assert len(predicts) == len(golds), 'total num mismatch'
        for s_idx, (predict, gold) in enumerate(zip(predicts, golds)):
            assert len(predict) == len(gold), f'{s_idx} instance mismatch'

            for w_idx, ((_, p_a, p_r), (g_w, g_a, g_r)) in enumerate(zip(predict, gold)):
                if self.ignore_punct and regex.match(r'\p{P}+$', g_w):
                    continue
                total += 1
                if p_a == g_a:
                    u_correct += 1
                    if p_r == g_r:
                        l_correct += 1
                        l_conll18_correct += 1
                    elif p_r.split(':')[0] == g_r.split(':')[0]:
                        l_conll18_correct += 1
        return u_correct / total * 100, l_correct / total * 100, l_conll18_correct / total * 100

    def _load(self, path):
        sents = []
        with open(path) as f:
            sent = []
            for line in f.readlines():
                if line[0] == '#':
                    continue
                line = line.strip().split('\t')
                if len(line) > 7:
                    word_id, word, arc, rel = line[0], line[1], line[6], line[7]
                    if word_id.isdigit():
                        sent.append((word, arc, rel))
                elif len(sent):
                    sents.append(sent)
                    sent = []
            if len(sent):
                sents.append(sent)
        return sents

    def _load_predict(self, path):
        if not os.path.exists(path):
            return None

        # always use *.train to deproj
        tgt_file = tempfile.mktemp(dir=self.tmpdir)
        command = f'cd {self.maltdir}; java -jar maltparser-1.9.2.jar -c {self.cfgname}.train -m deproj' \
                  f' -i {path} -o {tgt_file} -v off'
        os.system(command)

        predict = self._load(tgt_file)

        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(tgt_file):
            os.remove(tgt_file)

        return predict


class PseudoProjDepMetric(Metric):
    def __init__(self, extra_vocab, data, ignore_punct=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = _PseudoProjDepMetric(data, ignore_punct)
        self.n, self.total = 0, 0  # for compatibility
        self.mode = None

    def update(self, predict, gold, mask):
        return

    def compute(self):
        mode, (uas, las, las_conll18) = self.core.get()
        if self.mode is None:
            self.mode = mode
        else:
            assert mode == self.mode, f'Mismatch. Expect {self.mode} but get {mode}'
        return {'uas': uas, 'las': las, 'conll18': las_conll18}
