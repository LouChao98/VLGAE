from fastNLP import Vocabulary as _fastNLP_Vocabulary
from fastNLP.core.vocabulary import _check_build_vocab


class Vocabulary(_fastNLP_Vocabulary):
    @_check_build_vocab
    def __getitem__(self, w: str):
        if w.endswith("::"):
            w = [w[:-2], ":"]
        else:
            w = w.rsplit(":", 1)
        w[0] = w[0].lower()
        if (_w := ":".join(w)) in self._word2idx:
            return self._word2idx[_w]
        if (_w := "<unk>:" + w[1]) in self._word2idx:
            return self._word2idx[_w]
        # no need to check <unk>
        raise ValueError("word `{}` not in vocabulary".format(w))
