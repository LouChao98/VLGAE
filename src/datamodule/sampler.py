import math
from functools import partial
from math import ceil
from typing import List

import torch
from fastNLP import RandomSampler, SequentialSampler


from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func("sampler")


class ConstantTokenNumSampler:
    def __init__(
        self,
        seq_len: List[int],
        max_token: int = 4096,
        max_sentence: int = -1,
        num_bucket: int = 16,
        single_sent_threshold: int = -1,
        sort_in_batch: bool = True,
        shuffle: bool = True,
        force_same_len: bool = False,
    ):
        """
        :param List[int] seq_len: sample 的长度的列表。
        :param int max_token: 每个 batch 的最大的 token 数量
        :param int max_sentence: 每个 batch 最大的句子数量，与 max_token 同时生效, <=0 不生效
        :param int num_bucket: 将数据按长度拆分为 num_bucket 个 bucket
        :param int single_sent_threshold: 长度大于阈值的句子强制 batch_size=1, -1 不生效
        :param bool sort_in_batch: 使得一个 batch 内句子长度降序
        :param bool shuffle: shuffle
        :param bool force_same_len: 忽略 num_buckt, 每个长度为一个桶, 每个 batch 中所有的句子长度相同
        """

        assert (
            len(seq_len) >= num_bucket
        ), "The number of samples should be larger than buckets."
        assert (
            num_bucket > 1 or force_same_len
        ), "Use RandomSampler if you do not need bucket."

        self.seq_len = seq_len
        self.max_token = max_token
        self.max_sentence = max_sentence if max_sentence > 0 else 10000000000000000
        self.single_sent_threshold = single_sent_threshold
        self.sort_in_batch = sort_in_batch and not force_same_len
        self.shuffle = shuffle
        self.epoch = 0  # +=1 everytime __iter__ is called.

        # sizes: List[int], pseudo size of each buckets.
        # buckets: List[List[int]], each one is a bucket, containing idx.
        if force_same_len:
            self.sizes = list(set(seq_len))
            len2idx = dict((l, i) for i, l in enumerate(self.sizes))
            self.buckets = [[] for _ in range(len(self.sizes))]
            for i, l in enumerate(seq_len):
                self.buckets[len2idx[l]].append(i)
        else:
            self.sizes, self.buckets = self.kmeans(seq_len, num_bucket)

        # chunks: List[int], n chunk for each bucket
        self.chunks = [
            min(
                len(bucket),
                max(
                    ceil(size * len(bucket) / max_token),
                    ceil(len(bucket) / max_sentence),
                ),
            )
            for size, bucket in zip(self.sizes, self.buckets)
        ]

        self._batches = []
        self._all_batches = []  # including other workers
        self._exhausted = True
        self._init_iter_with_retry()  # init here for valid __len__ at any time.

    def __iter__(self):
        self._init_iter_with_retry()
        yield from self._batches
        self._exhausted = True

    def __len__(self):
        return len(self._batches)

    def _init_iter(self):
        if self.shuffle:
            self.epoch += 1
            g = torch.Generator()
            g.manual_seed(self.epoch)
            range_fn = partial(torch.randperm, generator=g)
        else:
            range_fn = torch.arange

        batches = []
        for i in range(len(self.buckets)):
            split_sizes = [
                (len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                for j in range(self.chunks[i])
            ]
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                batches.append([self.buckets[i][j] for j in batch])
        batches = [
            batch
            for i in range_fn(len(batches))
            for batch in self._process_batch(batches[i])
        ]

        self._batches = batches
        self._all_batches = batches
        self._exhausted = False

    def _init_iter_with_retry(self, max_try=5):
        _count = 0
        while self._exhausted:
            _count += 1
            if _count == max_try:
                raise ValueError("Failed to init iteration.")
            self._init_iter()

    def _process_batch(self, batch):
        # apply sort_in_batch and single_sent_threshold
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        if self.sort_in_batch:
            batch.sort(key=lambda i: -self.seq_len[i])
        if len(batch):
            return [batch] + singles
        else:
            return singles

    def set_epoch(self, epoch: int):
        # This is not a subclass of DistributedSampler， so will never be called by pytorch-lightning.
        breakpoint()  # any case call this?
        self.epoch = epoch

    @staticmethod
    def kmeans(x, k, max_it=32):
        """From https://github.com/yzhangcs/parser/blob/main/supar/utils/alg.py#L7"""

        # the number of clusters must not be greater than the number of datapoints
        x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
        # collect unique datapoints
        d = x.unique()
        # initialize k centroids randomly
        c = d[torch.randperm(len(d))[:k]]
        # assign each datapoint to the cluster with the closest centroid
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

        for _ in range(max_it):
            # if an empty cluster is encountered,
            # choose the farthest datapoint from the biggest cluster and move that the empty one
            mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
            while len(none) > 0:
                for i in none:
                    # the biggest cluster
                    b = torch.where(mask[mask.sum(-1).argmax()])[0]
                    # the datapoint farthest from the centroid of cluster b
                    f = dists[b].argmax()
                    # update the assigned cluster of f
                    y[b[f]] = i
                    # re-calculate the mask
                    mask = torch.arange(k).unsqueeze(-1).eq(y)
                none = torch.where(~mask.any(-1))[0].tolist()
            # update the centroids
            c, old = (x * mask).sum(-1) / mask.sum(-1), c
            # re-assign all datapoints to clusters
            dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
            # stop iteration early if the centroids converge
            if c.equal(old):
                break
        # assign all datapoints to the new-generated clusters
        # the empty ones are discarded
        assigned = y.unique().tolist()
        # get the centroids of the assigned clusters
        centroids = c[assigned].tolist()
        # map all values of datapoints to buckets
        clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

        return centroids, clusters


class BasicSampler:
    """RandomSampler and SequentialSampler"""

    def __init__(
        self,
        seq_len,
        batch_size,
        single_sent_threshold=-1,
        sort_in_batch=True,
        shuffle=True,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.single_sent_threshold = single_sent_threshold
        self.sort_in_batch = sort_in_batch
        self.shuffle = shuffle
        self.epoch = 0

        self._sampler = RandomSampler() if shuffle else SequentialSampler()

    def __iter__(self):
        batch = []
        for i in self._sampler(self.seq_len):
            batch.append(i)
            if len(batch) == self.batch_size:
                yield from self._process_batch(batch)
                batch.clear()
        if batch:
            yield from self._process_batch(batch)

    def __len__(self):
        return math.ceil(len(self.seq_len) / self.batch_size)

    def _process_batch(self, batch):
        # apply sort_in_batch and single_sent_threshold
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        if self.sort_in_batch:
            batch.sort(key=lambda i: -self.seq_len[i])
        if len(batch):
            return [batch] + singles
        else:
            return singles

    def set_epoch(self, epoch: int):
        # This is not a subclass of DistributedSampler
        # this function will never be called by pytorch-lightning.
        self.epoch = epoch
