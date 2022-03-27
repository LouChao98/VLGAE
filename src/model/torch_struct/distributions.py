import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

from .dmv import NOCHILD, RIGHT, DMV1oStruct
from .deptree import DepTree
from .helpers import _Struct

from .semirings import (
    CrossEntropySemiring,
    EntropySemiring,
    GumbelCRFSemiring,
    KLDivergenceSemiring,
    KMaxSemiring,
    LogSemiring,
    MaxSemiring,
    MultiSampledSemiring,
    RiskSemiring,
    StdSemiring,
)
from .semirings.semirings import NEGINF


class StructDistribution(Distribution):
    r"""
    Base structured distribution class.

    Dynamic distribution for length N of structures :math:`p(z)`.

    Implemented based on gradient identities from:

    * Inside-outside and forward-backward algorithms are just backprop :cite:`eisner2016inside`
    * Semiring Parsing :cite:`goodman1999semiring`
    * First-and second-order expectation semirings with applications to minimum-risk training on translation forests :cite:`li2009first`

    Parameters:
        log_potentials (tensor, batch_shape x event_shape) :  log-potentials :math:`\phi`
        lengths (long tensor, batch_shape) : integers for length masking
    """

    has_enumerate_support = True
    struct: _Struct = None

    def __init__(self, log_potentials, lengths=None, args={}):
        batch_shape = log_potentials.shape[:1]
        event_shape = log_potentials.shape[1:]
        self.log_potentials = log_potentials
        self.lengths = lengths
        self.args = args
        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=False
        )

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def log_prob(self, value):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): One-hot events (*sample_shape x batch_shape x event_shape*)

        Returns:
            log_probs (*sample_shape x batch_shape*)
        """

        d = value.dim()
        batch_dims = range(d - len(self.event_shape))
        v = self._struct().score(
            self.log_potentials,
            value.type_as(self.log_potentials),
            batch_dims=batch_dims,
        )

        return v - self.partition

    @lazy_property
    def entropy(self):
        """
        Compute entropy for distribution :math:`H[z]`.

        Returns:
            entropy (*batch_shape*)
        """

        return self._struct(EntropySemiring).sum(self.log_potentials, self.lengths)

    def cross_entropy(self, other):
        """
        Compute cross-entropy for distribution p(self) and q(other) :math:`H[p, q]`.

        Returns:
            cross entropy (*batch_shape*)
        """

        return self._struct(CrossEntropySemiring).sum(
            [self.log_potentials, other.log_potentials], self.lengths
        )

    def kl(self, other):
        """
        Compute KL-divergence for distribution p(self) and q(other) :math:`KL[p || q] = H[p, q] - H[p]`.

        Returns:
            cross entropy (*batch_shape*)
        """
        return self._struct(KLDivergenceSemiring).sum(
            [self.log_potentials, other.log_potentials], self.lengths
        )

    def risk(self, cost):
        return self._struct(RiskSemiring).sum([self.log_potentials, cost], self.lengths)

    @lazy_property
    def max(self):
        r"""
        Compute an max for distribution :math:`\max p(z)`.
        Returns:
            max (*batch_shape*)
        """
        return self._struct(MaxSemiring).sum(self.log_potentials, self.lengths)

    @lazy_property
    def argmax(self):
        r"""
        Compute an argmax for distribution :math:`\arg\max p(z)`.

        Returns:
            argmax (*batch_shape x event_shape*)
        """
        return self._struct(MaxSemiring).marginals(self.log_potentials, self.lengths)

    def kmax(self, k):
        r"""
        Compute the k-max for distribution :math:`k\max p(z)`.
        Returns:
            kmax (*k x batch_shape*)
        """
        with torch.enable_grad():
            return self._struct(KMaxSemiring(k)).sum(
                self.log_potentials, self.lengths, _raw=True
            )

    def topk(self, k):
        r"""
        Compute the k-argmax for distribution :math:`k\max p(z)`.

        Returns:
            kmax (*k x batch_shape x event_shape*)
        """
        with torch.enable_grad():
            return self._struct(KMaxSemiring(k)).marginals(
                self.log_potentials, self.lengths, _raw=True
            )

    @lazy_property
    def mode(self):
        return self.argmax

    @lazy_property
    def marginals(self):
        """
        Compute marginals for distribution :math:`p(z_t)`.

        Can be used in higher-order calculations, i.e.

        *

        Returns:
            marginals (*batch_shape x event_shape*)
        """
        return self._struct(LogSemiring).marginals(self.log_potentials, self.lengths)

    @lazy_property
    def count(self):
        "Compute the log-partition function."
        ones = torch.ones_like(self.log_potentials)
        ones[self.log_potentials.eq(-float("inf"))] = 0
        return self._struct(StdSemiring).sum(ones, self.lengths)

    def gumbel_crf(self, temperature=1.0):
        with torch.enable_grad():
            st_gumbel = self._struct(GumbelCRFSemiring(temperature)).marginals(
                self.log_potentials, self.lengths
            )
            return st_gumbel

    @lazy_property
    def partition(self):
        "Compute the log-partition function."
        return self._struct(LogSemiring).sum(self.log_potentials, self.lengths)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        r"""
        Compute structured samples from the distribution :math:`z \sim p(z)`.

        Parameters:
            sample_shape (int): number of samples

        Returns:
            samples (*sample_shape x batch_shape x event_shape*)
        """
        assert len(sample_shape) == 1
        nsamples = sample_shape[0]
        samples = []
        sample = None
        for k in range(nsamples):
            if k % 10 == 0:
                sample = self._struct(MultiSampledSemiring).marginals(
                    self.log_potentials, lengths=self.lengths
                )
                sample = sample.detach()
            tmp_sample = MultiSampledSemiring.to_discrete(sample, (k % 10) + 1)
            samples.append(tmp_sample)
        return torch.stack(samples)

    def to_event(self, sequence, extra, lengths=None):
        "Convert simple representation to event."
        return self.struct.to_parts(sequence, extra, lengths=None)

    def from_event(self, event):
        "Convert event to simple representation."
        return self.struct.from_parts(event)

    def enumerate_support(self, expand=True):
        """
        Compute the full exponential enumeration set.

        Returns:
            (enum, enum_lengths) - (*tuple cardinality x batch_shape x event_shape*)
        """
        _, _, edges, enum_lengths = self._struct().enumerate(
            self.log_potentials, self.lengths
        )
        # if expand:
        #     edges = edges.unsqueeze(1).expand(edges.shape[:1] + self.batch_shape[:1] + edges.shape[1:])
        return edges, enum_lengths

    def _struct(self, sr=None):
        return self.struct(sr if sr is not None else LogSemiring)


class DMV1o(StructDistribution):
    struct = DMV1oStruct

    def __init__(self, log_potentials, lengths, args={}):
        # dec, trans   merge first.
        super().__init__(log_potentials[0], lengths=lengths, args=args)
        self.log_potentials = log_potentials

    @staticmethod
    def merge(dec: Tensor, attach: Tensor, root: Tensor, one=0, zero=NEGINF):
        batch_size, N, *_ = dec.shape
        N += 1
        attach_wroot = torch.full((batch_size, N, N, 2), zero, device=dec.device)
        dec_wroot = torch.full((batch_size, N, 2, 2, 2), zero, device=dec.device)

        # treat the root token as the first token of the sentence.
        attach_wroot[:, 0, 1:, NOCHILD] = root
        attach_wroot[:, 1:, 1:, :] = attach
        dec_wroot[:, 0, RIGHT, :, :] = one
        dec_wroot[:, 1:] = dec
        return dec_wroot, attach_wroot



class DependencyCRF(StructDistribution):
    r"""
    Represents a projective dependency CRF.

    Reference:

    * Bilexical grammars and their cubic-time parsing algorithms :cite:`eisner2000bilexical`

    Event shape is of the form:

    Parameters:
       log_potentials (tensor) : event shape (*N x N*) head, child or (*N x N x L*) head,
                                 child, labels with arc scores with root scores on diagonal
                                 e.g. :math:`\phi(i, j)` where :math:`\phi(i, i)` is (root, i).
       lengths (long tensor) : batch shape integers for length masking.


    Compact representation: N long tensor in [0, .. N] (indexing is +1)

    Implementation uses linear-scan, forward-pass only.

    * Parallel Time: :math:`O(N)` parallel merges.
    * Forward Memory: :math:`O(N \log(N) C^2 K^2)`

    """

    def __init__(self, log_potentials, lengths=None, args={}, multiroot=False):
        super(DependencyCRF, self).__init__(log_potentials, lengths, args)
        setattr(self.struct, 'multiroot', multiroot)

    struct = DepTree