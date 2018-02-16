from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class MatchingSparse(TorchDistribution):
    def __init__(self, head_logits, tail_logits, edge_logits, heads, tails,
                 one_head_per_tail=False, one_tail_per_head=False):
        if heads.size(-1) != edge_logits.shape(-1):
            raise ValueError("edge_logits, heads shape mismatch: {} vs {}".format(
                edge_logits.shape, heads.shape))
        if tails.size(-1) != edge_logits.shape(-1):
            raise ValueError("edge_logits, tails shape mismatch: {} vs {}".format(
                edge_logits.shape, tails.shape))
        if heads.dim() != 1:
            raise ValueError("expected 1-dimensional heads, got {}".format(heads.shape))
        if tails.dim() != 1:
            raise ValueError("expected 1-dimensional tails, got {}".format(tails.shape))

        batch_shape = broadcast_shape(head_logits.shape[:-1], tail_logits.shape[:-1], edge_logits.shape[:-1])
        event_shape = head_logits.shape[-1] + tail_logits.shape[-1] + edge_logits.shape[-1]
        super(MatchingSparse, self).__init__(batch_shape, event_shape)

        self.head_logits = head_logits.expand(batch_shape + head_logits.shape[:-1])
        self.tail_logits = tail_logits.expand(batch_shape + tail_logits.shape[:-1])
        self.edge_logits = edge_logits.expand(batch_shape + edge_logits.shape[:-1])
        self.logits = torch.stack([self.head_logits, self.tail_logits, self.edge_logits])
        self.heads = heads
        self.tails = tails
        self.one_head_per_tail = one_head_per_tail
        self.one_tail_per_head = one_tail_per_head

        # these are cached by ._compute_stats()
        self._marginals = None
        self._log_normalizer = None

    @property
    def num_heads(self):
        return self.head_logits.shape(-1)

    @property
    def num_tails(self):
        return self.tail_logits.shape(-1)

    @property
    def num_edges(self):
        return self.edge_logits.shape(-1)

    def mean(self):
        self._compute_stats()
        return self._marginals

    def log_prob(self, value):
        expected_shape = self.batch_shape + self.event_shape
        if value.shape[-len(expected_shape):] != expected_shape:
            raise ValueError("misshaped value, expected {}, got {}".format(expected_shape, value.shape()))
        self._compute_stats()
        # Warning: this ignores hard constraints.
        return (self.logits * value).sum(-1) + self._log_normalizer

    def _compute_stats(self, iters=6):
        """
        Computes marginals and log normalizer using loopy belief propagation.

        This is very finicky and can only be run for a small number of
        iterations before diverging, e.g. iters=4 or 6 are safe.
        """
        if self._marginals is not None:
            return
        if not (self.one_tail_per_head and not self.one_head_per_tail):
            raise NotImplementedError

        one = self.logits.new([1])
        head_degree = self.logits.new(self.num_heads).zero_().scatter_add_(0, self.heads, one)
        tail_degree = self.logits.new(self.num_tails).zero_().scatter_add_(0, self.tails, one)
        head_probs = self.logits.new(self.num_heads, int(head_degree.data.max()[0]))
        tail_probs = self.logits.new(self.num_tails, int(tail_degree.data.max()[0]))

        # Initialze heads to uniform.
        head_probs.fill_(1)
        head_probs /= 1 + head_degree  # the 1 is for false alarm
        head_probs[..., head_degree.long():] = 0

        # Initialize tails to cover heads.
        tail_probs.zero_()
        # TODO ...

        # FIXME these are bogus values of the correct shape.
        self._marginals = 0.5 * torch.ones(self.batch_shape + self.event_shape)
        self._log_normalizer = torch.zeros(self.batch_shape)
