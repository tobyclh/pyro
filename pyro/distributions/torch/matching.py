from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


def _unstack(tensor, sizes):
    assert tensor.shape(-1) == sum(sizes)
    parts = []
    pos = 0
    for size in sizes:
        parts.append(tensor[..., pos:pos + size])
        pos += size
    return parts


@copy_docs_from(torch.distributions.Distribution)
class TorchMatching(torch.distributions.Distribution):
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
        super(TorchMatching, self).__init__(batch_shape, event_shape)

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

    def _compute_stats(self):
        """
        Computes marginals and log normalizer using loopy belief propagation.
        """
        if self._marginals is not None:
            return

        # FIXME these are bogus values of the correct shape.
        self._marginals = 0.5 * torch.ones(self.batch_shape + self.event_shape)
        self._log_normalizer = torch.zeros(self.batch_shape)


@copy_docs_from(TorchDistribution)
class Matching(TorchDistribution):
    def __init__(self, head_logits, tail_logits, edge_logits, heads, tails, *args, **kwargs):
        torch_dist = TorchMatching(head_logits, tail_logits, edge_logits, heads, tails)
        super(Matching, self).__init__(torch_dist, *args, **kwargs)
