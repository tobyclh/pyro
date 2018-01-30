from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.transforms import Transform

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Distribution)
class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied,

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, [f])
        log p(Y) = log p(X) + log det (dX/dY)
    """
    stateful = True  # often true because transforms may cache intermediate results

    def __init__(self, base_dist, transforms, log_pdf_mask=None):
        if not all(isinstance(t, Transform) for t in transforms):
            raise TypeError("Expected a list of Transforms, but got\n{}".format(transforms))
        self.base_dist = base_dist
        self.transforms = transforms
        self.log_pdf_mask = log_pdf_mask

    @property
    def reparameterized(self):
        return self.base_dist.reparameterized

    def batch_shape(self):
        return self.base_dist.batch_shape()

    def event_shape(self):
        return self.base_dist.event_shape()

    def sample(self, sample_shape=torch.Size()):
        x = self.base_dist.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def log_prob(self, y):
        log_prob = 0.0
        for t in reversed(self.transforms):
            x = t.inv(y)
            log_prob -= t.log_abs_det_jacobian(x, y)
            y = x
        log_prob += self.base_dist.log_prob(y)
        if self.log_pdf_mask is not None:
            # Prevent accidental broadcasting of log_prob tensor,
            # e.g. (64, 1), (64,) --> (64, 64)
            assert len(self.log_pdf_mask.size()) <= len(log_prob.size())
            log_prob = log_prob * self.log_pdf_mask
        return log_prob
