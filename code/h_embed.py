#!/usr/bin/env python3

from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt.manifolds.lorentz import Lorentz


class HEmbed(nn.Module):
    """
    Implementation of a hyperbolic embedding layer
    with "wrapped normal" distribution.
    TODO(ycho): is this same as "PGM" (pseudo-gaussian manifold) distribution ?
    """
    @dataclass
    class Config:
        z_dim: int = 0
        learn_curvature: bool = False
        curvature: float = 1.0
        euclidean_input: bool = False
        flat: bool = True

    def __init__(self, cfg: Config, manifold=None):
        super().__init__()
        self.cfg = cfg
        if manifold is None:
            manifold = Lorentz(k=cfg.curvature,
                               learnable=cfg.learn_curvature)
        self.manifold = manifold
        self.distr = LorentzWrappedNormal(self.manifold)

    def check_euclidean(self, x):
        if self.euclidean_input:
            x = self.manifold.projx(
                F.pad(
                    x,
                    pad=(
                        1,
                        0),
                    mode="constant",
                    value=0))
        return x

    def random_sample(self, num_samples, device, mean_H=None,
                      var=None, rescale_var=True):
        """ Draws multiple latent variables from the latent distribution.

        If no mean and variance is given, assume standard normal
        """
        if mean_H is None:
            mean_H = self.manifold.origin((1, self.z_dim + 1), device=device)
        if var is None:
            var = th.ones((1, self.z_dim), device=device)

        covar = self.distr.make_covar(var, rescale=rescale_var)
        samples = self.distr.rsample(
            mean_H, covar, num_samples).transpose(
            1, 0)[0]

        return samples

    def forward(self, mean, var):
        mean_H = self.check_euclidean(mean)

        # Sample from distribution
        covar = self.distr.make_covar(var, rescale=True)
        # Note: Loss is not implemented for multiple samples
        z, u, v = self.distr.rsample(mean_H, covar, num_samples=1, ret_uv=True)

        return z, mean_H, covar, u[0], v[0]

    def loss(self,
             z: th.Tensor,
             mean_H: th.Tensor,
             covar: th.Tensor,
             u: th.Tensor,
             v: th.Tensor):
        """ Computes kl divergence between posterior and prior. """

        # Compute density of posterior
        logp_z_posterior = self.distr.log_prob(z, mean_H, covar, u, v)

        # Compute density of prior (rescaled standard normal)
        mean_H_pr = self.manifold.origin(mean_H.shape, device=mean_H.device)
        covar_pr = th.ones(
            (covar.shape[0],
             covar.shape[1]),
            device=covar.device)
        covar_pr = self.distr.make_covar(covar_pr, rescale=True)

        logp_z_prior = self.distr.log_prob(z, mean_H_pr, covar_pr)

        # Compute KL-divergence between posterior and prior
        kl_div = logp_z_posterior.view(-1) - logp_z_prior.view(-1)

        return kl_div
