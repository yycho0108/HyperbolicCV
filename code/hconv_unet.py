#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops.layers.torch import Rearrange, Reduce
from hconv import HConv1d

from lib.geoopt.manifolds.lorentz import Lorentz
from lib.geoopt import PoincareBall
from etude.model.layers import SinusoidalPositionalEncoding


class Repeat(nn.Module):
    def forward(self, x):
        return einops.repeat(x, '... s c -> ... (s two) c', two=2)

    def extra_repr(self):
        return '... s c -> ... (s two) c'


class InputLayer(nn.Module):
    def forward(self, x):
        x = self.projx.projx(x)


class HConvUNet1d(nn.Module):

    @dataclass
    class Config:
        c_in: int = 0
        c_out: int = c_in
        c_mid: Tuple[int, ...] = ()

        k: float = 1.0
        learn_k: bool = True
        pos_emb: int = 0

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.projx = PoincareBall(c=cfg.k,
                                  learnable=cfg.learn_k)
        # self.projx = Lorentz(k=cfg.k, learnable=cfg.learn_k)

        layers = []
        cs = [cfg.c_in, *cfg.c_mid, cfg.c_out]

        cs_enc = list(cs)
        if isinstance(self.projx, Lorentz):
            cs_enc[0] += 1
        if cfg.pos_emb > 0:
            cs_enc[0] = cfg.pos_emb
            layers.append(SinusoidalPositionalEncoding(cfg.c_in,
                                                       cfg.pos_emb,
                                                       True, True))
        layers.append(Rearrange('... s c -> ... c s'))
        layers.append(nn.Conv1d(cs_enc[0],
                                cs_enc[0], kernel_size=3, stride=1, padding=1))
        layers.append(Rearrange('... c s -> ... s c'))

        for (ci, co) in zip(cs_enc[:-1], cs_enc[1:]):
            layers.append(HConv1d(ci, co, 3,
                                  stride=2,
                                  padding=1,
                                  manifold=self.projx))
        self.enc = nn.Sequential(*layers)

        layers = []
        cs_dec = list(reversed(cs))
        # layers.append(Repeat())
        for i, (ci, co) in enumerate(zip(cs_dec[:-1], cs_dec[1:])):
            # Hmm...
            # layers.append(Rearrange('... s c -> ... c s'))
            # layers.append(nn.Upsample(scale_factor=2))
            # layers.append(Rearrange('... c s -> ... s c'))
            # einops.layers.torch.re
            layers.append(HConv1d(ci,
                                  # co * 2, 3, 1, 1,
                                  co, 3, 1, 1,
                                  # manifold=self.projx
                                  ))
            layers.append(Repeat())
            # layers.append(Rearrange('... s (two c) -> ... (s two) c', two=2))

        # output
        # layers.append(nn.Conv1d(co, co, 3, 1, 1))
        layers.append(Rearrange('... s c -> ... c s'))
        # layers.append(nn.Upsample(scale_factor=2))

        self.dec = nn.Sequential(*layers)

        # try euclidean decoding
        layers = []
        layers.append(nn.Conv1d(co, co, 3, 1, 1))
        layers.append(Rearrange('... c s -> ... s c'))
        self.out = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        # hmm this is invalid
        # x = PoincareBall(c=1.0, learnable=False).projx(x)
        if isinstance(self.projx, Lorentz):
            x = F.pad(x, pad=(1, 0), mode="constant", value=0)
        x = self.projx.projx(x)

        # print('x0', x)
        z = self.enc(x)
        # print('z', z.shape)
        # print('z', z)
        h = self.dec(z)

        # h = self.dec[-1]
        y = self.out(h)
        return y

    def loss(self, x, y):
        rec_loss = F.mse_loss(x, y)
        kl_loss = th.zeros_like(rec_loss)
        return (rec_loss, kl_loss)


def main():
    net = HConvUNet1d(HConvUNet1d.Config(1, 256, [16, 64, 128]))
    print(net)
    x = th.rand((1, 64, 1))
    y = net(x)
    print(y)
    print('y', y.shape)
    print(net.loss(x, y))


if __name__ == '__main__':
    main()
