#!/usr/bin/env python3


from typing import Tuple
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange, Reduce

from lib.lorentz.blocks.layer_blocks import (
    LorentzConv1d, CustomLorentz,
    LConv1d_Block
)
from etude.model.layers import SinusoidalPositionalEncoding


class Repeat(nn.Module):
    def forward(self, x):
        y = einops.repeat(x, '... s c -> ... (s two) c', two=2)
        # print(F'{x.shape} -> {y.shape}')
        return y

    def extra_repr(self):
        return '... s c -> ... (s two) c'


class HConvUNet1d(nn.Module):

    @dataclass
    class Config:
        c_in: int = 0
        c_out: int = c_in
        c_mid: Tuple[int, ...] = ()

        k: float = 10.0
        learn_k: bool = False
        pos_emb: int = 0

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # self.projx = PoincareBall(c=cfg.k,
        #                           learnable=cfg.learn_k)
        self.projx = CustomLorentz(cfg.k, cfg.learn_k)

        layers = []
        cs = [cfg.c_in, *cfg.c_mid, cfg.c_out]
        # print('cs',cs)

        cs_enc = list(cs)
        if cfg.pos_emb > 0:
            cs_enc[0] = cfg.pos_emb
            layers.append(SinusoidalPositionalEncoding(cfg.c_in,
                                                       cfg.pos_emb,
                                                       True, True))
        else:
            cs_enc[0] += 1
        # layers.append(Rearrange('... s c -> ... c s'))

        # layers.append(LorentzConv1d(manifold,
        #                             cs_enc[0],
        #                             cs_enc[0],
        #                             kernel_size=3, stride=1, padding=1))
        # layers.append(Rearrange('... c s -> ... s c'))

        for (ci, co) in zip(cs_enc[:-1], cs_enc[1:]):
            layers.append(LConv1d_Block(
                # CustomLorentz(cfg.k, cfg.learn_k),
                self.projx,
                ci, co, 3,
                stride=2,
                padding=1,
                activation=nn.GELU(),  # th.relu,
                normalization='batchnorm',
                LFC_normalize=False
            ))
        self.enc = nn.Sequential(*layers)

        layers = []
        cs_dec = list(reversed(cs))
        cs_dec[-1] = cs_dec[-2]
        print('cs_dec', cs_dec)
        # layers.append(Repeat())
        for i, (ci, co) in enumerate(zip(cs_dec[:-1], cs_dec[1:])):
            # Hmm...
            # layers.append(Rearrange('... s c -> ... c s'))
            # layers.append(nn.Upsample(scale_factor=2))
            # layers.append(Rearrange('... c s -> ... s c'))
            # einops.layers.torch.re
            # layers.append(HConv1d(ci, co * 2, 3, 1, 1))
            # layers.append(nn.Upsample(scale_factor=2))
            layers.append(Repeat())
            layers.append(LConv1d_Block(
                # CustomLorentz(cfg.k, cfg.learn_k),
                self.projx,
                ci, co, 3,
                stride=1,
                padding=1,
                activation=nn.GELU(),
                normalization='batchnorm',
                LFC_normalize=False
            ))
            print('ci -> co', ci, co)
            # layers.append(Rearrange('... s (two c) -> ... (s two) c', two=2))

        # output
        # layers.append(nn.Conv1d(co, co, 3, 1, 1))
        # layers.append(Rearrange('... s c -> ... c s'))
        # layers.append(nn.Upsample(scale_factor=2))

        self.dec = nn.Sequential(*layers)

        # try euclidean decoding
        layers = []
        layers.append(Rearrange('... s c -> ... c s'))
        layers.append(nn.Conv1d(co, cfg.c_out, 3, 1, 1))
        layers.append(Rearrange('... c s -> ... s c'))
        self.out = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        # hmm thisis invalid
        # x = PoincareBall(c=1.0, learnable=False).projx(x)
        x = F.pad(x, pad=(1, 0), mode="constant", value=0)
        # x = self.manifold.projx(x)
        x = self.projx.projx(x, dim=-1)

        # print('x0', x)
        # print('x0', x.shape)
        z = self.enc(x)
        # print('z', z.shape)  # 1,4,128
        # print('z', z)
        h = self.dec(z)
        # print('h', h.shape)

        # h = self.dec[-1]
        y = self.out(h)
        print('y', y)
        return y

    def loss(self, x, y):
        # print('x', x.shape,
        #       'y', y.shape)
        rec_loss = F.mse_loss(x, y)
        kl_loss = th.zeros_like(rec_loss)
        return (rec_loss, kl_loss)


def main():
    net = HConvUNet1d(HConvUNet1d.Config(c_in=3,
                                         c_mid=[4, 4, 8],
                                         c_out=3,
                                         k=1.0
                                         ))
    print(net)

    # x = th.randn((1, 64, 3)) * 200
    x = th.rand((256, 64, 3))
    print('x', x.shape)
    y = net(x)
    # print(y)
    print('y', y.shape)
    print(net.loss(x, y))


if __name__ == '__main__':
    main()
