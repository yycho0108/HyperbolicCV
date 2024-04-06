#!/usr/bin/env python3

import sys
import torch as th
from dataclasses import dataclass

# == import LVAE ==
sys.path.append('lib')
sys.path.append('generation')
from generation.models.LVAE import LVAE
sys.path.pop(-1)
sys.path.pop(-1)

# == import ETUDE ==
from etude.data.factory import get_dataset, DataConfig


@dataclass
class Config:
    data: DataConfig = DataConfig(
        dataset_type='shelf',
    )


def main():
    img_dim = (3, 256, 256)
    net = LVAE(img_dim,
               4,
               5,
               # z_dim
               128,
               8,
               learn_curvature=True,
               enc_K=1.0,
               dec_K=1.0,
               rank=2,
               flat=False)
    print('net', net)
    x = th.zeros((1, 3, 128, 128),
                 dtype=th.float32)
    print('inputs', x.shape)
    outputs = net(x)
    loss = net.loss(x, outputs)
    print('outputs', outputs[0].shape)


if __name__ == '__main__':
    main()
