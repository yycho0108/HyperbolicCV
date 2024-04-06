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
    # it's ok to leave the seq_len parameter
    # as `None` in case `flat`=False.
    img_dim = (3, 768)
    net = LVAE(img_dim,
               3,
               7,
               # z_dim
               128,
               8,
               learn_curvature=True,
               enc_K=1.0,
               dec_K=1.0,
               rank=1,
               flat=True)
    print('net', net)
    x = th.zeros((1, 3, 768),
                 dtype=th.float32)
    print('inputs', x.shape)
    outputs = net(x)
    print('outputs', outputs[0].shape)
    loss = net.loss(x, outputs)


if __name__ == '__main__':
    main()
