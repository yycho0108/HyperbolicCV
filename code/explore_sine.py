#!/usr/bin/env python3

import torch as th
import numpy as np
from pkm.train.ckpt import (load_ckpt, save_ckpt)
from pkm.util.torch_util import dcn
from hae3 import PoincareUNet
from matplotlib import pyplot as plt
from hypll.tensors import ManifoldTensor, TangentTensor


def main():
    device = 'cuda:1'
    model = PoincareUNet(PoincareUNet.Config()).to(device)
    step: int = 8
    load_ckpt(dict(model=model),
              '/tmp/lvae/last.ckpt')

    z = th.randn((1, 64, 16), device=device)
    z_T = TangentTensor(data=z,
                        man_dim=1,
                        manifold=model.manifold)
    z = model.manifold.expmap(z_T)
    z = ManifoldTensor(
        z.tensor *
        th.logspace(np.log10(0.01), np.log10(1), step,
                    device=device)[:, None, None],
        z.manifold,
        z.man_dim)
    with th.no_grad():
        y = model.decode(z)
    y = dcn(y)
    print('y', y.shape)

    # print('y', y.shape)
    fig, axs = plt.subplot_mosaic([[str(i)] for i in range(step)])
    for i in range(step):
        print(y[i].shape)
        axs[str(i)].plot(y[i].ravel())
    plt.show()


if __name__ == '__main__':
    main()
