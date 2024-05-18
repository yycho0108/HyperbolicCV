#!/usr/bin/env python3

from typing import Optional, Union, Tuple, Iterable
from dataclasses import dataclass
import math
import numpy as np
import torch as th
import einops
from matplotlib import pyplot as plt


def nxtp2(n: int):
    return int(2**np.ceil(np.log2(n)))


def lerp(x0: th.Tensor, x1: th.Tensor, w: th.Tensor):
    return x0 + w * (x1 - x0)


def fade(x: th.Tensor):
    # return t * t * t * (t * (t * 6. - 15.) + 10.)
    return (6 * x**5) - (15 * x**4) + (10 * x**3)


def perlin_1d(batch_size: int,
              seq_len: int,
              num_octave: Optional[int] = None,
              persistence: Union[float, Tuple[float, float]] = 0.5,
              device: Optional[str] = None):
    orig_seq_len = seq_len
    seq_len = nxtp2(seq_len)

    if num_octave is None:
        num_octave = int(np.log2(seq_len))

    grid = einops.repeat(th.arange(seq_len, device=device),
                         '... -> n ...',
                         n=batch_size)
    y = th.zeros((batch_size, seq_len),
                 device=device)

    if isinstance(persistence, Iterable):
        p0, p1 = persistence
        persistence = p0 + (p1 - p0) * th.rand(batch_size, device=device)

    # Generate gradients
    # for o in range(4,5):
    for o in range(num_octave):
        substep = (1 << o)
        n = seq_len // substep
        grad = 2 * th.rand(batch_size,
                           n + 1,
                           device=device) - 1
        gprv, gnxt = grad[..., :-1], grad[..., 1:]
        gprv = einops.repeat(gprv, '... n -> ... (n r)',
                             r=substep)
        gnxt = einops.repeat(gnxt, '... n -> ... (n r)',
                             r=substep)

        # relative positions w.r.t. grid corners
        relp = (grid % substep) / substep
        amplitude = persistence ** (num_octave - o)
        dy = amplitude * lerp(gprv,
                              gnxt,
                              fade(relp))

        y += dy  # amplitude*lerp(gprv*relp, gnxt*(relp-1), fade(relp))

    return y[..., :orig_seq_len]


class PerlinNoiseDataset(th.utils.data.IterableDataset):
    @dataclass
    class Config:
        epoch_size: int = 256
        batch_size: int = 1
        seq_len: int = 256
        num_octave: Optional[int] = None
        persistence_bound: Tuple[float, float] = (0.3, 0.7)
        device: Optional[str] = None

    def __init__(self,
                 cfg: Config,
                 device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        if device is None:
            self.device = cfg.device

    @property
    def obs_dim(self):
        return 1

    @property
    def seq_len(self):
        return self.cfg.seq_len

    def __iter__(self):
        cfg = self.cfg
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = cfg.epoch_size
        else:
            # in a worker process; split workload
            per_worker = int(math.ceil(cfg.epoch_size
                                       / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, cfg.epoch_size)
        return ({'trajectory': perlin_1d(cfg.batch_size,
                                         cfg.seq_len,
                                         cfg.num_octave,
                                         cfg.persistence_bound,
                                         self.device)[..., None]}
                for _ in range(iter_start, iter_end))


def plot_fade():
    x = np.linspace(0.0, 1.0)
    y = fade(x)
    plt.plot(x, y)
    plt.show()


def plot_noise():
    z = perlin_1d(1, 63, device='cpu')
    plt.plot(z[0].detach().cpu().numpy())
    plt.show()


def test_dataset():
    dataset = PerlinNoiseDataset(PerlinNoiseDataset.Config())
    # print(type(iter(dataset)))  # -> generator
    loader = th.utils.data.DataLoader(dataset,
                                      batch_size=None)
    for epoch in range(4):
        for i, batch in enumerate(loader):
            print(epoch, i, batch['trajectory'].shape)
            break


def main():
    # plot_fade()
    # plot_noise()
    test_dataset()


if __name__ == '__main__':
    main()
