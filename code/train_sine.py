#!/usr/bin/env python3

import sys
import numpy as np
import torch as th
from dataclasses import dataclass, replace
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm


# == import LVAE ==
sys.path.append('lib')
sys.path.append('generation')
from generation.models.LVAE import LVAE
sys.path.pop(-1)
sys.path.pop(-1)
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD


# == import ETUDE ==

# == import packman ==
from pkm.util.config import with_oc_cli_v2
from pkm.models.common import grad_step
from pkm.train.ckpt import (load_ckpt, save_ckpt)

from perlin import PerlinNoiseDataset


@dataclass
class Config:
    data: PerlinNoiseDataset.Config = PerlinNoiseDataset.Config(
        epoch_size=65536,
        batch_size=256)
    device: str = 'cuda:0'
    batch_size: int = 256
    kl_coeff: float = 0.024

    def __post_init__(self):
        self.data = replace(self.data,
                            batch_size=self.batch_size,
                            device=self.device)


def collate_fn(xlist):
    cols = [x.pop('col-label') for x in xlist]
    # print('cols', cols)
    out = th.utils.data.default_collate(xlist)
    out['col-label'] = np.stack(cols, axis=0)
    return out


@with_oc_cli_v2
def main(cfg: Config = Config):
    print(cfg)
    th.backends.cudnn.benchmark = True
    # th.autograd.set_detect_anomaly(True)

    device = cfg.device
    dataset = PerlinNoiseDataset(cfg.data)
    print('!', dataset.obs_dim,
          dataset.seq_len)

    # th.autograd.detect_anomaly(True)

    # NOTE(ycho):
    # In case of using `EtudeDataset1Sphere`,
    # since all tensors need to be pre-allocated on the GPU,
    # it's necessary to set num_workers=0.
    loader = th.utils.data.DataLoader(dataset,
                                      # batch_size=cfg.batch_size,
                                      batch_size=None,
                                      shuffle=False,
                                      num_workers=0,
                                      # collate_fn=collate_fn
                                      )

    # it's ok to leave the seq_len parameter
    # as `None` in case `flat`=False.
    img_dim = (dataset.obs_dim,
               dataset.seq_len)
    model = LVAE(img_dim,
                 # encoder/decoder
                 5,  # 3,
                 6,  # 4,
                 # z_dim
                 128,
                 # initial filters ??
                 64,
                 learn_curvature=True,
                 enc_K=1.0,
                 dec_K=1.0,
                 rank=1,
                 flat=False).to(device)
    print(model)

    optimizer = RiemannianAdam(model.parameters(),
                               lr=3e-4,
                               weight_decay=0,
                               stabilize=1)
    lr_scheduler = StepLR(
        optimizer,
        step_size=50,
        gamma=0.1
    )
    try:
        for epoch in tqdm(range(8), desc='epoch'):
            loss_sum = th.zeros(1, device=device)
            loss_rec_sum = th.zeros(1, device=device)
            loss_kl_sum = th.zeros(1, device=device)
            count = 0
            with tqdm(loader, desc='batch', leave=False,
                      total=cfg.data.epoch_size) as pbar:
                for batch in pbar:
                    x = batch['trajectory'].swapaxes(-1, -2).to(device)

                    # Make prediction and loss.
                    outputs = model(x)
                    rec_loss, kl_loss = model.loss(x, outputs)
                    # kl_loss[~th.isfinite(kl_loss)] = 0
                    loss = th.sum(rec_loss + cfg.kl_coeff * kl_loss)

                    # == logging...
                    with th.no_grad():
                        pbar.set_postfix_str(F'loss={loss.item():.3f}')
                        loss_sum += loss.item()
                        loss_rec_sum += rec_loss.mean().item()
                        loss_kl_sum += kl_loss.mean().item()
                    grad_step(loss, optimizer)

                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    lr_scheduler.step()
                    count += 1
            print(loss_sum / count,
                  loss_rec_sum / count,
                  loss_kl_sum / count)
            save_ckpt(dict(model=model,
                           sched=lr_scheduler,
                           optim=optimizer),
                      F'/tmp/lvae/epoch-{epoch:04d}.ckpt')
    finally:
        save_ckpt(dict(model=model,
                       sched=lr_scheduler,
                       optim=optimizer), '/tmp/lvae/last.ckpt')


if __name__ == '__main__':
    main()
