#!/usr/bin/env python3

import sys
import numpy as np
import torch as th
import pickle
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
from hypll.optim import RiemannianAdam as HRA


# == import ETUDE ==

# == import packman ==
from pkm.util.config import with_oc_cli_v2
from pkm.models.common import grad_step
from pkm.train.ckpt import (load_ckpt, save_ckpt)
from pkm.util.torch_util import dcn

from perlin import PerlinNoiseDataset
from hconv_unet import HConvUNet1d
from hae2 import HConvUNet1d as HLConvUNet1d
from hae3 import PoincareUNet


@dataclass
class Config:
    data: PerlinNoiseDataset.Config = PerlinNoiseDataset.Config(
        epoch_size=256,
        batch_size=256,
        hack_2d=False)
    device: str = 'cuda:0'
    batch_size: int = 256
    kl_coeff: float = 0.024
    num_epoch: int = 8
    unet: HConvUNet1d.Config = HConvUNet1d.Config(c_mid=[16, 32, 64, 128, 256])
    # unet: HLConvUNet1d.Config = HLConvUNet1d.Config(
    #     c_mid=[16, 32, 64, 128, 256],
    #     c_out = 1)
    punet: PoincareUNet.Config = PoincareUNet.Config()

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
    if cfg.data.hack_2d:
        img_dim = (dataset.obs_dim,
                   dataset.seq_len,
                   dataset.seq_len)
    else:
        img_dim = (dataset.obs_dim,
                   dataset.seq_len)

    cfg = replace(cfg, unet=replace(cfg.unet,
                                    c_in=dataset.obs_dim,
                                    # c_out=dataset.obs_dim
                                    c_out=256
                                    ))
    print('cfg', cfg)
    if False:
        model = LVAE(img_dim,
                     # encoder/decoder
                     5,  # 3,
                     6,  # 4,
                     # z_dim
                     128,
                     # initial filters ??
                     64,
                     learn_curvature=False,
                     enc_K=1.0,
                     dec_K=1.0,
                     rank=(2 if cfg.data.hack_2d else 1),
                     # flat=True
                     flat=True
                     ).to(device)
    elif False:
        model = HConvUNet1d(cfg.unet).to(device)
    elif False:
        model = HLConvUNet1d(cfg.unet).to(device)
    else:
        model = PoincareUNet(cfg.punet).to(device)
    print(model)

    # optimizer = RiemannianAdam(model.parameters(),
    #                            lr=3e-4,
    #                            weight_decay=0,
    #                            stabilize=1)
    # optimizer = th.optim.Adam(model.parameters(),
    #                           lr=3e-4,
    #                           weight_decay=0)
    optimizer = HRA(model.parameters(), lr=3e-4)
    lr_scheduler = StepLR(
        optimizer,
        step_size=cfg.data.epoch_size,
        gamma=0.5,
        # verbose=True
    )
    try:
        for epoch in tqdm(range(cfg.num_epoch), desc='epoch'):
            loss_sum = th.zeros(1, device=device)
            loss_rec_sum = th.zeros(1, device=device)
            loss_kl_sum = th.zeros(1, device=device)
            count = 0

            # hmm
            # kl_coeff = (cfg.kl_coeff if epoch >= 1 else 0.0)
            kl_coeff = cfg.kl_coeff
            with tqdm(loader, desc='batch', leave=False,
                      total=cfg.data.epoch_size) as pbar:
                for batch in pbar:
                    x = batch['trajectory']
                    if cfg.data.hack_2d:
                        pass
                    else:
                        if False:
                            x = batch['trajectory'].swapaxes(-1, -2).to(device)
                    # print('x', x.shape)

                    # Make prediction and loss.
                    # print('x', x.shape)
                    outputs = model(x)
                    rec_loss, kl_loss = model.loss(x, outputs)
                    # kl_loss[~th.isfinite(kl_loss)] = 0
                    # print(rec_loss,
                    #       kl_loss,
                    #       cfg.kl_coeff)
                    # loss = rec_loss
                    loss = th.sum(rec_loss + kl_coeff * kl_loss)

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
                    count += 1
                    lr_scheduler.step()

            # x_, x_hat_ = dcn(x), dcn(outputs)
            print(F'<dump epoch={epoch:03d}>')
            with open(F'/tmp/lvae/dump-{epoch:03d}.pkl', 'wb') as fp:
                # print(x.shape, outputs.shape)
                pickle.dump({'x': dcn(x),
                             'x_hat': dcn(outputs)}, fp)
            print('</dump>')

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
