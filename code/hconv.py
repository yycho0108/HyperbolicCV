#!/usr/bin/env python3

from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
# import geoopt
import einops
# from lib.lorentz.manifold import CustomLorentz
# from lib.geoopt.manifolds.lorentz import Lorentz
# from lib.geoopt.manifolds. import Lorentz
from lib.geoopt import PoincareBall


def hconv1d(
        # data
        x: th.Tensor,

        # tangent-space weights and biases
        W: th.Tensor,
        b: th.Tensor,
        K: float,

        # "center"
        # - in case of MLPs, maybe this should be
        # (1) the channel average (local?)
        # (2) the batchnorm statistics? (global)
        # (3) the _zero_ origin
        # - in case of Convolution, this is more naturally(?)
        # defined as the "centered" pixel, I think.
        c: Optional[th.Tensor] = None,
        M=PoincareBall
):
    """
    x: (..., S, C)

    # For now, let's say this is shaped like inputs to F.conv1d
    W: weight mapping from k_1... -> k_2... with receptive field
    b: "C" I think
    """
    x0 = x

    # Project every single pixel to tangent space.
    # i.e. (..., S, C) --> (..., S, K, C)
    x = tangent_from_global(x, c, M=M)

    # We'd like to define the kernels in the tangent space,
    # which is the only way I think this makes intuitive sense.
    # x = F.linear(x, W, b)
    s = x.shape
    # x = x.reshape(-1, x.shape[-1])
    x = einops.rearrange(x, '... s c -> (...) c s')
    x = F.conv1d(x)
    x = x.swapaxes(-1, -2)
    x = x.reshape(*s[:-2], *x.shape[-2:])  # ... s' c'

    # F.conv1d
    x = global_from_tangent(x, c, M=M)
    return x

# class LorentzFullyConnected(nn.Module):
#     """
#         Modified Lorentz fully connected layer of Chen et al. (2022).

# Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

#         args:
#             manifold: Instance of Lorentz manifold
#             in_features, out_features, bias: Same as nn.Linear
#             init_scale: Scale parameter for internal normalization
#             learn_scale: If scale parameter should be learnable
#             normalize: If internal normalization should be applied
#     """

#     def __init__(
#             self,
#             manifold: CustomLorentz,
#             in_features,
#             out_features,
#             bias=False,
#             init_scale=None,
#             learn_scale=False,
#             normalize=False
#         ):
#         super(LorentzFullyConnected, self).__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias
#         self.normalize = normalize

# self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

#         self.init_std = 0.02
#         self.reset_parameters()

#         # Scale for internal normalization
#         if init_scale is not None:
#             self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
#         else:
# self.scale = nn.Parameter(torch.ones(()) * 2.3,
# requires_grad=learn_scale)

#     def forward(self, x):

#         x = self.weight(x)
#         x_space = x.narrow(-1, 1, x.shape[-1] - 1)

#         if self.normalize:
#             scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
#             square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

#             mask = square_norm <= 1e-10

#             square_norm[mask] = 1
#             unit_length = x_space/torch.sqrt(square_norm)
#             x_space = scale*unit_length

#             x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
#             x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

#             mask = mask==False
#             x_space = x_space * mask

#             x = torch.cat([x_time, x_space], dim=-1)
#         else:
#             x = self.manifold.add_time(x_space)

#         return x

#     def reset_parameters(self):
#         nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

#         if self.bias:
#             nn.init.constant_(self.weight.bias, 0)


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class HLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 *args,
                 **kwds):
        manifold = kwds.pop('manifold', None)
        if manifold is None:
            curvature = kwds.pop('curvature', 1.0)
            learn_k = kwds.pop('learn_k', False)
            # manifold = Lorentz(k=1.0)
            # manifold = PoincareBall(c=1.0)
            manifold = PoincareBall(c=curvature,
                                    learnable=learn_k)
        super().__init__(in_features,
                         out_features,
                         bias, *args, **kwds)
        self.M = manifold
        # self.norm = kwds.pop('norm', nn.Identity())
        # self.actv = kwds.pop('actv', nn.Identity())
        self.norm = kwds.pop('norm', nn.LayerNorm(out_features))
        # self.norm = kwds.pop('norm', nn.BatchNorm1d(out_features))
        # self.norm = kwds.pop('norm', BatchNorm1d(out_features))

        self.actv = kwds.pop('actv',
                             # nn.Identity()
                             nn.GELU()
                             )

    def forward(self, x: th.Tensor):
        # y=sigmoid(Wx+b)
        return self.M.expmap0(
            # super().forward(self.M.logmap0(x))
            # self.actv(self.norm(F.linear(self.M.logmap0(x),
            #                              self.weight, self.bias)))
            self.norm(self.actv(F.linear(self.M.logmap0(x),
                                         self.weight, self.bias)))
        )


class HConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 *args,
                 **kwds
                 ):
        manifold = kwds.pop('manifold', None)
        curvature = kwds.pop('curvature', 1.0)
        learn_k = kwds.pop('learn_k', True)
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         *args, **kwds)
        if manifold is None:
            # manifold = Lorentz(k=curvature,
            #                    learnable=learn_k)
            manifold = PoincareBall(c=curvature,
                                    learnable=learn_k)

        self.__kernel_size = kernel_size
        self.__i_c = kernel_size // 2
        self.__stride = stride
        self.__pad = (padding, 0)
        self.M = manifold
        self.h_linear = HLinear(in_channels,
                                out_channels,
                                bias=True,
                                manifold=self.M
                                )
        # self.norm = kwds.pop('norm', nn.LayerNorm(out_channels))
        # self.norm = kwds.pop('norm', nn.Identity())  # (out_channels))
        self.norm = kwds.pop('norm', BatchNorm1d(out_channels))
        # self.norm = kwds.pop('norm', nn.Identity())  # (out_channels))
        self.actv = kwds.pop('actv', nn.GELU())

    def forward(self, x):
        x0 = x
        s = x.shape

        # ..., s, c -> ..., s, k, c
        # print('input', x.shape)
        x = einops.repeat(x, '... s c -> (...) c s one',
                             one=1)
        channels = x.shape[1]
        # 8, 9, 3
        # print('x', x.shape)
        x = F.unfold(x,
                     kernel_size=(self.__kernel_size, 1),
                     stride=(self.__stride, 1),
                     padding=self.__pad)

        # print('after unfold', x.shape)
        # print('unfold', x)
        # print('x', x.shape)  # 8,9,3 N=8, C=9, '*'=3
        # print('x', u_x.shape)  # 8, 81, 2
        x = einops.rearrange(x, 'n (c k) s -> (n s) c k', c=channels)
        # print('x', x.shape)

        # FIXME(ycho): what if kernel size is even??
        c = x[..., self.__i_c]

        # x = tangent_from_global(x, c, M=M)
        # print('c', c.shape, x0.shape)
        # print(c - x0.reshape(c.shape))
        # print('>>x', x.shape, self.out_channels)
        x = self.M.logmap(c[..., None], x, dim=-2)
        # print('log-x', x)
        # x = x.squeeze
        # print('x', x.shape)
        # print('c', c)
        # print('log_c(x)', x)
        # x = x.reshape(*x.shape[:-2], -1)
        # W = einops.rearrange(self.weight, 'co ci ... -> co (... ci)')
        # x = F.linear(x, W, self.bias)

        # x = (B, C, iW)
        # W = (co, ci, kW)
        x = F.conv1d(x, self.weight, self.bias).squeeze(dim=-1)
        # print('conv-x', x)
        # x = self.actv(self.norm(x))
        x = self.norm(self.actv(x))
        # print('norm-x', x)
        c1 = self.h_linear(c)
        # print('>>x', x.shape, self.out_channels)
        x = self.M.expmap(c1, x)
        x = x.reshape(*s[:-2], -1, self.out_channels)
        # print('out', x)
        return x


class HConvT1d(nn.ConvTranspose1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 *args,
                 **kwds
                 ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         output_padding,
                         *args, **kwds)
        print(self.output_padding)

        # hmm.
        # padding_implicit = (0, kernel_size - padding[0] - 1)
        padding_implicit = (kernel_size - padding - 1)
        self.conv = HConv1d(in_channels,
                            out_channels,
                            kernel_size,
                            # stride,
                            1,
                            padding_implicit,
                            *args, **kwds)
        self.stride = stride

        # hmm~
        w = F.pad(th.ones((self.in_channels, 1, 1, 1)),
                  (1, 1, 1, 1))
        w = w[..., 0].detach().clone()
        self.pad_weight = nn.Parameter(w, requires_grad=False)

        # self.pad_weight = nn.Parameter(
        #     F.pad(th.ones((self.in_channels, 1, 1)),
        #           (1, 1, 1)),
        #     requires_grad=False)

        # manifold = kwds.pop('manifold', None)
        # if manifold is None:
        #     manifold = Lorentz(k=1.0)

        # self.__kernel_size = kernel_size
        # self.__i_c = kernel_size // 2
        # self.__stride = stride
        # self.__pad = padding
        # self.M = manifold
        # self.h_linear = HLinear(in_channels,
        #                         out_channels,
        #                         bias=True,
        #                         manifold=self.M)
        # self.norm = kwds.pop('norm', nn.LayerNorm(out_channels))
        # self.actv = kwds.pop('actv', nn.GELU())

    def forward(self, x):
        # print('got x', x.shape)  # 8, 96, 8
        if self.stride > 1:
            # Insert hyperbolic origin vectors between features
            # x = einops.repeat(x,
            #                   '... s c -> (...) c s one',
            #                   one=1)
            s = x.shape
            x = einops.rearrange(x, '... s c -> (...) c s')
            # print('pre-pad x', x.shape)  # 8, 8, 96
            x = F.conv_transpose1d(x,
                                   self.pad_weight,
                                   stride=self.stride,
                                   padding=1,
                                   groups=self.in_channels)
            # print('post-pad x', x.shape)
            x = x.swapaxes(-1, -2)
            x = x.reshape(*s[:-2], *x.shape[-2:])
            # x = x.permute(0, 2, 3, 1)
        # print('x', x.shape)  # 8, 8, 191 ??
        # print('conv-in', x.shape)
        x = self.conv(x)
        # print('after', x.shape)

        if self.output_padding[0] > 0:
            # print('opad')
            # Pad one side of each dimension (bottom+right) (see PyTorch
            # documentation)
            x = F.pad(x, pad=(0, 0,
                              0, self.output_padding[0]))
            # x[..., 0].clamp_(min=self.manifold.k.sqrt()) # Fix origin padding
        return x

        # x0 = x
        # s = x.shape
        # x = einops.repeat(x, '... s c -> (...) c s one', one=1)
        # channels = x.shape[1]

        # # x = F.unfold(x,
        # #             kernel_size=(self.__kernel_size, 1),
        # #             stride=(self.__stride, 1),
        # #             padding=self.__pad)

        # # print('x', x.shape)  # 8,9,3 N=8, C=9, '*'=3
        # # print('x', u_x.shape)  # 8, 81, 2
        # x = einops.rearrange(x, 'n (c k) s -> (n s) c k', c=channels)
        # # print('x', x.shape)

        # # FIXME(ycho): what if kernel size is even??
        # c = x[..., self.__i_c]

        # # x = tangent_from_global(x, c, M=M)
        # # print('c', c.shape, x0.shape)
        # # print(c - x0.reshape(c.shape))
        # x = self.M.logmap(c[..., None], x, dim=-2)
        # # x = x.squeeze
        # # print('x', x.shape)
        # # print('c', c)
        # # print('log_c(x)', x)
        # # x = x.reshape(*x.shape[:-2], -1)
        # # W = einops.rearrange(self.weight, 'co ci ... -> co (... ci)')
        # # x = F.linear(x, W, self.bias)

        # # x = (B, C, iW)
        # # W = (co, ci, kW)
        # x = F.conv1d(x, self.weight, self.bias).squeeze(dim=-1)
        # x = self.actv(self.norm(x))
        # c1 = self.h_linear(c)
        # x = self.M.expmap(c1, x)
        # x = x.reshape(*s[:-2], -1, self.out_channels)
        # return x


def test_one_layer():
    # net = HConv1d(9, 8, 3, 1, (1, 0))
    net = HConv1d(1, 3, 3, stride=2,
                  # padding=(1, 0)
                  padding=1
                  )
    print(net)
    # z = th.randn((1, 8, 1))
    z = th.rand((96, 8, 1))
    # x = Lorentz(k=1.0).expmap0(z)
    # x2 = Lorentz(k=1.0).projx(z)

    # >> This seems to be the "correct" option <<
    x = PoincareBall(c=1.0).projx(z)
    print('?x', x.mean(), x.std(),
          x.min(), x.max())

    # print('are they different?')
    # print(x, x2)
    print('x', x.shape)
    y = net(x)
    print('y', y.shape)


def test_ae():
    net = nn.Sequential(HConv1d(9, 8, 3, 2, 1),
                        HConvT1d(8, 9, 3, 2, 1,
                                 output_padding=1),
                        )
    x = Lorentz(k=1.0).expmap0(th.randn((8, 64, 9)))
    y = net(x)
    # print(y[:, -2, :])
    # print('y', y.shape)
    # print( (y != 0).float().mean())


def main():
    test_one_layer()
    # test_ae()


if __name__ == '__main__':
    main()
