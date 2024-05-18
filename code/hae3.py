#!/usr/bin/env python3

from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

from hypll import nn as hnn
from hypll.tensors import ManifoldTensor
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor  # ?


# class HConv1d(nn.Module):
#     def __init__(self, in_channels: int,
#                  out_channels: int,
#                  kernel_size: int,
#                  padding: int = 0,
#                  stride: int = 1,
#                  manifold=None):
#         super().__init__()
#         self.conv = hnn.HConvolution2d(
#             in_channels,
#             out_channels,
#             kernel_size=(kernel_size, 1),
#             manifold=manifold,
#             padding=(padding, 0),
#             stride=(stride, 1),
#         )

#     def forward(self, x):
#         """ x shape: ... c s """
#         print('x', x.shape)
#         x = x.unsqueeze(dim=-1,)
#         x.man_dim = 1
#         print('x', x.shape)
#         return self.conv(x).squeeze(dim=-1)

class HConvolution1d(nn.Module):
    """Applies a 2D convolution over a hyperbolic input signal.

    Attributes:
        in_channels:
            Number of channels in the input image.
        out_channels:
            Number of channels produced by the convolution.
        kernel_size:
            Size of the convolving kernel.
        manifold:
            Hyperbolic manifold of the tensors.
        bias:
            If True, adds a learnable bias to the output. Default: True
        stride:
            Stride of the convolution. Default: 1
        padding:
            Padding added to all four sides of the input. Default: 0
        id_init:
            Use identity initialization (True) if appropriate or use HNN++ initialization (False).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        manifold,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size)
        # self.kernel_vol = self.kernel_size[0] * self.kernel_size[1]
        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.id_init = id_init
        self.has_bias = bias

        self.weights, self.bias = self.manifold.construct_dl_parameters(
            in_features=self.kernel_size * in_channels,
            out_features=out_channels,
            bias=self.has_bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameter weights based on the manifold."""
        self.manifold.reset_parameters(weight=self.weights, bias=self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """Does a forward pass of the 2D convolutional layer.

        Args:
            x:
                Manifold tensor of shape (B, C_in, H, W) with manifold dimension 1.

        Returns:
            Manifold tensor of shape (B, C_in, H_out, W_out) with manifold dimension 1.

        Raises:
            ValueError: If the manifolds or manifold dimensions don't match.

        """
        # check_if_manifolds_match(layer=self, input=x)
        # check_if_man_dims_match(layer=self, man_dim=1, input=x)

        batch_size, length = x.size(0), x.size(2)
        out_length = _output_side_length(
            input_side_length=length,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        x = self.manifold.unfold(
            input=x[..., None],
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )
        x = self.manifold.fully_connected(x=x, z=self.weights, bias=self.bias)
        aa = x.tensor.reshape(batch_size, self.out_channels, out_length)
        x = ManifoldTensor(
            data=x.tensor.reshape(batch_size, self.out_channels, out_length),
            manifold=x.manifold,
            man_dim=1,
        )
        return x


HConv1d = HConvolution1d


def _output_side_length(
    input_side_length: int, kernel_size: int, padding: int, stride: int
) -> int:
    """Calculates the output side length of the kernel.

    Based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

    """
    if kernel_size > input_side_length:
        raise RuntimeError(
            f"Encountered invalid kernel size {kernel_size} "
            f"larger than input side length {input_side_length}"
        )
    if stride > input_side_length:
        raise RuntimeError(
            f"Encountered invalid stride {stride} "
            f"larger than input side length {input_side_length}"
        )
    return 1 + (input_side_length + 2 * padding - (kernel_size - 1) - 1) // stride


class HBatchNorm1d(nn.Module):
    """
    1D implementation of hyperbolic batch normalization.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        manifold,
        use_midpoint: bool = False,
    ) -> None:
        super().__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        self.norm = hnn.HBatchNorm(
            features=features,
            manifold=manifold,
            use_midpoint=use_midpoint,
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        # check_if_manifolds_match(layer=self, input=x)
        batch_size, length = x.size(0), x.size(2)
        flat_x = ManifoldTensor(
            data=einops.rearrange(x.tensor, '... c s -> (... s) c'),
            # data=x.tensor.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
            manifold=x.manifold,
            man_dim=-1,
        )
        flat_x = self.norm(flat_x)
        new_tensor = einops.rearrange(flat_x.tensor,
                                      '(n s) c -> n c s', s=length)
        # new_tensor = flat_x.tensor.reshape(batch_size, height, width, self.features).permute(
        #     0, 3, 1, 2
        # )
        return ManifoldTensor(data=new_tensor, manifold=x.manifold, man_dim=1)


class HAvgPool1d(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        manifold,  # : Manifold,
        stride: Optional[int] = None,
        padding: int = 0,
        use_midpoint: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.manifold = manifold
        self.stride = stride if (stride is not None) else self.kernel_size
        self.padding = padding
        self.use_midpoint = use_midpoint

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        # check_if_manifolds_match(layer=self, input=input)
        # check_if_man_dims_match(layer=self, man_dim=1, input=input)

        batch_size, channels, length = input.size()
        # out_height = int((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        out_length = int(
            (length +
             2 *
             self.padding -
             self.kernel_size) /
            self.stride +
            1)

        unfolded_input = self.manifold.unfold(
            input=input[..., None],
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1)
        )
        per_kernel_view = unfolded_input.tensor.view(
            batch_size,
            channels,
            self.kernel_size,  # [0] * self.kernel_size[1],
            unfolded_input.size(-1),
        )

        x = ManifoldTensor(
            data=per_kernel_view,
            manifold=self.manifold,
            man_dim=1)

        if self.use_midpoint:
            aggregates = self.manifold.midpoint(x=x, batch_dim=2)

        else:
            aggregates = self.manifold.frechet_mean(x=x, batch_dim=2)

        return ManifoldTensor(
            data=aggregates.tensor.reshape(batch_size, channels, out_length),
            manifold=self.manifold,
            man_dim=1,
        )


class CBA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = HConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            stride=stride,
            padding=1,
        )
        self.bn1 = HBatchNorm1d(features=out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x: th.Tensor):
        return self.relu(self.bn1(self.conv1(x)))


class PoincareResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        # We can replace each operation in the usual ResidualBlock by a manifold-agnostic
        # operation and supply the PoincareBall object to these operations.
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        self.stride = stride
        self.downsample = downsample

        self.conv1 = HConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            stride=stride,
            padding=1,
        )
        self.bn1 = HBatchNorm1d(features=out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = HConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn2 = HBatchNorm1d(features=out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        # We replace the addition operation inside the skip connection by a
        # Mobius addition.
        x = self.manifold.mobius_add(x, residual)
        x = self.relu(x)

        return x


class PoincareResNet(nn.Module):
    def __init__(
        self,
        channel_sizes: Tuple[int, int, int],
        group_depths: Tuple[int, int, int],
        manifold: PoincareBall,
        num_cls: int = 10,
    ):
        # For the Poincare ResNet itself we again replace each layer by a manifold-agnostic one
        # and supply the PoincareBall to each of these. We also replace the ResidualBlocks by
        # the manifold-agnostic one defined above.
        super().__init__()
        self.channel_sizes = channel_sizes
        self.group_depths = group_depths
        self.manifold = manifold

        self.conv = HConv1d(
            in_channels=3,
            out_channels=channel_sizes[0],
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn = HBatchNorm1d(
            features=channel_sizes[0],
            manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.group1 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[0],
            depth=group_depths[0],
        )
        self.group2 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[1],
            depth=group_depths[1],
            stride=2,
        )
        self.group3 = self._make_group(
            in_channels=channel_sizes[1],
            out_channels=channel_sizes[2],
            depth=group_depths[2],
            stride=2,
        )

        # self.avg_pool = hnn.HAvgPool2d(kernel_size=8, manifold=manifold)
        self.avg_pool = HAvgPool1d(kernel_size=8, manifold=manifold)
        self.fc = hnn.HLinear(
            in_features=channel_sizes[2],
            out_features=num_cls,
            manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        print('x', x.shape)
        x = self.avg_pool(x)
        print('x', x.shape)

        s = x.shape
        x_flat = einops.rearrange(x.tensor, 'n c s -> (n s) c')
        x = ManifoldTensor(
            data=x_flat,
            manifold=x.manifold,
            man_dim=1,
        )
        x = self.fc(x)
        x_back = einops.rearrange(x.tensor, '(n s) c -> n c s',
                                  s=s[-1])
        x = ManifoldTensor(
            data=x_back,
            manifold=x.manifold,
            man_dim=1
        )
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        if stride == 1:
            downsample = None
        else:
            downsample = HConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                manifold=self.manifold,
                stride=stride,
            )

        layers = [
            PoincareResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                manifold=self.manifold,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                PoincareResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=self.manifold,
                )
            )

        return nn.Sequential(*layers)


class Repeat(nn.Module):
    def forward(self, x):
        return ManifoldTensor(
            einops.repeat(x.tensor, '... c s -> ... c (s two)', two=2),
            x.manifold, man_dim=x.man_dim)
        return y

    def extra_repr(self):
        return '... c s -> ... c (s two)'


class PoincareUNet(nn.Module):
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Config):
        super().__init__()
        manifold = PoincareBall(c=Curvature(value=0.1,
                                            requires_grad=True))
        self.manifold = manifold
        self.encoder = nn.Sequential(
            CBA(1, 16, manifold, 2),
            CBA(16, 32, manifold, 2),
            CBA(32, 64, manifold, 2)
        )
        self.decoder = nn.Sequential(
            Repeat(),
            CBA(64, 32, manifold, 1),
            Repeat(),
            CBA(32, 16, manifold, 1),
            Repeat(),
            CBA(16, 4, manifold, 1)
        )
        # self.predictor = nn.Conv1d()
        self.predictor = nn.Sequential(
            nn.Conv1d(4, 1, 3, 1, 1)
        )

    def decode(self, z: th.Tensor) -> th.Tensor:
        # <...batch> _stuff_
        z0 = z
        if not isinstance(z, ManifoldTensor):
            z = einops.rearrange(z, '... s c -> (...) c s')
            if False:
                z = ManifoldTensor(z, self.manifold, man_dim=1)
            else:
                z_T = TangentTensor(data=z,
                                    man_dim=1,
                                    manifold=self.manifold)
                z = self.manifold.expmap(z_T)
        y = self.decoder(z)
        y = self.predictor(y.tensor)
        y = y.swapaxes(-1, -2).reshape(*z0.shape[:-2], *y.shape[-2:])
        return y

    def forward(self, x: th.Tensor):
        s = x.shape
        x = einops.rearrange(x, '... s c -> (...) c s')
        x_T = TangentTensor(data=x,
                            man_dim=1,
                            manifold=self.manifold)
        x_P = self.manifold.expmap(x_T)
        z = self.encoder(x_P)
        y = self.decoder(z)

        y = self.predictor(y.tensor)
        # print('y', y.shape)
        y = y.swapaxes(-1, -2).reshape(s)
        return y

    def loss(self, x, y):
        rec_loss = F.mse_loss(x, y)
        return (rec_loss, 0 * rec_loss)  # th.zeros_like(rec_loss))


def test_restnet():
    # manifold = PoincareBall(th.as_tensor(1.0))
    manifold = PoincareBall(c=Curvature(value=0.1,
                                        requires_grad=True))
    net = PoincareResNet([3, 8, 16], [1, 1, 1],
                         manifold=manifold)
    x = th.zeros((5, 3, 128))
    tangents = TangentTensor(data=x,
                             man_dim=1,
                             manifold=manifold)
    manifold_inputs = manifold.expmap(tangents)
    y = net(manifold_inputs)
    print('y', y.shape)


def test_unet():
    unet = PoincareUNet(PoincareUNet.Config())
    x = th.randn((8, 128, 1))
    y = unet(x)
    print('y', y.shape)


def main():
    test_unet()


if __name__ == '__main__':
    main()
