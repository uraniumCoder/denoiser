# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math

import torch as th
from torch.nn import functional as F

LINSPACE = th.tensor([-55.5000, -54.5000, -53.5000, -52.5000, -51.5000, -50.5000, -49.5000,
        -48.5000, -47.5000, -46.5000, -45.5000, -44.5000, -43.5000, -42.5000,
        -41.5000, -40.5000, -39.5000, -38.5000, -37.5000, -36.5000, -35.5000,
        -34.5000, -33.5000, -32.5000, -31.5000, -30.5000, -29.5000, -28.5000,
        -27.5000, -26.5000, -25.5000, -24.5000, -23.5000, -22.5000, -21.5000,
        -20.5000, -19.5000, -18.5000, -17.5000, -16.5000, -15.5000, -14.5000,
        -13.5000, -12.5000, -11.5000, -10.5000,  -9.5000,  -8.5000,  -7.5000,
         -6.5000,  -5.5000,  -4.5000,  -3.5000,  -2.5000,  -1.5000,  -0.5000,
          0.5000,   1.5000,   2.5000,   3.5000,   4.5000,   5.5000,   6.5000,
          7.5000,   8.5000,   9.5000,  10.5000,  11.5000,  12.5000,  13.5000,
         14.5000,  15.5000,  16.5000,  17.5000,  18.5000,  19.5000,  20.5000,
         21.5000,  22.5000,  23.5000,  24.5000,  25.5000,  26.5000,  27.5000,
         28.5000,  29.5000,  30.5000,  31.5000,  32.5000,  33.5000,  34.5000,
         35.5000,  36.5000,  37.5000,  38.5000,  39.5000,  40.5000,  41.5000,
         42.5000,  43.5000,  44.5000,  45.5000,  46.5000,  47.5000,  48.5000,
         49.5000,  50.5000,  51.5000,  52.5000,  53.5000,  54.5000,  55.5000])

def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = LINSPACE
    # t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = LINSPACE
    # t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    # if x.shape[-1] % 2 != 0:
        # x = F.pad(x, (0, 1))
    assert x.shape[-1] % 2 == 0
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, time).mul(0.5)
