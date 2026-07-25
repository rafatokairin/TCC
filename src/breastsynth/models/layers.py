"""Style-based building blocks.

Attribute names are kept identical to the original implementation so that the
released checkpoint (`data/ckpt400.pth`, Git LFS) loads without key remapping.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.stride = stride
        self.padding = padding
        self.scale = math.sqrt(2) / math.sqrt(in_ch * kernel_size * kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = math.sqrt(2) / math.sqrt(in_dim)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers=8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = EqualizedLinear(w_dim, channels)
        self.style_bias = EqualizedLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w)[:, :, None, None]
        style_bias = self.style_bias(w)[:, :, None, None]
        return style_scale * x + style_bias


class StyleGAN2GeneratorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, initial_block=False):
        super().__init__()
        self.initial_block = initial_block
        if initial_block:
            self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                EqualizedConv2d(in_ch, out_ch, 3, padding=1),
            )
        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.adain1 = AdaIN(out_ch, w_dim)
        self.adain2 = AdaIN(out_ch, w_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, w):
        batch_size = x.size(0)
        x = self.conv1(x)
        noise1 = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_scale1 * noise1
        x = self.leaky_relu(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        noise2 = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_scale2 * noise2
        x = self.leaky_relu(x)
        x = self.adain2(x, w)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(2), EqualizedConv2d(in_ch, out_ch, 1))
        self.block = nn.Sequential(
            EqualizedConv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        self.alpha = 0.0

    def forward(self, x):
        if self.alpha < 1.0:
            return self.alpha * self.block(x) + (1 - self.alpha) * self.residual(x)
        return self.block(x)
