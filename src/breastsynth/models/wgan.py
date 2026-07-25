"""Conditional WGAN-GP baseline generator/discriminator.

Retained as the comparison baseline reported in the paper (StyleGAN2-ADA vs a
convolutional cWGAN-GP). Condensed and modularised from the original prototype.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def one_hot_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(labels, n_classes)


def combine_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x.float(), y.float()), 1)


class WGANGenerator(nn.Module):
    def __init__(self, input_dim, im_chan=1, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self._block(input_dim, hidden_dim * 16, kernel_size=4, stride=1, padding=0),
            self._block(hidden_dim * 16, hidden_dim * 8),
            self._block(hidden_dim * 8, hidden_dim * 4),
            self._block(hidden_dim * 4, hidden_dim * 2),
            self._block(hidden_dim * 2, hidden_dim),
            self._block(hidden_dim, hidden_dim // 2),
            self._block(hidden_dim // 2, im_chan, kernel_size=4, stride=2, padding=1, final=True),
        )

    def _block(self, ci, co, kernel_size=4, stride=2, padding=1, final=False):
        if final:
            return nn.Sequential(nn.ConvTranspose2d(ci, co, kernel_size, stride, padding), nn.Tanh())
        return nn.Sequential(
            nn.ConvTranspose2d(ci, co, kernel_size, stride, padding),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.gen(noise.view(len(noise), self.input_dim, 1, 1))


class WGANDiscriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super().__init__()
        self.disc = nn.Sequential(
            self._block(im_chan, hidden_dim // 2),
            self._block(hidden_dim // 2, hidden_dim),
            self._block(hidden_dim, hidden_dim * 2),
            self._block(hidden_dim * 2, hidden_dim * 4),
            self._block(hidden_dim * 4, hidden_dim * 8),
            self._block(hidden_dim * 8, hidden_dim * 16),
            self._block(hidden_dim * 16, 1, kernel_size=4, stride=1, padding=0, final=True),
        )

    def _block(self, ci, co, kernel_size=4, stride=2, padding=1, final=False):
        if final:
            return nn.Sequential(nn.Conv2d(ci, co, kernel_size, stride, padding))
        return nn.Sequential(
            nn.Conv2d(ci, co, kernel_size, stride, padding),
            nn.BatchNorm2d(co),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, image):
        return self.disc(image).view(len(image), -1)


def gradient_penalty(disc, real, fake, image_one_hot_labels, device):
    """WGAN-GP gradient penalty on interpolated (real, fake) samples."""
    bs = real.shape[0]
    epsilon = torch.rand(bs, 1, 1, 1, device=device, requires_grad=True)
    interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    mixed = combine_vectors(interpolated, image_one_hot_labels)
    scores = disc(mixed)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
