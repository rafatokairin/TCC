"""Conditional StyleGAN2-ADA generator/discriminator (Reviewer 1: implementation detail).

Module/attribute names match the released checkpoint so `state_dict` loads
cleanly. The generator block forward is the standard style-modulation path
(conv -> noise -> LReLU -> AdaIN, twice). For the leakage-free protocol the
generator is retrained on the DEV partition only.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from breastsynth.models.layers import (
    DiscriminatorBlock,
    EqualizedConv2d,
    MappingNetwork,
    StyleGAN2GeneratorBlock,
)


def one_hot_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(labels, n_classes).float()


def combine_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, y), dim=1)


class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, num_classes, out_chan=1, base_ch=32, target_size=128):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=8)
        self.initial_constant = nn.Parameter(torch.ones(1, base_ch, 4, 4))
        self.initial_block = StyleGAN2GeneratorBlock(base_ch, base_ch, w_dim, initial_block=True)

        num_layers = int(math.log2(target_size) - 2)
        channels = base_ch
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            out_ch = max(base_ch // (2 ** (i + 1)), 8)
            self.blocks.append(StyleGAN2GeneratorBlock(channels, out_ch, w_dim))
            channels = out_ch
        self.to_rgb = EqualizedConv2d(channels, out_chan, 1)

    def forward(self, zc: torch.Tensor) -> torch.Tensor:
        w = self.mapping(zc)
        x = self.initial_constant.expand(zc.size(0), -1, -1, -1)
        x = self.initial_block(x, w)
        for block in self.blocks:
            x = block(x, w)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, in_chan, n_classes, base_ch=32, target_size=128):
        super().__init__()
        n_blocks = int(math.log2(target_size)) - 2
        ch = base_ch
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_chan + n_classes, ch, 1), nn.LeakyReLU(0.2)
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(DiscriminatorBlock(ch, min(512, ch * 2)))
            ch = min(512, ch * 2)
        self.final_block = nn.Sequential(
            EqualizedConv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(ch, 1, 4, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return self.final_block(x).view(-1)


class AdaptiveAugment:
    """Adaptive Discriminator Augmentation (ADA) probability controller."""

    def __init__(self, ada_target=0.6, ada_speed=0.01, device="cuda"):
        import torchvision.transforms as T

        self.ada_target = ada_target
        self.ada_speed = ada_speed
        self.device = device
        self.ada_p = 0.0
        self.ada_r_t = torch.tensor(0.0, device=device)
        self.augment = T.Compose(
            [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=5, fill=0)]
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.ada_p > 0:
            mask = torch.rand(x.size(0), device=self.device) < self.ada_p
            if mask.any():
                x = x.clone()
                x[mask] = torch.stack([self.augment(img) for img in x[mask]])
        return x

    def update(self, real_logits: torch.Tensor) -> float:
        signs = torch.sign(real_logits).mean()
        self.ada_r_t = (1 - self.ada_speed) * self.ada_r_t + self.ada_speed * signs
        self.ada_p = min(1.0, max(0.0, self.ada_p + self.ada_speed * (self.ada_r_t - self.ada_target)))
        return self.ada_p


def build_generator(cfg) -> Generator:
    """Instantiate the generator from a GanConfig (input includes one-hot label)."""
    return Generator(
        z_dim=cfg.z_dim + cfg.n_classes,
        w_dim=cfg.w_dim,
        num_classes=cfg.n_classes,
        out_chan=1,
        base_ch=cfg.base_ch,
        target_size=getattr(cfg, "image_size", 128),
    )


def load_generator(cfg, ckpt_path: str, device: str = "cuda") -> Generator:
    """Load a trained generator checkpoint (expects a dict with a 'gen' key)."""
    gen = build_generator(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["gen"] if isinstance(ckpt, dict) and "gen" in ckpt else ckpt
    gen.load_state_dict(state)
    gen.eval()
    return gen
