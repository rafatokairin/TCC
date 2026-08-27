"""Train the conditional StyleGAN2-ADA generator on a GIVEN set of images.

The training set is passed in explicitly (paths + labels), which is what enforces
the leakage-free contract: callers pass only DEV images (hold-out protocol) or
only the current fold's training images (fold-wise protocol). The generator can
therefore never see the evaluation data.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from breastsynth.config import GanConfig
from breastsynth.models.stylegan2ada import (
    AdaptiveAugment,
    Discriminator,
    Generator,
    combine_vectors,
    one_hot_labels,
)


class _GrayDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], img_size: int = 128):
        self.paths = paths
        self.labels = labels
        self.t = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.t(Image.open(self.paths[i]).convert("L"))
        return img, torch.tensor(int(self.labels[i]), dtype=torch.long)


def _d_logistic_loss(real_pred, fake_pred):
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def _g_logistic_loss(fake_pred):
    return F.softplus(-fake_pred).mean()


def _r1_loss(real_pred, real_img, gamma):
    (grad,) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    return gamma / 2 * grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def train_stylegan2ada(
    paths: list[str],
    labels: list[int],
    cfg: GanConfig,
    out_ckpt: str | Path,
    device: str = "cuda",
    log_every: int = 50,
) -> Path:
    """Train StyleGAN2-ADA on (paths, labels); save checkpoint to `out_ckpt`.

    Returns the checkpoint path. The checkpoint dict contains keys
    {'gen','disc','gen_opt','disc_opt','epoch','config'} so it is fully
    reloadable and self-describing.
    """
    img_size = getattr(cfg, "image_size", 128)
    ds = _GrayDataset(paths, labels, img_size)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    gen = Generator(cfg.z_dim + cfg.n_classes, cfg.w_dim, cfg.n_classes, 1, cfg.base_ch, img_size).to(device)
    disc = Discriminator(1, cfg.n_classes, cfg.base_ch, img_size).to(device)
    opt_g = optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(disc.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    ada = AdaptiveAugment(cfg.ada_target, cfg.ada_speed, device)
    scaler_g, scaler_d = GradScaler(enabled=cfg.mixed_precision), GradScaler(enabled=cfg.mixed_precision)

    def cond(one_hot):
        return one_hot[:, :, None, None].repeat(1, 1, img_size, img_size)

    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(loader, desc=f"GAN epoch {epoch}/{cfg.epochs}")
        for i, (real, lbl) in enumerate(pbar):
            real, lbl = real.to(device), lbl.to(device)
            real = ada.apply(real)
            bs = real.size(0)
            oh = one_hot_labels(lbl, cfg.n_classes).to(device)
            zc = combine_vectors(torch.randn(bs, cfg.z_dim, device=device), oh)

            # Discriminator
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = gen(zc)
            real_in = combine_vectors(real, cond(oh))
            fake_in = combine_vectors(fake.detach(), cond(oh))
            with autocast(enabled=cfg.mixed_precision):
                real_pred = disc(real_in)
                fake_pred = disc(fake_in)
                d_loss = _d_logistic_loss(real_pred, fake_pred)
            if i % cfg.r1_interval == 0:
                real_reg = real_in.detach().requires_grad_(True)
                real_pred_reg = disc(real_reg)
                d_loss = d_loss + _r1_loss(real_pred_reg, real_reg, cfg.r1_gamma)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            ada.update(real_pred.detach())

            # Generator
            opt_g.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.mixed_precision):
                fake = gen(zc)
                fake_pred = disc(combine_vectors(fake, cond(oh)))
                g_loss = _g_logistic_loss(fake_pred)
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            if i % log_every == 0:
                pbar.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}", ada_p=f"{ada.ada_p:.2f}")

    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gen": gen.state_dict(),
            "disc": disc.state_dict(),
            "gen_opt": opt_g.state_dict(),
            "disc_opt": opt_d.state_dict(),
            "epoch": cfg.epochs,
            "config": cfg.__dict__,
        },
        out_ckpt,
    )
    return out_ckpt
