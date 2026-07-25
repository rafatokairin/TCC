"""Train EfficientNet-B0 with early stopping (Reviewer 1: implementation detail).

The trainer receives explicit train / inner-val path lists so the caller fully
controls composition (real + synthetic in train; DEV-only reals in val). Early
stopping monitors inner-val AUC. The held-out TEST set is NEVER passed here.
"""
from __future__ import annotations

import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from breastsynth.config import ClassifierConfig
from breastsynth.data.dataset import MammogramDataset, build_transforms
from breastsynth.metrics.classification import classification_metrics
from breastsynth.models.classifier import build_classifier, predict_probs
from breastsynth.seed import worker_init_fn


def _loader(paths, labels, cfg, train, seed):
    ds = MammogramDataset(paths, labels, build_transforms(cfg.image_size, train=train))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g,
    )


def train_classifier(
    train_paths: list[str],
    train_labels: list[int],
    val_paths: list[str],
    val_labels: list[int],
    cfg: ClassifierConfig,
    device: str = "cuda",
    seed: int = 42,
) -> tuple[nn.Module, dict]:
    """Train and return (best_model, history). Selection metric = inner-val AUC."""
    train_loader = _loader(train_paths, train_labels, cfg, True, seed)
    val_loader = _loader(val_paths, val_labels, cfg, False, seed)

    model = build_classifier(cfg.pretrained, cfg.freeze_features, num_outputs=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=cfg.lr)

    best_auc, best_state, epochs_no_improve = -1.0, None, 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        y_true, y_prob = predict_probs(model, val_loader, device)
        val = classification_metrics(y_true, y_prob)
        history.append({"epoch": epoch, **val})

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": best_auc, "history": history}
