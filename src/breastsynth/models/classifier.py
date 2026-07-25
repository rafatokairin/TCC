"""EfficientNet-B0 classifier factory.

Cite the ORIGINAL EfficientNet paper (Tan & Le, ICML 2019) for the architecture
(Reviewer 1). Uses the modern torchvision `weights=` API (no deprecated
`pretrained=` flag).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_classifier(
    pretrained: bool = True,
    freeze_features: bool = True,
    num_outputs: int = 1,
) -> nn.Module:
    """EfficientNet-B0 with a single-logit binary head.

    Args:
        pretrained: load ImageNet weights.
        freeze_features: freeze the convolutional backbone (train only the head).
        num_outputs: 1 -> single logit for BCEWithLogitsLoss.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


def unfreeze_features(model: nn.Module) -> None:
    """Enable fine-tuning of the backbone (optional second training phase)."""
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def predict_probs(model: nn.Module, loader, device: str) -> tuple[list[int], list[float]]:
    """Return (labels, positive-class probabilities) over a loader."""
    model.eval()
    labels_all, probs_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits)
        probs_all.extend(probs.cpu().tolist())
        labels_all.extend(labels.tolist())
    return labels_all, probs_all
