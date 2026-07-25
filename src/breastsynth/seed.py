"""Global reproducibility control (Reviewer 1: seeds / reproducibility).

A single entry point seeds Python, NumPy and PyTorch, and optionally forces
deterministic cuDNN. The seed actually applied is returned so it can be logged
into every run report.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs used in the pipeline.

    Args:
        seed: base seed.
        deterministic: if True, request deterministic cuDNN kernels. This makes
            results reproducible at a small speed cost. Disable for the fastest
            (but non-deterministic) training.

    Returns:
        The seed that was applied (echoed for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt-in stricter determinism where kernels support it.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch is optional at import time (e.g. manifest building only).
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Deterministic DataLoader worker seeding.

    Cast to a plain Python int and keep the seed strictly < 2**32 (NumPy 2.x
    rejects seeds at or above that bound, including exact-boundary values).
    """
    base = int(np.random.get_state()[1][0])
    seed = (base + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
