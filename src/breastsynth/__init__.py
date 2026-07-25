"""breastsynth — leakage-free synthetic data augmentation for mammography.

See METHODOLOGY.md (repository root) for the authoritative experimental design.
"""

__version__ = "1.0.0"

from breastsynth.seed import seed_everything  # noqa: E402

__all__ = ["seed_everything", "__version__"]
