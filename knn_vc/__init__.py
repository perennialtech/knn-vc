"""Public API for kNN-VC."""

from .loaders import (load_hifigan_config, load_hifigan_wavlm, load_knn_vc,
                      load_wavlm_large)
from .matcher import (DEFAULT_FEATURE_LOUDNESS_CEILING_DB, KNeighborsVC,
                      fast_cosine_dist)

__all__ = [
    "DEFAULT_FEATURE_LOUDNESS_CEILING_DB",
    "KNeighborsVC",
    "fast_cosine_dist",
    "load_hifigan_config",
    "load_hifigan_wavlm",
    "load_knn_vc",
    "load_wavlm_large",
]
