import logging

import torch

from ..devices import resolve_device
from .features import SPEAKER_INFORMATION_LAYER
from .WavLM import WavLM, WavLMConfig

LOGGER = logging.getLogger(__name__)


def init_wavlm_large(
    pretrained: bool = True, progress: bool = True, device: str | torch.device = "cuda"
) -> torch.nn.Module:
    """Load the WavLM large checkpoint using the original unilm implementation."""
    if not pretrained:
        raise ValueError(
            "pretrained=False is not supported for WavLM large because we read "
            "the large configuration dict locally from the checkpoint."
        )

    device = resolve_device(device)

    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt",
        map_location="cpu",
        progress=progress,
    )

    cfg_dict = checkpoint["cfg"]
    if not isinstance(cfg_dict, dict):
        cfg_dict = vars(cfg_dict)

    cfg = WavLMConfig(cfg_dict)
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])

    setattr(model, "extract_from_layer", SPEAKER_INFORMATION_LAYER)

    model = model.to(device)
    model.eval()

    LOGGER.info(
        "WavLM-Large loaded with %s parameters.",
        f"{sum(p.numel() for p in model.parameters()):,d}",
    )

    return model
