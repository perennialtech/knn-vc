import logging

import torch
from torchaudio.pipelines import WAVLM_LARGE

from ..devices import resolve_device
from .features import SPEAKER_INFORMATION_LAYER

LOGGER = logging.getLogger(__name__)


def init_wavlm_large(
    pretrained: bool = True, progress: bool = True, device: str | torch.device = "cuda"
) -> torch.nn.Module:
    """Load the WavLM large checkpoint from torchaudio pipelines.
    This replaces the legacy unilm/fairseq implementation.
    """
    if not pretrained:
        raise ValueError(
            "pretrained=False is not supported for WavLM large because torchaudio "
            "does not expose an uninitialized WAVLM_LARGE constructor."
        )

    device = resolve_device(device)

    model = WAVLM_LARGE.get_model(dl_kwargs={"progress": progress})

    setattr(model, "extract_from_layer", SPEAKER_INFORMATION_LAYER)

    model = model.to(device)
    model.eval()

    LOGGER.info(
        "WavLM-Large loaded with %s parameters.",
        f"{sum(p.numel() for p in model.parameters()):,d}",
    )

    return model
