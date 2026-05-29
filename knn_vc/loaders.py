from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path

import torch
from torchaudio.models import Wav2Vec2Model

from .hifigan.models import Generator as HiFiGAN
from .hifigan.utils import AttrDict
from .matcher import KNeighborsVC
from .wavlm import init_wavlm_large

DEFAULT_HIFIGAN_CONFIG = "config_v1_wavlm.json"

SPEAKER_INFORMATION_LAYER = 6

LOGGER = logging.getLogger(__name__)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the requested torch device, falling back from CUDA when needed."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Overriding device %s to cpu since CUDA is unavailable.", device)
        return torch.device("cpu")

    return resolved


def load_hifigan_config(config_path: str | Path | None = None) -> AttrDict:
    """Load a HiFi-GAN config."""

    if config_path is None:
        config_text = (
            resources.files("knn_vc.hifigan")
            .joinpath(DEFAULT_HIFIGAN_CONFIG)
            .read_text()
        )
    else:
        config_text = Path(config_path).read_text()

    return AttrDict(json.loads(config_text))


def load_hifigan_wavlm(
    pretrained: bool = True,
    progress: bool = True,
    prematched: bool = True,
    device: str | torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[HiFiGAN, AttrDict]:
    """Load a HiFi-GAN generator trained to vocode WavLM features."""

    resolved_device = resolve_device(device)
    config = load_hifigan_config(config_path)
    generator = HiFiGAN(config).to(resolved_device)

    if pretrained:
        if prematched:
            url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
        else:
            url = (
                "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt"
            )

        state_dict_g = torch.hub.load_state_dict_from_url(
            url,
            map_location=resolved_device,
            progress=progress,
        )
        generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()

    LOGGER.info(
        "Loaded HiFi-GAN generator with %s parameters.",
        f"{sum(p.numel() for p in generator.parameters()):,d}",
    )

    return generator, config


def load_wavlm_large(
    pretrained: bool = True,
    progress: bool = True,
    device: str | torch.device | None = None,
) -> Wav2Vec2Model:
    """Load WavLM large from torchaudio."""

    resolved_device = resolve_device(device)
    return init_wavlm_large(
        pretrained=pretrained,
        progress=progress,
        device=resolved_device,
    )


def load_knn_vc(
    pretrained: bool = True,
    progress: bool = True,
    prematched: bool = True,
    device: str | torch.device | None = None,
) -> KNeighborsVC:
    """Load the complete kNN-VC pipeline."""

    resolved_device = resolve_device(device)
    hifigan, hifigan_cfg = load_hifigan_wavlm(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=resolved_device,
    )
    wavlm = load_wavlm_large(
        pretrained=pretrained,
        progress=progress,
        device=resolved_device,
    )

    return KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=resolved_device)
