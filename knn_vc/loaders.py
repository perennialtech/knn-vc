from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import torch

from .devices import resolve_device
from .hifigan.models import Generator as HiFiGAN
from .hifigan.utils import AttrDict
from .matcher import KNeighborsVC
from .wavlm import init_wavlm_large

DEFAULT_HIFIGAN_CONFIG = "config_v1_wavlm.json"

_REQUIRED_HIFIGAN_CONFIG_KEYS = {
    "resblock",
    "upsample_rates",
    "upsample_kernel_sizes",
    "upsample_initial_channel",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "hubert_dim",
    "hifi_dim",
    "hop_size",
    "sampling_rate",
}

LOGGER = logging.getLogger(__name__)


def _read_hifigan_config_text(config_path: str | Path | None) -> str:
    if config_path is None:
        return (
            resources.files("knn_vc.hifigan")
            .joinpath(DEFAULT_HIFIGAN_CONFIG)
            .read_text()
        )

    path = Path(config_path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            "HiFi-GAN generator configs must be JSON. The training YAML is not "
            "accepted by the inference loader."
        )

    return path.read_text()


def _validate_hifigan_config(config: AttrDict) -> None:
    missing = sorted(key for key in _REQUIRED_HIFIGAN_CONFIG_KEYS if key not in config)
    if missing:
        raise ValueError(f"Missing HiFi-GAN config keys: {', '.join(missing)}")

    if len(config.upsample_rates) != len(config.upsample_kernel_sizes):
        raise ValueError(
            "upsample_rates and upsample_kernel_sizes must have the same length"
        )

    if config.resblock not in {"1", "2"}:
        raise ValueError(f"resblock must be '1' or '2', got {config.resblock!r}")


def _safe_torch_load(path: str | Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_generator_state_dict(checkpoint: object) -> Mapping[str, object]:
    if isinstance(checkpoint, Mapping) and "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise ValueError("HiFi-GAN checkpoint must contain a generator state dict")

    return state_dict


def _load_hifigan_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Mapping[str, object]:
    checkpoint = _safe_torch_load(checkpoint_path, device)
    return _extract_generator_state_dict(checkpoint)


def load_hifigan_config(config_path: str | Path | None = None) -> AttrDict:
    """Load a HiFi-GAN generator config."""

    config = AttrDict(json.loads(_read_hifigan_config_text(config_path)))
    _validate_hifigan_config(config)
    return config


def load_hifigan_wavlm(
    pretrained: bool = True,
    progress: bool = True,
    prematched: bool = True,
    device: str | torch.device | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    remove_weight_norm: bool | None = None,
) -> tuple[HiFiGAN, AttrDict]:
    """Load a HiFi-GAN generator trained to vocode WavLM features."""

    resolved_device = resolve_device(device)
    config = load_hifigan_config(config_path)
    generator = HiFiGAN(config).to(resolved_device)

    if remove_weight_norm is None:
        remove_weight_norm = pretrained or checkpoint_path is not None

    if checkpoint_path is not None:
        generator.load_state_dict(
            _load_hifigan_checkpoint(checkpoint_path, resolved_device)
        )
    elif pretrained:
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

    if remove_weight_norm:
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
) -> torch.nn.Module:
    """Load WavLM large using the native implementation."""

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
    hifigan_checkpoint_path: str | Path | None = None,
    hifigan_config_path: str | Path | None = None,
) -> KNeighborsVC:
    """Load the complete kNN-VC pipeline."""

    if not pretrained:
        raise ValueError(
            "load_knn_vc(pretrained=False) is not supported because torchaudio "
            "does not expose an uninitialized WAVLM_LARGE constructor."
        )

    resolved_device = resolve_device(device)
    hifigan, hifigan_cfg = load_hifigan_wavlm(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=resolved_device,
        config_path=hifigan_config_path,
        checkpoint_path=hifigan_checkpoint_path,
    )
    wavlm = load_wavlm_large(
        pretrained=pretrained,
        progress=progress,
        device=resolved_device,
    )

    return KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=resolved_device)
