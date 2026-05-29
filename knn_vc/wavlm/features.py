from collections.abc import Iterable

import torch
from torch import Tensor

MAX_WAVLM_LAYER = 24
SPEAKER_INFORMATION_LAYER = 6


def validate_wavlm_layer(layer: int, name: str = "layer") -> None:
    if not 0 <= layer <= MAX_WAVLM_LAYER:
        raise ValueError(f"{name} must be between 0 and {MAX_WAVLM_LAYER}, got {layer}")


def _sorted_unique_layers(layers: Iterable[int]) -> tuple[int, ...]:
    layer_ids = tuple(sorted(set(layers)))

    for layer in layer_ids:
        validate_wavlm_layer(layer)

    return layer_ids


def _validate_single_waveform(waveform: Tensor) -> None:
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        raise ValueError(
            "waveform must contain a single waveform with shape (1, samples), "
            f"got {tuple(waveform.shape)}"
        )


def _project_convolutional_features(wavlm, waveform: Tensor) -> Tensor:
    conv_features, _ = wavlm.feature_extractor(waveform, None)
    projected = wavlm.encoder.feature_projection(conv_features)

    if isinstance(projected, tuple):
        projected = projected[0]

    return projected.squeeze(0)


@torch.inference_mode()
def extract_wavlm_layers(
    wavlm,
    waveform: Tensor,
    layers: Iterable[int],
) -> dict[int, Tensor]:
    _validate_single_waveform(waveform)

    layer_ids = _sorted_unique_layers(layers)

    if not layer_ids:
        return {}

    selected: dict[int, Tensor] = {}
    max_transformer_layer = max((layer for layer in layer_ids if layer > 0), default=0)

    if max_transformer_layer:
        transformer_features, _ = wavlm.extract_features(
            waveform,
            num_layers=max_transformer_layer,
        )

        for layer in layer_ids:
            if layer > 0:
                selected[layer] = transformer_features[layer - 1].squeeze(0)

    if 0 in layer_ids:
        selected[0] = _project_convolutional_features(wavlm, waveform)

    return selected
