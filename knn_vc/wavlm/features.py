from collections.abc import Iterable
import torch
from torch import Tensor

MAX_WAVLM_LAYER = 24
SPEAKER_INFORMATION_LAYER = 6

def validate_wavlm_layer(layer: int, name: str = "layer") -> None:
    if not 1 <= layer <= MAX_WAVLM_LAYER:
        raise ValueError(f"{name} must be between 1 and {MAX_WAVLM_LAYER}, got {layer}")

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

@torch.inference_mode()
def extract_wavlm_layers(
    wavlm,
    waveform: Tensor,
    layers: Iterable[int],
) -> dict[int, Tensor]:
    """
    Return selected one-based transformer-layer features.

    The released kNN-VC vocoders were trained with the original WavLM code,
    where `output_layer=6` returns the sixth transformer block output. Torchaudio
    returns those transformer outputs in a zero-based list, so public layer id 6
    maps to `transformer_features[5]`.
    """

    _validate_single_waveform(waveform)

    layer_ids = _sorted_unique_layers(layers)

    if not layer_ids:
        return {}

    ((_, layer_results), _) = wavlm.extract_features(
        waveform,
        output_layer=max(layer_ids),
        ret_layer_results=True,
    )

    selected: dict[int, Tensor] = {}
    for layer in layer_ids:
        selected[layer] = layer_results[layer][0].transpose(0, 1).squeeze(0)

    return selected
