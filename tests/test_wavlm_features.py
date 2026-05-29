import pytest
import torch

from knn_vc.wavlm import extract_wavlm_layers, validate_wavlm_layer


class _FakeWavLM:
    def __init__(self):
        self.num_layers = 0

    def extract_features(self, waveform, output_layer=None, ret_layer_results=False):
        self.num_layers = output_layer
        batch_size, n_frames = waveform.shape

        layer_results = []
        for layer in range(output_layer + 1):
            feat = torch.full((n_frames, batch_size, 2), float(layer))
            layer_results.append((feat, None))

        return ((None, layer_results), None)


def test_extract_wavlm_layers_selects_one_based_transformer_layers():
    wavlm = _FakeWavLM()
    selected = extract_wavlm_layers(wavlm, torch.ones(1, 3), {1, 3})

    assert wavlm.num_layers == 3
    assert set(selected) == {1, 3}
    assert torch.equal(selected[1], torch.full((3, 2), 1.0))
    assert torch.equal(selected[3], torch.full((3, 2), 3.0))


def test_validate_wavlm_layer_rejects_out_of_range_layer():
    with pytest.raises(ValueError, match="speaker_layer"):
        validate_wavlm_layer(0, "speaker_layer")

    with pytest.raises(ValueError, match="speaker_layer"):
        validate_wavlm_layer(25, "speaker_layer")


def test_extract_wavlm_layers_rejects_batched_waveforms():
    with pytest.raises(ValueError, match="single waveform"):
        extract_wavlm_layers(_FakeWavLM(), torch.ones(2, 3), {1})
