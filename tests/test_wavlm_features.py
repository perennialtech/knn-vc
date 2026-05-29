import pytest
import torch

from knn_vc.wavlm import extract_wavlm_layers, validate_wavlm_layer


class _FakeFeatureProjection:
    def __call__(self, features):
        return features + 100


class _FakeEncoder:
    def __init__(self):
        self.feature_projection = _FakeFeatureProjection()


class _FakeWavLM:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def extract_features(self, waveform, num_layers):
        batch_size, n_frames = waveform.shape
        features = [
            torch.full((batch_size, n_frames, 2), float(layer + 1))
            for layer in range(num_layers)
        ]
        return features, None

    def feature_extractor(self, waveform, lengths):
        batch_size, n_frames = waveform.shape
        return torch.zeros(batch_size, n_frames, 2), None


def test_extract_wavlm_layers_selects_conv_and_transformer_layers():
    selected = extract_wavlm_layers(_FakeWavLM(), torch.ones(1, 3), {0, 2})

    assert set(selected) == {0, 2}
    assert torch.equal(selected[0], torch.full((3, 2), 100.0))
    assert torch.equal(selected[2], torch.full((3, 2), 2.0))


def test_validate_wavlm_layer_rejects_out_of_range_layer():
    with pytest.raises(ValueError, match="speaker_layer"):
        validate_wavlm_layer(25, "speaker_layer")


def test_extract_wavlm_layers_rejects_batched_waveforms():
    with pytest.raises(ValueError, match="single waveform"):
        extract_wavlm_layers(_FakeWavLM(), torch.ones(2, 3), {1})
