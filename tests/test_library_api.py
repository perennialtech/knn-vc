import json
import math

import pytest
import torch
import torchaudio

import knn_vc
from knn_vc import (DEFAULT_FEATURE_LOUDNESS_CEILING_DB, KNeighborsVC,
                    fast_cosine_dist, load_hifigan_config, load_hifigan_wavlm,
                    load_knn_vc, load_wavlm_large)
from knn_vc.hifigan.utils import AttrDict
from knn_vc.matcher import attenuate_loud_waveform


class _FakeHiFiGAN(torch.nn.Module):
    def __init__(self, amplitude: float = 0.0):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, x):
        samples = x.shape[1] * 320

        if self.amplitude == 0.0:
            return torch.zeros(x.shape[0], 1, samples, device=x.device)

        time = torch.arange(samples, device=x.device, dtype=x.dtype)
        wave = self.amplitude * torch.sin(math.tau * 440.0 * time / 16_000.0)
        return wave.view(1, 1, samples).expand(x.shape[0], 1, samples)


class _FakeWavLM(torch.nn.Module):
    pass


def _fake_knn_vc(hifigan: torch.nn.Module | None = None) -> KNeighborsVC:
    return KNeighborsVC(
        _FakeWavLM(),
        hifigan if hifigan is not None else _FakeHiFiGAN(),
        AttrDict({"sampling_rate": 16000, "hop_size": 320, "hubert_dim": 2}),
        device="cpu",
    )


def test_public_api_exports_core_symbols():
    assert "DEFAULT_FEATURE_LOUDNESS_CEILING_DB" in knn_vc.__all__
    assert DEFAULT_FEATURE_LOUDNESS_CEILING_DB is None
    assert "KNeighborsVC" in knn_vc.__all__
    assert "load_knn_vc" in knn_vc.__all__
    assert "fast_cosine_dist" in knn_vc.__all__


def test_fast_cosine_dist_matches_expected_neighbors():
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pool = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    dists = fast_cosine_dist(source, pool)

    assert torch.equal(dists.argmin(dim=-1), torch.tensor([0, 1]))
    assert torch.allclose(dists[0], torch.tensor([0.0, 1.0, 2.0]))


def test_load_hifigan_config_reads_packaged_json():
    config = load_hifigan_config()

    assert hasattr(config, "hubert_dim")
    assert hasattr(config, "sampling_rate")
    assert len(config.upsample_rates) == len(config.upsample_kernel_sizes)


def test_load_hifigan_config_rejects_missing_required_keys(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))

    with pytest.raises(ValueError, match="Missing HiFi-GAN config keys"):
        load_hifigan_config(config_path)


def test_load_hifigan_config_rejects_training_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("hifigan: {}\n")

    with pytest.raises(ValueError, match="JSON"):
        load_hifigan_config(config_path)


def test_load_hifigan_wavlm_loads_local_generator_checkpoint(tmp_path):
    generator, _ = load_hifigan_wavlm(
        pretrained=False,
        device="cpu",
        remove_weight_norm=False,
    )
    checkpoint_path = tmp_path / "g.pt"
    torch.save({"generator": generator.state_dict()}, checkpoint_path)

    loaded, _ = load_hifigan_wavlm(
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device="cpu",
        remove_weight_norm=False,
    )

    assert isinstance(loaded, torch.nn.Module)


def test_load_wavlm_large_rejects_random_initialization_before_download():
    with pytest.raises(ValueError, match="pretrained=False"):
        load_wavlm_large(pretrained=False, device="cpu")


def test_load_knn_vc_rejects_random_initialization_before_download():
    with pytest.raises(ValueError, match="pretrained=False"):
        load_knn_vc(pretrained=False, device="cpu")


def test_attrdict_missing_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        AttrDict({}).missing


def test_fast_cosine_dist_rejects_mismatched_feature_dims():
    with pytest.raises(ValueError, match="feature dimensions"):
        fast_cosine_dist(torch.ones(2, 3), torch.ones(4, 2))


def test_get_matching_set_rejects_empty_references():
    with pytest.raises(ValueError, match="at least one"):
        _fake_knn_vc().get_matching_set([])


def test_match_rejects_mismatched_feature_dims():
    model = _fake_knn_vc()

    with pytest.raises(ValueError, match="feature dimensions"):
        model.match(
            torch.ones(2, 3),
            torch.ones(4, 2),
            tgt_loudness_db=None,
        )


def test_match_target_duration_returns_requested_sample_count():
    model = _fake_knn_vc()

    out = model.match(
        torch.ones(2, 2),
        torch.ones(4, 2),
        topk=2,
        target_duration=0.1,
        tgt_loudness_db=None,
    )

    assert out.shape == (1600,)


def test_match_preserves_vocoder_amplitude_by_default():
    model = _fake_knn_vc(_FakeHiFiGAN(amplitude=0.01))

    out = model.match(
        torch.ones(2, 2),
        torch.ones(4, 2),
        topk=2,
    )

    assert out.abs().max().item() == pytest.approx(0.01, rel=0.05)


def test_attenuate_loud_waveform_default_is_noop():
    waveform = torch.ones(1, 1600) * 0.5

    assert torch.equal(attenuate_loud_waveform(waveform, 16_000), waveform)


def test_attenuate_loud_waveform_only_reduces_audio_above_ceiling():
    sample_rate = 16_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    hot = (0.5 * torch.sin(math.tau * 440.0 * time)).unsqueeze(0)
    quiet = (0.001 * torch.sin(math.tau * 440.0 * time)).unsqueeze(0)

    cooled = attenuate_loud_waveform(hot, sample_rate, -30.0)
    unchanged = attenuate_loud_waveform(quiet, sample_rate, -10.0)

    assert torchaudio.functional.loudness(cooled, sample_rate).item() == pytest.approx(
        -30.0, abs=0.1
    )
    assert torch.allclose(unchanged, quiet)


def test_match_device_uses_device_resolver():
    if torch.cuda.is_available():
        pytest.skip("CUDA fallback test only applies when CUDA is unavailable")

    model = _fake_knn_vc()

    out = model.match(
        torch.ones(2, 2),
        torch.ones(4, 2),
        device="cuda",
        tgt_loudness_db=None,
    )

    assert out.device.type == "cpu"
