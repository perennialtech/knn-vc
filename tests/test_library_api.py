import json

import pytest
import torch

import knn_vc
from knn_vc import (KNeighborsVC, fast_cosine_dist, load_hifigan_config,
                    load_hifigan_wavlm, load_knn_vc, load_wavlm_large)
from knn_vc.hifigan.utils import AttrDict


class _FakeHiFiGAN(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, x.shape[1] * 320, device=x.device)


class _FakeWavLM(torch.nn.Module):
    pass


def _fake_knn_vc() -> KNeighborsVC:
    return KNeighborsVC(
        _FakeWavLM(),
        _FakeHiFiGAN(),
        AttrDict({"sampling_rate": 16000, "hop_size": 320, "hubert_dim": 2}),
        device="cpu",
    )


def test_public_api_exports_core_symbols():
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
