import torch

import knn_vc
from knn_vc import fast_cosine_dist, load_hifigan_config


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
