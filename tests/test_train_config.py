from importlib import resources

import pytest
from omegaconf import OmegaConf

from knn_vc import load_hifigan_config
from knn_vc.hifigan.train import override_with_args, resolve_run_directory


def test_override_with_args_parses_values_with_omegaconf_types():
    config = OmegaConf.create(
        {
            "batch_size": 128,
            "fp16": True,
            "mel": {"fmax_for_loss": 8000},
        }
    )

    override_with_args(
        config,
        ["batch_size", "32", "fp16", "false", "mel.fmax_for_loss", "null"],
    )

    assert config.batch_size == 32
    assert config.fp16 is False
    assert config.mel.fmax_for_loss is None


def test_override_with_args_rejects_unknown_keys():
    config = OmegaConf.create({"batch_size": 128})

    with pytest.raises(KeyError, match="missing"):
        override_with_args(config, ["missing", "32"])


def test_resolve_run_directory_requires_run_dir_for_resume():
    config = OmegaConf.create({"checkpoint_dir": "logs"})

    with pytest.raises(RuntimeError, match="--run-dir"):
        resolve_run_directory(config, None, True)


def test_resolve_run_directory_rejects_nonempty_new_run_dir(tmp_path):
    config = OmegaConf.create({"checkpoint_dir": str(tmp_path)})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("seed: 0\n")

    with pytest.raises(FileExistsError, match="not empty"):
        resolve_run_directory(config, str(run_dir), False)


def test_training_and_inference_hifigan_architecture_config_match():
    training_config_path = resources.files("knn_vc.hifigan").joinpath("config.yaml")
    training = OmegaConf.load(str(training_config_path))
    inference = load_hifigan_config()

    for key in (
        "resblock",
        "upsample_rates",
        "upsample_kernel_sizes",
        "upsample_initial_channel",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "hubert_dim",
        "hifi_dim",
        "hop_size",
    ):
        assert training.hifigan[key] == inference[key]

    assert training.sample_rate == inference.sampling_rate
