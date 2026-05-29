dependencies = ["torch", "torchaudio", "numpy"]

from knn_vc import load_knn_vc  # noqa: E402
from knn_vc import load_hifigan_wavlm, load_wavlm_large


def knn_vc(
    pretrained=True,
    progress=True,
    prematched=True,
    device=None,
    hifigan_checkpoint_path=None,
    hifigan_config_path=None,
):
    """Load kNN-VC through the package API."""
    return load_knn_vc(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=device,
        hifigan_checkpoint_path=hifigan_checkpoint_path,
        hifigan_config_path=hifigan_config_path,
    )


def hifigan_wavlm(
    pretrained=True,
    progress=True,
    prematched=True,
    device=None,
    checkpoint_path=None,
    config_path=None,
):
    """Load the WavLM HiFi-GAN vocoder through the package API."""
    return load_hifigan_wavlm(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=device,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def wavlm_large(pretrained=True, progress=True, device=None):
    """Load WavLM large through the package API."""
    return load_wavlm_large(
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
