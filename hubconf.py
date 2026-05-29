dependencies = ["torch", "torchaudio", "numpy"]

from knn_vc import (load_hifigan_wavlm, load_knn_vc,  # noqa: E402
                    load_wavlm_large)


def knn_vc(pretrained=True, progress=True, prematched=True, device="cuda"):
    """Load kNN-VC through the package API."""
    return load_knn_vc(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=device,
    )


def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device="cuda"):
    """Load the WavLM HiFi-GAN vocoder through the package API."""
    return load_hifigan_wavlm(
        pretrained=pretrained,
        progress=progress,
        prematched=prematched,
        device=device,
    )


def wavlm_large(pretrained=True, progress=True, device="cuda"):
    """Load WavLM large through the package API."""
    return load_wavlm_large(
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
