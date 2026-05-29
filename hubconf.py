dependencies = ["torch", "torchaudio", "numpy"]

import json
from pathlib import Path

import torch
from torchaudio.models import Wav2Vec2Model

from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNeighborsVC
from wavlm.load_wavlm import init_wavlm_large


def knn_vc(
    pretrained=True, progress=True, prematched=True, device="cuda"
) -> KNeighborsVC:
    """Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data."""
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc


def hifigan_wavlm(
    pretrained=True, progress=True, prematched=True, device="cuda"
) -> HiFiGAN:
    """Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data."""
    cp = Path(__file__).parent.absolute()

    with open(cp / "hifigan" / "config_v1_wavlm.json") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    if pretrained:
        if prematched:
            url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
        else:
            url = (
                "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt"
            )
        state_dict_g = torch.hub.load_state_dict_from_url(
            url, map_location=device, progress=progress
        )
        generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    print(
        f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters."
    )
    return generator, h


def wavlm_large(pretrained=True, progress=True, device="cuda") -> Wav2Vec2Model:
    """Load the WavLM large checkpoint using torchaudio pipelines."""
    return init_wavlm_large(pretrained=pretrained, progress=progress, device=device)
