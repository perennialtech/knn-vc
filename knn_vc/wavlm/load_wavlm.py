import logging

import torch
from torchaudio.pipelines import WAVLM_LARGE


def init_wavlm_large(pretrained=True, progress=True, device="cuda"):
    """
    Load the WavLM large checkpoint from torchaudio pipelines.
    This replaces the legacy unilm/fairseq implementation.
    """
    if not torch.cuda.is_available() and str(device) != "cpu":
        logging.getLogger("wavlm").warning(
            f"Overriding device {device} to cpu since no GPU is available."
        )
        device = "cpu"

    if pretrained:
        model = WAVLM_LARGE.get_model()
    else:
        model = WAVLM_LARGE.get_model()

        def reset_parameters(module):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        model.apply(reset_parameters)

    model.extract_from_layer = 6

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    print(
        f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )
    return model
