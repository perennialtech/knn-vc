import logging

import torch

LOGGER = logging.getLogger(__name__)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Overriding device %s to cpu since CUDA is unavailable.", device)
        return torch.device("cpu")

    return resolved


def module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device

    for buffer in module.buffers(recurse=True):
        return buffer.device

    return torch.device("cpu")
