from pathlib import Path

import torch
import torch.nn.functional as F

def exists(v):
    """Return whether a value is not ``None``."""
    return v is not None


def default(v, d):
    """Return ``v`` when it exists, otherwise return the default ``d``."""
    return v if exists(v) else d


def identity(t):
    """Return the input unchanged."""
    return t


def append_dims(t, dims):
    """Append singleton dimensions to a tensor shape."""
    shape = t.shape
    ones = (1,) * dims
    return t.reshape(*shape, *ones)


def logit_normal_schedule(t, loc=0.0, scale=1.0):
    """Map uniform samples to a logit-normal time schedule."""
    logits = torch.logit(t, eps=1e-5)
    return 1.0 - torch.sigmoid(
        logits * scale + loc
    )  # sticking with 0 -> 1 convention of noise to data


def cosine_sim_loss(x, y):
    """Return mean cosine distance between two representation batches."""
    return 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
    

def choose_device(device_arg: str | None = None) -> str:
    """Choose an accelerator string from CLI input or available hardware."""
    if device_arg:
        return device_arg
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def autocast_context(device: str):
    """Return an autocast context for supported accelerator types."""
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_config_relative_path(path_value: str, config_path: Path) -> Path:
    """Resolve a config path relative to the directory containing that config."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()