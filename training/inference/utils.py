from __future__ import annotations

import importlib
import sys
from typing import Any

import torch


def _install_legacy_module_aliases() -> None:
    alias_map = {
        "bert_pytorch": "training.bert_pytorch",
        "bert_pytorch.model": "training.bert_pytorch.model",
        "bert_pytorch.model.bert": "training.bert_pytorch.model.bert",
        "bert_pytorch.model.log_model": "training.bert_pytorch.model.log_model",
        "bert_pytorch.model.language_model": "training.bert_pytorch.model.language_model",
        "bert_pytorch.model.transformer": "training.bert_pytorch.model.transformer",
        "bert_pytorch.model.attention": "training.bert_pytorch.model.attention",
        "bert_pytorch.model.embedding": "training.bert_pytorch.model.embedding",
        "bert_pytorch.model.utils": "training.bert_pytorch.model.utils",
    }
    for legacy_name, new_name in alias_map.items():
        if legacy_name in sys.modules:
            continue
        sys.modules[legacy_name] = importlib.import_module(new_name)


def safe_torch_load(path: str, device: torch.device) -> Any:
    """Load a torch checkpoint safely across PyTorch versions."""
    _install_legacy_module_aliases()
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions or custom loaders without weights_only
        return torch.load(path, map_location=device)
    except Exception:
        # Fall back to full unpickling when weights-only fails
        return torch.load(path, map_location=device, weights_only=False)


def load_logbert_checkpoint(
    path: str,
    device: torch.device,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Load a LogBERT checkpoint robustly.

    Supports:
    - full model objects (torch.save(model, ...))
    - state_dict-only checkpoints
    - dicts with 'state_dict' or 'model_state_dict'
    """
    state = safe_torch_load(path, device)

    if isinstance(state, torch.nn.Module):
        return state.to(device)

    if hasattr(state, "state_dict"):
        state = state.state_dict()

    if isinstance(state, dict) and any(k in state for k in ("state_dict", "model_state_dict")):
        state = state.get("state_dict", state.get("model_state_dict", state))

    model.load_state_dict(state, strict=False)
    return model
