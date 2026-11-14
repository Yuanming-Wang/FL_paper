"""Capture a server-visible client upload for the inversion runners.

The observation deliberately separates the attacker-visible transcript
(``attack``: the uploaded model update and public protocol parameters) from
reference data that is used exclusively to score reconstructions
(``evaluation_only``).  The payload produced here is consumed by
``attack_make_bundle.py``, which converts the upload into a gradient bundle
using the known SGD step scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

OBSERVATION_FORMAT = "fl-paper-upload-observation-bundle"
OBSERVATION_FORMAT_VERSION = 1


def save_upload_observation(
    path: str | Path,
    model: nn.Module,
    model_state_dict: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    num_batches: int,
    local_epochs: int,
    round_index: int,
    client_id: int,
    num_classes: int = 10,
    mechanism: str = "actual_selected_client_upload",
    algorithm: str = "unknown",
) -> Path:
    """Save the upload transcript; reference data stays evaluation-only.

    ``images`` must be the normalized network inputs in the ``[-1, 1]`` domain,
    matching the bundle convention used by the inversion runners.
    """
    if model_state_dict is None:
        raise ValueError("pre-update model state is required")
    if learning_rate <= 0 or num_batches <= 0 or local_epochs <= 0:
        raise ValueError("learning rate and local step counts must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_state = {
        name: value.detach().cpu().clone() for name, value in model_state_dict.items()
    }
    local_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    model_update = [
        local_state[name] - pre_state[name] for name, _ in model.named_parameters()
    ]
    payload = {
        "format": OBSERVATION_FORMAT,
        "format_version": OBSERVATION_FORMAT_VERSION,
        "attack": {
            "mechanism": str(mechanism),
            "algorithm": str(algorithm),
            "model_state_dict": pre_state,
            "model_update": model_update,
            "num_classes": int(num_classes),
            "learning_rate": float(learning_rate),
            "num_batches": int(num_batches),
            "local_epochs": int(local_epochs),
            "round_index": int(round_index),
            "client_id": int(client_id),
        },
        "evaluation_only": {
            "images": images.detach().cpu(),
            "labels": labels.detach().to(dtype=torch.long).cpu(),
        },
    }
    torch.save(payload, path)
    return path


def load_upload_observation(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a client-upload observation."""
    path = Path(path)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if bundle.get("format") != OBSERVATION_FORMAT:
        raise ValueError("not a client-upload observation bundle")
    attack = bundle.get("attack", {})
    required = {
        "model_state_dict",
        "model_update",
        "num_classes",
        "learning_rate",
        "num_batches",
        "local_epochs",
        "round_index",
        "client_id",
    }
    missing = sorted(required.difference(attack))
    if missing:
        raise ValueError(f"upload observation is missing: {', '.join(missing)}")
    return bundle
