"""Gradient-bundle utilities shared by the GI-NAS and HF-GradInv attack runners.

This module provides:
- ``build_victim_model``: the project victim classifier used by all runners.
- ``save_gradient_bundle`` / ``load_gradient_bundle``: the on-disk gradient
  observation format (``fl-paper-gradient-observation-bundle``, version 3).
- ``require_gradient_protocol``: the observation-protocol gate that refuses
  approximate or server-invisible bundles unless explicitly allowed.
- ``evaluate_reconstruction``: the shared reconstruction metrics (MSE, PSNR,
  SSIM, recognition rate, classifier confidence) used by every attack report.
- ``label_multiset_score``: multiset label agreement used by HF-GradInv.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from flcore.servers.inversefed.pytorch_ssim_master import pytorch_ssim

__all__ = [
    "build_victim_model",
    "save_gradient_bundle",
    "load_gradient_bundle",
    "require_gradient_protocol",
    "evaluate_reconstruction",
    "label_multiset_score",
]

BUNDLE_FORMAT = "fl-paper-gradient-observation-bundle"
BUNDLE_FORMAT_VERSION = 3

DEFAULT_APPROXIMATION_STATEMENT = (
    "The attacker observes the complete uploaded local model update "
    "from one selected client and uses its known-SGD-scale effective "
    "update as the optimization target; raw gradients and private "
    "control variates are unavailable."
)


def build_victim_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Build the victim classifier used to define the gradient space.

    ``resnet`` and ``resnet18`` map to ``torchvision.models.resnet18``;
    ``resnet34`` maps to ``resnet34``. The model is returned in eval mode.
    """
    import torchvision

    name = str(model_name).strip().lower()
    if name in {"resnet", "resnet18"}:
        builder = torchvision.models.resnet18
    elif name == "resnet34":
        builder = torchvision.models.resnet34
    else:
        raise ValueError(f"unsupported victim model name: {model_name!r}")
    model = builder(num_classes=int(num_classes))
    model.eval()
    return model


def save_gradient_bundle(
    path: Any,
    model: torch.nn.Module,
    model_name: str,
    num_classes: int,
    gradients: Sequence[torch.Tensor],
    labels: Any,
    normalized_images: torch.Tensor,
    metadata: Optional[Mapping[str, Any]] = None,
    normalization_mean: Sequence[float] = (0.5, 0.5, 0.5),
    normalization_std: Sequence[float] = (0.5, 0.5, 0.5),
) -> Path:
    """Save one gradient observation bundle and return its resolved path.

    ``normalized_images`` must already be in the victim model's input domain
    (the project convention is ``[-1, 1]``). ``images_01`` is derived from it.
    """
    output = Path(path)
    if output.parent and not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    labels_tensor = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    images = normalized_images.detach().cpu().float()
    if images.ndim != 4:
        raise ValueError(f"expected normalized_images [N,C,H,W], got {tuple(images.shape)}")
    if images.shape[0] != labels_tensor.shape[0]:
        raise ValueError("image and label counts differ")
    gradient_tensors = [gradient.detach().cpu() for gradient in gradients]
    if not gradient_tensors:
        raise ValueError("at least one target gradient tensor is required")

    images_01 = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    payload: Dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_name": str(model_name),
        "num_classes": int(num_classes),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "target_gradient": gradient_tensors,
        "labels": labels_tensor,
        "normalized_images": images,
        "images_01": images_01,
        "image_shape": [int(v) for v in images.shape[1:]],
        "normalization_mean": [float(v) for v in normalization_mean],
        "normalization_std": [float(v) for v in normalization_std],
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(payload, output)
    return output.resolve()


def load_gradient_bundle(path: Any) -> Dict[str, Any]:
    """Load a gradient observation bundle as a plain dictionary."""
    source = Path(path)
    try:
        value = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        value = torch.load(source, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"gradient bundle must be a dictionary payload: {source}")
    if value.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported gradient bundle format: {value.get('format')!r}")
    return value


def require_gradient_protocol(
    bundle: Mapping[str, Any],
    allow_approximate: bool = False,
    allow_observed_update: bool = False,
) -> Dict[str, Any]:
    """Validate the observation protocol of a gradient bundle.

    Two gates:
    1. ``allow_approximate`` permits bundles flagged as approximate gradients.
    2. ``allow_observed_update`` permits running against an actual selected-client
       upload observation (threat model B).

    Returns the 5-key threat-model record consumed by the run scripts and
    refuses bundles whose declared observation is approximate (or, when the
    caller did not opt in, a server-invisible raw gradient) so that stale
    bundles cannot silently be re-used under the wrong protocol.
    """
    metadata = bundle.get("metadata") or {}
    observation_mode = str(metadata.get("observation_mode", "unknown"))
    threat_model_id = str(metadata.get("threat_model_id", "unknown"))
    is_approximate_gradient = bool(metadata.get("is_approximate_gradient", False))
    is_observed_client_update = bool(
        metadata.get(
            "is_observed_client_update",
            observation_mode == "actual_selected_client_local_model_update",
        )
    )
    approximation_statement = str(
        metadata.get("approximation_statement") or DEFAULT_APPROXIMATION_STATEMENT
    )
    if is_observed_client_update and not allow_observed_update:
        raise RuntimeError(
            "Bundle is an actual selected-client upload observation "
            f"(threat model {threat_model_id!r}). Pass allow_observed_update=True "
            "to run against this observation."
        )
    if is_approximate_gradient and not allow_approximate:
        raise RuntimeError(
            "this bundle declares an approximate gradient observation; "
            "re-capture the bundle or pass --allow-approximate-observation"
        )
    return {
        "threat_model_id": threat_model_id,
        "observation_mode": observation_mode,
        "is_approximate_gradient": is_approximate_gradient,
        "is_observed_client_update": is_observed_client_update,
        "approximation_statement": approximation_statement,
    }


def label_multiset_score(expected: Sequence[int], inferred: Sequence[int]) -> Dict[str, Any]:
    """Multiset agreement between expected and inferred label collections."""
    from collections import Counter

    expected_list = [int(v) for v in expected]
    inferred_list = [int(v) for v in inferred]
    overlap = sum((Counter(expected_list) & Counter(inferred_list)).values())
    return {
        "label_multiset_accuracy": float(overlap / max(len(expected_list), 1)),
        "label_multiset_exact_match": sorted(expected_list) == sorted(inferred_list),
        "inferred_label_count": int(len(inferred_list)),
        "expected_label_count": int(len(expected_list)),
    }


def evaluate_reconstruction(
    reference_images_01: Any,
    reconstructed: Any,
    labels: Any,
    model: torch.nn.Module,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
) -> Tuple[Any, Dict[str, Any]]:
    """Match and score a reconstructed batch against the reference batch.

    Returns ``(aligned_reconstructed, metrics)``. ``aligned_reconstructed`` is
    the reconstruction reordered by the greedy minimum-batch-MSE assignment;
    the 14-key ``metrics`` dict is inserted verbatim into the attack reports.
    """
    reference = torch.as_tensor(reference_images_01).detach().cpu().float()
    rec = torch.as_tensor(reconstructed).detach().cpu().float()
    labels_tensor = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    if reference.shape != rec.shape:
        raise ValueError(
            f"reference {tuple(reference.shape)} and reconstruction {tuple(rec.shape)} shapes differ"
        )
    batch_size = int(reference.shape[0])
    if labels_tensor.shape[0] != batch_size:
        raise ValueError("label count does not match batch size")

    # Greedy minimum-batch-MSE assignment (reference i -> best free j).
    pair_mse = torch.zeros(batch_size, batch_size, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            pair_mse[i, j] = F.mse_loss(rec[j], reference[i])
    remaining = set(range(batch_size))
    order: List[int] = []
    for i in range(batch_size):
        best_value, best_j = min(
            ((float(pair_mse[i, jj]), jj) for jj in remaining),
            key=lambda t: (t[0], t[1]),
        )
        order.append(best_j)
        remaining.remove(best_j)
    aligned = rec[torch.as_tensor(order, dtype=torch.long)]

    # MSE.
    mse_per_image = [float(F.mse_loss(aligned[i], reference[i])) for i in range(batch_size)]
    mse_mean = float(F.mse_loss(aligned, reference))

    # PSNR (naive running sum, matching the reference implementation).
    psnr_db_per_image = [
        float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        for mse in mse_per_image
    ]
    psnr_run = 0.0
    for value in psnr_db_per_image:
        psnr_run += value
    psnr_db_mean = psnr_run / batch_size

    # Classifier agreement on the [-1,1]-normalized reconstruction.
    device = next(model.parameters()).device
    mean_t = torch.tensor(normalization_mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(normalization_std, dtype=torch.float32).view(1, -1, 1, 1)
    normalized = (aligned - mean_t) / std_t
    with torch.no_grad():
        probability = F.softmax(model(normalized.to(device)), dim=1)
    confidence, prediction = probability.max(dim=1)
    target = labels_tensor.to(device)
    recognition_rate = float((prediction == target).float().mean())
    classifier_confidence_mean = float(confidence.mean())

    # SSIM per image.
    ssim_per_image = [
        float(pytorch_ssim.ssim(reference[i : i + 1], aligned[i : i + 1]))
        for i in range(batch_size)
    ]
    ssim_mean = float(sum(ssim_per_image) / len(ssim_per_image))

    # Optional perceptual metrics via piq; keep the schema stable if missing.
    fsim_mean: Optional[float] = None
    lpips_mean: Optional[float] = None
    perceptual_metric_backend: Optional[str] = None
    perceptual_metric_error: Optional[str] = None
    try:
        import piq  # type: ignore

        fsim_mean = float(piq.fsim(aligned, reference, data_range=1.0))
        lpips_mean = float(piq.LPIPS(replace_pooling=True)(aligned, reference))
        perceptual_metric_backend = "piq"
    except Exception as error:  # pragma: no cover - depends on optional deps
        perceptual_metric_error = f"{type(error).__name__}: {error}"

    metrics: Dict[str, Any] = {
        "matching": "minimum batch MSE",
        "reconstruction_assignment": [int(j) for j in order],
        "mse_per_image": mse_per_image,
        "mse_mean": mse_mean,
        "psnr_db_per_image": psnr_db_per_image,
        "psnr_db_mean": psnr_db_mean,
        "recognition_rate": recognition_rate,
        "classifier_confidence_mean": classifier_confidence_mean,
        "ssim_per_image": ssim_per_image,
        "ssim_mean": ssim_mean,
        "fsim_mean": fsim_mean,
        "lpips_mean": lpips_mean,
        "perceptual_metric_backend": perceptual_metric_backend,
        "perceptual_metric_error": perceptual_metric_error,
    }
    return aligned, metrics
