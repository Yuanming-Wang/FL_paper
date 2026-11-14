"""Run the repository's Geiping-style inversion on a final update bundle.

The input bundle is deliberately the server-visible, delta-derived gradient
observation exported by ``attack_make_bundle.py``.  It does not consume the
private client correction state or raw batch gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from flcore.attacks.gi_nas.bundle import (
    build_victim_model,
    evaluate_reconstruction,
    load_gradient_bundle,
    require_gradient_protocol,
)
from flcore.servers.reconstructor import GradientReconstructor


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Geiping-style inversion against a server-visible update bundle"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batchnorm-mode", choices=("running", "batch"), default="running")
    parser.add_argument("--allow-approximate-observation", action="store_true")
    parser.add_argument("--allow-observed-update", action="store_true",
                        help="Run against an actual selected-client upload observation")
    args = parser.parse_args()

    if args.iterations < 1 or args.restarts < 1:
        raise ValueError("--iterations and --restarts must be positive")
    bundle_path = Path(args.bundle)
    bundle = load_gradient_bundle(bundle_path)
    protocol = require_gradient_protocol(
        bundle, allow_approximate=args.allow_approximate_observation,
        allow_observed_update=args.allow_observed_update,
    )
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = build_victim_model(bundle["model_name"], int(bundle["num_classes"])).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    target_gradient = [value.to(device) for value in bundle["target_gradient"]]
    labels = bundle["labels"].to(device=device, dtype=torch.long)
    mean = tuple(float(value) for value in bundle["normalization_mean"])
    std = tuple(float(value) for value in bundle["normalization_std"])
    config = {
        "signed": not args.unsigned,
        "boxed": True,
        "cost_fn": "sim",
        "indices": "def",
        "weights": "equal",
        "lr": args.lr,
        "optim": args.optimizer,
        "restarts": args.restarts,
        "max_iterations": args.iterations,
        "init": "randn",
        "filter": "none",
        "lr_decay": True,
        "scoring_choice": "loss",
        "batchnorm_mode": args.batchnorm_mode,
    }
    attack = GradientReconstructor(
        "ResNet18", model, device, mean_std=(
            torch.tensor(mean, device=device)[:, None, None],
            torch.tensor(std, device=device)[:, None, None],
        ), config=config, num_images=int(labels.shape[0])
    )
    reconstructed_normalized, stats = attack.reconstruct(
        target_gradient, labels, tuple(int(value) for value in bundle["image_shape"])
    )
    mean_tensor = reconstructed_normalized.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = reconstructed_normalized.new_tensor(std).view(1, -1, 1, 1)
    reconstructed = (reconstructed_normalized * std_tensor + mean_tensor).clamp(0, 1).cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(reconstructed, output_dir / "reconstructed_grid.png", nrow=max(1, reconstructed.shape[0]))
    if "images_01" in bundle:
        save_image(bundle["images_01"].detach().cpu().clamp(0, 1), output_dir / "original_grid.png",
                   nrow=max(1, int(bundle["images_01"].shape[0])))
        _, metrics = evaluate_reconstruction(
            bundle["images_01"], reconstructed, bundle["labels"], model, mean, std
        )
    else:
        metrics = None
    payload = {
        "attack": "Geiping-style gradient inversion",
        "implementation": "repository GradientReconstructor",
        "threat_model": protocol,
        "seed": args.seed,
        "config": config,
        "bundle": str(bundle_path.resolve()),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_metadata": bundle.get("metadata", {}),
        "stats": stats,
        "metrics": metrics,
        "resources": {
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
        },
    }
    torch.save(reconstructed, output_dir / "reconstructed.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Geiping-style reconstruction saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
