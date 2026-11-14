"""Run the Adap-CTA adaptive reconstruction attack on one captured bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flcore.attacks.adaptive import (
    AdaptiveAttack,
    AdaptiveAttackConfig,
    load_adaptive_observation,
)
from flcore.attacks.gi_nas.bundle import build_victim_model, evaluate_reconstruction
from flcore.servers.inversefed import consts


def _to_cifar_images(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(consts.cifar10_mean).view(1, -1, 1, 1)
    std = images.new_tensor(consts.cifar10_std).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="resnet", help="Model name used by main.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    args = parser.parse_args()

    bundle = load_adaptive_observation(args.bundle)
    attack = bundle["attack"]
    num_classes = int(attack.get("num_classes", 10))
    model = build_victim_model(args.model, num_classes).to(args.device)
    model.load_state_dict(attack["model_state_dict"])
    images, stats = AdaptiveAttack(
        AdaptiveAttackConfig(
            iterations=args.iterations,
            learning_rate=args.lr,
            restarts=args.restarts,
            tv_weight=args.tv_weight,
            seed=args.seed,
        )
    ).reconstruct(model, bundle)

    stats["attack"] = "adaptive"
    stats["implementation"] = "Adap-CTA public-transcript adaptive attack"
    stats["threat_model"] = {
        "threat_model_id": "C_model_update_without_complete_cc",
        "observation_mode": "server_visible_model_update_sc_delta_cc",
        "private_observations_excluded": ["complete_client_cc", "raw_batch_gradient", "raw_client_data"],
    }
    stats["label_mode"] = "inferred_soft_labels"
    stats["seed"] = args.seed
    stats["config"] = {
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "restarts": args.restarts,
        "tv_weight": args.tv_weight,
    }

    output_dir = args.output_dir or args.bundle.parent / (args.bundle.stem + "_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(images.cpu(), output_dir / "reconstructed.pt")
    evaluation = bundle.get("evaluation_only", {})
    if "images" in evaluation and "labels" in evaluation:
        reference_01 = _to_cifar_images(evaluation["images"].float())
        reconstruction_01 = _to_cifar_images(images.detach().cpu().float())
        _, metrics = evaluate_reconstruction(
            reference_01,
            reconstruction_01,
            evaluation["labels"],
            model,
            consts.cifar10_mean,
            consts.cifar10_std,
        )
        stats["metrics"] = metrics
        try:
            from torchvision.utils import save_image

            save_image(reference_01, output_dir / "original_grid.png", nrow=max(1, min(8, reference_01.shape[0])))
        except Exception as error:
            stats["original_grid_export_error"] = str(error)
    try:
        from torchvision.utils import save_image

        low, high = float(images.min()), float(images.max())
        display = ((images - low) / max(high - low, 1e-6)).clamp(0, 1)
        save_image(display, output_dir / "reconstructed_grid.png", nrow=max(1, min(8, images.shape[0])))
    except Exception as error:
        stats["grid_export_error"] = str(error)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
