"""Project-integrated GI-NAS gradient inversion attack.

This module reconstructs private training images from a batch of gradient
tensors using the upstream GI-NAS search-then-train procedure
(https://github.com/cswbyu/GI-NAS). The generator architecture search space and
the generator/loss helpers are vendored under ``official/`` and reused verbatim.

Semantic conventions specific to this project:

* The victim classifier consumes inputs normalized to ``[-1, 1]``
  (``Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`` on ``[0, 1]`` images). The
  generator emits ``[0, 1]`` images (``need_sigmoid=True``), so every candidate
  image is normalized with the bundle's ``normalization_mean``/``normalization_std``
  before it is fed to the victim model, so that the computed gradients are
  comparable with the target gradient.
* The reconstructed output returned to the caller stays in ``[0, 1]`` so that
  ``torchvision.utils.save_image`` and the evaluation helpers work unchanged.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from ..batchnorm import configure_attack_batchnorm

# The vendored upstream modules fall back to flat top-level imports (e.g.
# ``import model`` in gen_upsample_layer).  Put ``official/`` on sys.path so
# those fallbacks resolve to the sibling vendored packages without editing the
# upstream files.
_OFFICIAL_ROOT = str(Path(__file__).resolve().parent / "official")
if _OFFICIAL_ROOT not in sys.path:
    sys.path.insert(0, _OFFICIAL_ROOT)

from .official.model.cross_skip import skip
from .official.utils.reconstructed import (
    loss_cosine_similarity,
    loss_inverting_gt,
)

__all__ = ["GINASConfig", "GINASAttack"]


@dataclass
class GINASConfig:
    """Hyper-parameters for the GI-NAS attack.

    ``signed`` gradient updates follow the upstream default and are applied
    unconditionally in the training stage (the upstream attacker always signs
    generator gradients).
    """

    search_size: int = 5000
    iterations: int = 30000
    learning_rate: float = 1e-3
    tv_weight: float = 0.0
    seed: int = 42
    report_every: int = 0
    learning_rate_decay: bool = True
    verbose: bool = True
    batchnorm_mode: str = "running"


class GINASAttack:
    """GI-NAS reconstruction attack driven by an observed gradient bundle."""

    def __init__(self, config: Optional[GINASConfig] = None) -> None:
        self.config = config if config is not None else GINASConfig()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(
        images_01: torch.Tensor,
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map ``[0, 1]`` images into the victim model's ``[-1, 1]`` domain."""
        mean = torch.as_tensor(
            normalization_mean, device=device, dtype=dtype
        ).reshape(1, -1, 1, 1)
        std = torch.as_tensor(
            normalization_std, device=device, dtype=dtype
        ).reshape(1, -1, 1, 1)
        return (images_01 - mean) / std

    @staticmethod
    def _search_space_path() -> Path:
        return Path(__file__).resolve().parent / "official" / "model_search_space.txt"

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        model: nn.Module,
        target_gradient: Sequence[torch.Tensor],
        labels: torch.Tensor,
        image_shape: Sequence[int],
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run GI-NAS search + training and return ``(reconstructed, stats)``.

        ``reconstructed`` is a ``[0, 1]`` tensor of shape
        ``(batch, channels, height, width)``. ``stats`` reports the search
        outcome and the recorded training-loss history.
        """
        config = self.config
        start = time.perf_counter()
        configure_attack_batchnorm(model, config.batchnorm_mode)

        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype

        labels = torch.as_tensor(labels, device=device, dtype=torch.long)
        batch_size = int(labels.shape[0])

        image_shape = [int(value) for value in image_shape]
        channels, height, width = image_shape[0], image_shape[1], image_shape[2]

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        # Target gradient moved to the victim device/dtype.
        original_gt = [
            torch.as_tensor(grad, device=device, dtype=dtype) for grad in target_gradient
        ]
        criterion = nn.CrossEntropyLoss()

        # Fixed input noise shared by every candidate generator.
        self.noise = torch.randn(
            batch_size, channels, height, width, device=device, dtype=dtype
        )

        # ------------------------------------------------------------------
        # Stage 1: architecture search over the vendored search space.
        # ------------------------------------------------------------------
        with self._search_space_path().open("r", encoding="utf-8") as handle:
            skip_indexs = handle.readlines()[: config.search_size]

        best_search_metric = 999.0
        best_model_index = -1
        best_skip_index: Optional[np.ndarray] = None
        best_model: Optional[nn.Module] = None
        searched_candidates = 0
        failed_candidates = 0

        for model_index, skip_line in enumerate(skip_indexs):
            if config.verbose and model_index and model_index % 250 == 0:
                print(f"GI-NAS search {model_index}/{len(skip_indexs)}: best cosine loss {best_search_metric:.6f}, failed {failed_candidates}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Each search-space line encodes one 5x5 skip-connectivity matrix
            # as 25 bits (e.g. "1100110000010001010010101").
            skip_index = np.reshape(
                np.array([int(ch) for ch in skip_line.strip() if not ch.isspace()]),
                (5, 5),
            )

            # Layer 1: generator construction may fail on an invalid topology.
            try:
                trial_netG = skip(
                    model_index=(model_index % 300),
                    skip_index=skip_index,
                    num_input_channels=channels,
                    num_output_channels=channels,
                    num_channels_down=[128] * 5,
                    num_channels_up=[128] * 5,
                    num_channels_skip=[4] * 5,
                    upsample_mode="bilinear",
                    downsample_mode="stride",
                    need_sigmoid=True,
                    need_bias=True,
                    pad="constant",
                    act_fun="LeakyReLU",
                ).to(device)
            except Exception:
                searched_candidates += 1
                failed_candidates += 1
                continue

            # Layer 2: forward / backward through the victim may fail.
            try:
                trial_fake = trial_netG(self.noise)
                trial_input = self._normalize(
                    trial_fake, normalization_mean, normalization_std, device, dtype
                )
                trial_dummy_loss = criterion(model(trial_input), labels)
                trial_fake_dy_dx = list(
                    torch.autograd.grad(
                        trial_dummy_loss, model.parameters(), create_graph=True
                    )
                )
                search_metric = float(
                    loss_cosine_similarity(original_gt, trial_fake_dy_dx)
                    .detach()
                    .cpu()
                    .item()
                )
            except Exception:
                searched_candidates += 1
                failed_candidates += 1
                del trial_netG
                continue

            searched_candidates += 1
            if search_metric < best_search_metric:
                best_search_metric = search_metric
                best_model_index = model_index
                best_skip_index = skip_index
                best_model = copy.deepcopy(trial_netG)

            del trial_netG, trial_fake, trial_input, trial_dummy_loss, trial_fake_dy_dx

        if best_model is None:
            raise RuntimeError(
                "GI-NAS architecture search produced no valid generator candidate"
            )

        if config.verbose:
            print(f"GI-NAS searched {searched_candidates} candidates; selected {best_model_index}, cosine loss {best_search_metric:.6f}", flush=True)

        # ------------------------------------------------------------------
        # Stage 2: train the selected generator against the target gradient.
        # ------------------------------------------------------------------
        netG = best_model
        optimizerG = optim.Adam(netG.parameters(), lr=config.learning_rate)
        scheduler = MultiStepLR(
            optimizerG,
            milestones=[
                int(config.iterations // 2.667),
                int(config.iterations // 1.6),
                int(config.iterations // 1.142),
            ],
            gamma=0.1,
        )

        record_every = config.report_every if config.report_every > 0 else 120
        loss_history: List[Dict[str, Any]] = []
        fake: Optional[torch.Tensor] = None

        for iteration in range(1, config.iterations + 1):
            optimizerG.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)

            fake = netG(self.noise)
            fake_output = model(
                self._normalize(
                    fake, normalization_mean, normalization_std, device, dtype
                )
            )
            dummy_loss = criterion(fake_output, labels)
            fake_dy_dx = list(
                torch.autograd.grad(
                    dummy_loss, model.parameters(), create_graph=True
                )
            )
            errG = loss_inverting_gt(original_gt, fake_dy_dx, fake, config.tv_weight)
            errG.backward()

            # Upstream default: signed generator gradients.
            for layer in netG.parameters():
                if layer.grad is not None:
                    layer.grad.sign_()

            optimizerG.step()

            gradient_loss = float(errG.detach().cpu().item())
            if iteration == 1 or iteration % record_every == 0:
                loss_history.append(
                    {"iteration": int(iteration), "gradient_loss": gradient_loss}
                )
                if config.verbose:
                    print(f"GI-NAS iteration {iteration}/{config.iterations}: loss {gradient_loss:.6f}", flush=True)

            if config.learning_rate_decay:
                scheduler.step()

            fake = fake.detach().cpu()

        reconstructed = fake

        stats: Dict[str, Any] = {
            "method": "GI-NAS",
            "best_model_index": int(best_model_index),
            "best_search_metric": float(best_search_metric),
            "searched_candidates": int(searched_candidates),
            "failed_candidates": int(failed_candidates),
            "iterations": int(config.iterations),
            "batchnorm_mode": config.batchnorm_mode,
            "loss_history": loss_history,
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        return reconstructed, stats
