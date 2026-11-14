"""HF-GradInv-style staged gradient inversion attack.

The optimization follows the official HF-GradInv implementation
(https://github.com/MiLab-HITSZ/2023YeHFGradInv): a sigmoid-parameterized
image batch is optimized against the observed per-layer gradients through a
staged, layer-wise dynamic cosine-similarity objective
(``DynaLayerRandPickedCosineSimilarity``), with warm start across stages and
optional signed updates. Labels are inferred from the final-classifier
negative gradient mass unless the known-label protocol is requested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..batchnorm import configure_attack_batchnorm

__all__ = ["HFGradInvConfig", "HFGradInvAttack", "HFGradInv"]


@dataclass
class HFGradInvConfig:
    iterations: int = 3000
    stages: int = 4
    learning_rate: float = 0.1
    restarts: int = 1
    tv_weight: float = 1e-4
    gradient_dropout: float = 0.1
    signed_gradients: bool = False
    seed: int = 42
    report_every: int = 0
    batchnorm_mode: str = "running"
    learning_rate_decay: bool = False


class DynaLayerRandPickedCosineSimilarity:
    """Layer-wise dynamic cosine-similarity loss (official HF-GradInv objective).

    The gradient-tensor mask keeps each entry with probability
    ``1 - drop_rate`` (the upstream code hardcodes 0.3); the project adapter
    exposes that threshold as ``drop_rate`` so the run scripts can configure it.
    """

    def __init__(
        self,
        scale: float = 1.0,
        start: int = 0,
        min_start: int = 0,
        broken_tail: int = 0,
        peroid_Add10: int = 1000,
        add10: int = 10,
        min_weight_scale: float = 0,
        max_weight_scale: float = 1,
        drop_rate: float = 0.3,
    ) -> None:
        self.scale = scale
        self.start = start
        self.min_start = min_start
        self.broken_tail = broken_tail
        self.peroid_Add10 = peroid_Add10
        self.add10 = add10
        self.min_weight_scale = min_weight_scale
        self.max_weight_scale = max_weight_scale
        self.drop_rate = drop_rate
        self.iter = 0

    def gradient_based_loss(self, gradient_rec, gradient_data):
        len_layer = len(gradient_rec)
        final = len_layer - self.broken_tail
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        if (self.iter + 1) % self.peroid_Add10 == 0:
            self.start = self.start - self.add10 if (self.start - self.add10) > self.min_start else self.min_start
        self.iter += 1
        if self.iter >= self.peroid_Add10 + 1:
            self.min_weight_scale = 1
            self.max_weight_scale = 1
        for count, (rec, data) in enumerate(zip(gradient_rec, gradient_data)):
            if self.start <= count + 1 <= final:
                mask = torch.rand_like(data) > self.drop_rate
                weight = (count + 1 - self.start) / (final - self.start) * (self.max_weight_scale - self.min_weight_scale) + self.min_weight_scale
                scalar_product = scalar_product + (rec * data * mask).sum() * weight
                rec_norm = rec_norm + (rec * mask).pow(2).sum() * weight
                data_norm = data_norm + (data * mask).pow(2).sum() * weight
        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()
        return objective * self.scale


class HFGradInvAttack:
    def __init__(self, config: Optional[HFGradInvConfig] = None) -> None:
        self.config = config if config is not None else HFGradInvConfig()

    @staticmethod
    def _normalize(images_01, normalization_mean, normalization_std, device, dtype):
        mean = torch.as_tensor(normalization_mean, device=device, dtype=dtype).reshape(1, -1, 1, 1)
        std = torch.as_tensor(normalization_std, device=device, dtype=dtype).reshape(1, -1, 1, 1)
        return (images_01 - mean) / std

    @staticmethod
    def _infer_labels(target_gradient, batch_size: int, seed: int):
        """Final-classifier negative-mass label inference (official heuristic)."""
        g_fc = target_gradient[-2]
        mass = (-g_fc.sum(dim=1)).clamp(min=0.0)
        probability = mass / mass.sum()
        generator = torch.Generator().manual_seed(seed)
        inferred = torch.multinomial(
            probability.detach().cpu(),
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
        return probability, inferred

    def reconstruct_from_bundle(
        self,
        model: nn.Module,
        bundle: Dict[str, Any],
        auxiliary_data: Optional[torch.Tensor] = None,
        use_ground_truth_labels: bool = False,
    ):
        """Optimize one reconstructed batch and return ``(reconstructed, stats)``."""
        config = self.config
        start_time = time.perf_counter()
        parameter = next(model.parameters())
        device, dtype = parameter.device, parameter.dtype
        configure_attack_batchnorm(model, config.batchnorm_mode)

        labels_gt = torch.as_tensor(bundle["labels"]).detach().cpu().long().reshape(-1)
        batch_size = int(labels_gt.shape[0])
        image_shape = [int(v) for v in bundle["image_shape"]]
        normalization_mean = bundle.get("normalization_mean", [0.5, 0.5, 0.5])
        normalization_std = bundle.get("normalization_std", [0.5, 0.5, 0.5])
        target_gradient = [
            torch.as_tensor(g, device=device, dtype=dtype) for g in bundle["target_gradient"]
        ]

        # Label inference is always reported; only the optimization labels
        # depend on the requested protocol.
        probability, inferred_labels = self._infer_labels(
            target_gradient, batch_size, config.seed
        )
        label_inference = {
            "inference_mode": "final_classifier_negative_mass",
            "class_probability": probability.detach().cpu().tolist(),
            "inferred_labels": [int(v) for v in inferred_labels.tolist()],
        }
        if use_ground_truth_labels:
            optimization_labels = labels_gt.to(device)
        else:
            optimization_labels = inferred_labels.to(device)

        # Staged layer subsets: stage s trains layers [(S-1-s)*T//S, T).
        total_layers = len(target_gradient)
        stage_layers = [
            list(range((config.stages - 1 - s) * total_layers // config.stages, total_layers))
            for s in range(config.stages)
        ]
        iters_per_stage = config.iterations // config.stages
        criterion = nn.CrossEntropyLoss()
        loss_history: List[Dict[str, Any]] = []
        gradient_tensors_used = 0

        best_objective: Optional[float] = None
        best_reconstruction: Optional[torch.Tensor] = None
        for restart in range(config.restarts):
            torch.manual_seed(config.seed + restart)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed + restart)
            if auxiliary_data is not None:
                # CV-assisted warm start: seed the sigmoid-parameterized batch
                # with disjoint public images.
                aux = torch.as_tensor(auxiliary_data).detach().cpu().float()
                init = aux[torch.arange(batch_size) % aux.shape[0]]
                raw = nn.Parameter(
                    torch.logit(init.clamp(1e-3, 1.0 - 1e-3))
                    .reshape(batch_size, *image_shape)
                    .to(device=device, dtype=dtype)
                )
            else:
                raw = nn.Parameter(
                    torch.randn(batch_size, *image_shape, device=device, dtype=dtype)
                )
            optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[config.iterations // 2.667,
                    config.iterations // 1.6, config.iterations // 1.142], gamma=.33,
            ) if config.learning_rate_decay else None
            last_objective: Optional[float] = None
            dyna = DynaLayerRandPickedCosineSimilarity(
                scale=1.0, start=0, min_start=0, broken_tail=0,
                peroid_Add10=1000, add10=10,
                min_weight_scale=0, max_weight_scale=1,
                drop_rate=config.gradient_dropout,
            )
            for stage_index in range(config.stages):
                subset = stage_layers[stage_index]
                data_subset = [target_gradient[i] for i in subset]
                for iteration in range(1, iters_per_stage + 1):
                    candidate = torch.sigmoid(raw)
                    model.zero_grad(set_to_none=True)
                    predict = model(
                        self._normalize(candidate, normalization_mean, normalization_std, device, dtype)
                    )
                    task_loss = criterion(predict, optimization_labels)
                    gradient = list(
                        torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
                    )
                    rec_subset = [gradient[i] for i in subset]
                    gradient_tensors_used += len(subset)
                    gradient_loss = dyna.gradient_based_loss(rec_subset, data_subset)
                    tv = (
                        (candidate[:, :, :, 1:] - candidate[:, :, :, :-1]).abs().mean()
                        + (candidate[:, :, 1:, :] - candidate[:, :, :-1, :]).abs().mean()
                    )
                    objective = gradient_loss + config.tv_weight * tv
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    if config.signed_gradients:
                        raw.grad.sign_()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    last_objective = float(objective.detach().cpu().item())
                    if (
                        iteration == 1
                        or iteration == iters_per_stage
                        or (config.report_every > 0 and iteration % config.report_every == 0)
                    ):
                        loss_history.append(
                            {
                                "stage": float(stage_index + 1),
                                "iteration": float(iteration),
                                "gradient_loss": float(gradient_loss.detach().cpu().item()),
                                "objective": last_objective,
                            }
                        )
                        if config.report_every > 0:
                            print(f"HF restart {restart + 1}/{config.restarts} stage {stage_index + 1}/{config.stages} "
                                  f"iteration {iteration}/{iters_per_stage} objective={last_objective:.6f}", flush=True)
            if best_objective is None or last_objective < best_objective:
                best_objective = last_objective
                best_reconstruction = torch.sigmoid(raw).detach().cpu()

        stats: Dict[str, Any] = {
            "method": "HF-GradInv",
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "batch_size": int(batch_size),
            "restarts": int(config.restarts),
            "best_objective": float(best_objective),
            "auxiliary_data_used": bool(auxiliary_data is not None),
            "label_inference": label_inference,
            "loss_history": loss_history,
            "stages": stage_layers,
            "iterations": int(config.iterations),
            "gradient_tensors_used": int(gradient_tensors_used),
            "batchnorm_mode": config.batchnorm_mode,
            "objective_schedule_iterations": int(dyna.iter),
            "final_learning_rate": float(optimizer.param_groups[0]['lr']),
        }
        return best_reconstruction, stats


HFGradInv = HFGradInvAttack
