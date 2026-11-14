"""Small assert-based smoke check for the adaptive observation protocol."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

from flcore.attacks.adaptive import (
    AdaptiveAttack,
    AdaptiveAttackConfig,
    load_adaptive_observation,
    save_adaptive_observation,
)
from flcore.privacy_cost import privacy_cost
from flcore.servers.saveserver import exp


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    def forward(self, images):
        return self.layers(images)


def main():
    model = TinyClassifier()
    images = torch.rand(2, 1, 2, 2)
    labels = torch.tensor([0, 1])
    pre_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    update = []
    with torch.no_grad():
        for parameter in model.parameters():
            delta = 0.01 * torch.randn_like(parameter)
            parameter.add_(delta)
            update.append(delta)
    path = Path(__file__).with_name(".adaptive_check.pt")
    try:
        save_adaptive_observation(
            path, model,
            [torch.zeros_like(value) for value in update],
            [-value / 0.1 for value in update], images, labels,
            0.1, 1, 1, 0, 0, 1, num_classes=2, model_state_dict=pre_state,
        )
        bundle = load_adaptive_observation(path)
        victim = TinyClassifier()
        victim.load_state_dict(pre_state)
        reconstruction, _ = AdaptiveAttack(
            AdaptiveAttackConfig(iterations=2, learning_rate=0.01)
        ).reconstruct(victim, bundle)
    finally:
        path.unlink(missing_ok=True)
    assert reconstruction.shape == images.shape
    assert privacy_cost(0.0) == 1.0
    assert privacy_cost(1.0) < privacy_cost(0.0)
    server = SimpleNamespace(
        total_layer=2, min_decision=1, global_model=TinyClassifier(),
        args=SimpleNamespace(num_clients=100, batch_size=64), rs_test_acc=[0.2],
    )
    client = SimpleNamespace(num_batches=1, target_acc=0.6)
    total, privacy, energy, model_cost = exp.calculate_cost(server, 1, client, [2], 0)
    # TinyClassifier has 8 weight and 2 bias parameters; one selected tensor gives ratio 2/10.
    expected_ratio = 2 / 10
    assert abs(privacy - privacy_cost(expected_ratio)) < 1e-12
    assert abs(energy - (0.82 / 9.375 + 0.09 + expected_ratio * 0.09)) < 1e-12
    assert abs(model_cost - 0.4) < 1e-12
    assert abs(total - privacy - energy - model_cost) < 1e-12
    print("adaptive protocol check passed")


if __name__ == "__main__":
    main()


