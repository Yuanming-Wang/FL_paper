"""Run with python check_hf_gradinv.py; check HF's candidate-batch BN mode."""

import copy

import torch
from torch import nn

from flcore.attacks.hf_gradinv import HFGradInvAttack, HFGradInvConfig
from flcore.attacks.batchnorm import configure_attack_batchnorm
from flcore.servers.reconstructor import GradientReconstructor, reconstruction_costs


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(), nn.Linear(4, 2)).to(device).eval()
    images = torch.rand(2, 1, 2, 2, device=device)
    labels = torch.zeros(2, dtype=torch.long, device=device)
    training_model = copy.deepcopy(model).train()
    target = torch.autograd.grad(nn.functional.cross_entropy(training_model(images), labels),
                                 training_model.parameters())
    bundle = dict(labels=labels, image_shape=[1, 2, 2], target_gradient=target,
                  normalization_mean=[0.0], normalization_std=[1.0])
    weights = [parameter.detach().clone() for parameter in model.parameters()]
    result, stats = HFGradInvAttack(HFGradInvConfig(
        iterations=4, stages=2, gradient_dropout=0, batchnorm_mode="batch",
    )).reconstruct_from_bundle(model, bundle, use_ground_truth_labels=True)
    assert torch.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1
    assert all(torch.isfinite(torch.tensor(row["objective"])) for row in stats["loss_history"])
    assert stats["batchnorm_mode"] == "batch"
    assert model[0].running_mean is None and model[0].running_var is None
    assert not model[0].track_running_stats
    assert all(torch.equal(before, after) for before, after in zip(weights, model.parameters()))
    model.eval()
    training_model.train()
    assert torch.allclose(model(images), training_model(images), atol=1e-6)
    running_model = copy.deepcopy(training_model).eval()
    buffers = {name: value.clone() for name, value in running_model.named_buffers()}
    _, running_stats = HFGradInvAttack(HFGradInvConfig(
        iterations=4, stages=2, gradient_dropout=0,
    )).reconstruct_from_bundle(running_model, bundle, use_ground_truth_labels=True)
    assert running_stats["batchnorm_mode"] == "running"
    assert all(torch.equal(value, dict(running_model.named_buffers())[name]) for name, value in buffers.items())
    geiping_model = copy.deepcopy(training_model).eval()
    torch.manual_seed(11)
    expected = torch.randn(2, 1, 2, 2, device=device).clamp(0, 1)
    torch.manual_seed(11)
    reconstructed, geiping_stats = GradientReconstructor(
        "ResNet18", geiping_model, torch.device(device), num_images=2,
        mean_std=(torch.tensor(0., device=device), torch.tensor(1., device=device)),
        config=dict(batchnorm_mode="batch", max_iterations=4, restarts=1, lr_decay=False, lr=0),
    ).reconstruct(target, labels, (1, 2, 2))
    assert torch.isfinite(reconstructed).all()
    assert torch.equal(reconstructed, expected), "Geiping did not honor the configured zero learning rate"
    assert geiping_stats["batchnorm_mode"] == "batch"
    assert torch.allclose(geiping_model(images), training_model(images), atol=1e-6)
    assert all(torch.equal(before, after) for before, after in zip(weights, geiping_model.parameters()))
    schedule_model = copy.deepcopy(model)
    torch.manual_seed(123)
    candidate = torch.randn(2, 1, 2, 2, device=device).sigmoid().requires_grad_()
    gradient = torch.autograd.grad(nn.functional.cross_entropy(schedule_model(candidate), labels),
                                   schedule_model.parameters())
    expected_loss = float(reconstruction_costs([gradient], target, cost_fn='sim'))
    _, schedule_stats = HFGradInvAttack(HFGradInvConfig(
        iterations=1004, stages=4, learning_rate=0, gradient_dropout=0,
        tv_weight=0, seed=123, batchnorm_mode='batch',
    )).reconstruct_from_bundle(schedule_model, bundle, use_ground_truth_labels=True)
    assert abs(schedule_stats['loss_history'][-1]['gradient_loss'] - expected_loss) < 1e-5, \
        'HF layer-weight schedule restarted at a stage boundary'
    torch.manual_seed(123)
    expected_boxed = (torch.randn(2, 1, 2, 2, device=device) * .25 + .5).clamp(0, 1)
    boxed_bundle = dict(bundle, normalization_mean=[.5], normalization_std=[.25])
    boxed, _ = HFGradInvAttack(HFGradInvConfig(
        iterations=4, stages=2, learning_rate=0, gradient_dropout=0, seed=123,
        input_parameterization='normalized', quantized_gradients=True, batchnorm_mode='batch',
    )).reconstruct_from_bundle(copy.deepcopy(model), boxed_bundle, use_ground_truth_labels=True)
    assert torch.equal(boxed, expected_boxed.cpu()), 'HF normalized-input projection changed pixel scale'
    try:
        configure_attack_batchnorm(copy.deepcopy(training_model), "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid BatchNorm policy was accepted")
    print("PASS: shared batch-stat mode matches training forward, preserves weights, and keeps running-stat default intact")


if __name__ == "__main__":
    main()
