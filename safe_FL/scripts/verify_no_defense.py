"""Check exact single-sample inversion before interpreting upload attacks."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=29)
    parser.add_argument('--samples', type=int, nargs='+')
    parser.add_argument('--attack', choices=('geiping', 'gi_nas', 'hf_gradinv'), default='geiping')
    parser.add_argument('--search-size', type=int, default=5000)
    parser.add_argument('--iterations', type=int, default=4800)
    parser.add_argument('--hf-lr-decay', action='store_true')
    parser.add_argument('--hf-input-parameterization', choices=('sigmoid', 'normalized'), default='sigmoid')
    parser.add_argument('--hf-quantized', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
    from flcore.attacks.gi_nas.bundle import build_victim_model, load_gradient_bundle, evaluate_reconstruction
    from flcore.servers.reconstructor import GradientReconstructor, reconstruction_costs

    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = load_gradient_bundle(args.bundle)
    model = build_victim_model(bundle['model_name'], bundle['num_classes']).to(device)
    model.load_state_dict(bundle['model_state_dict'], strict=True)
    samples = args.samples if args.samples is not None else [args.sample]
    assert samples and all(0 <= i < len(bundle['labels']) for i in samples)
    images = bundle['normalized_images'][samples].to(device)
    labels = bundle['labels'][samples].to(device)
    mean = torch.tensor(bundle['normalization_mean'], device=device)[:, None, None]
    std = torch.tensor(bundle['normalization_std'], device=device)[:, None, None]
    gradient = [p.detach() for p in torch.autograd.grad(F.cross_entropy(model(images), labels), model.parameters())]
    assert all(torch.isfinite(p).all() for p in gradient)
    assert abs(float(reconstruction_costs([gradient], gradient, cost_fn='sim'))) < 2e-5
    if len(set(samples)) == 1 and len(samples) > 1:
        singleton = torch.autograd.grad(F.cross_entropy(model(images[:1]), labels[:1]), model.parameters())
        assert abs(float(reconstruction_costs([singleton], gradient, cost_fn='sim'))) < 1e-3
    weights = [p.detach().clone() for p in model.parameters()]
    if args.attack == 'geiping':
        config = dict(optim='adam', lr=.1, signed=False, boxed=True, restarts=1,
                      max_iterations=args.iterations, batchnorm_mode='running', lr_decay=True)
        result, stats = GradientReconstructor('ResNet18', model, device,
                mean_std=(mean, std), config=config, num_images=len(samples)).reconstruct(
                    gradient, labels, tuple(bundle['image_shape']))
        result = (result * std + mean).clamp(0, 1).detach().cpu()
    elif args.attack == 'hf_gradinv':
        from flcore.attacks.hf_gradinv.attack import HFGradInvAttack, HFGradInvConfig
        attack_config = HFGradInvConfig(iterations=args.iterations, seed=0,
            report_every=100, batchnorm_mode='running', learning_rate_decay=args.hf_lr_decay,
            input_parameterization=args.hf_input_parameterization, quantized_gradients=args.hf_quantized)
        config = asdict(attack_config)
        observation = dict(bundle, target_gradient=gradient, labels=labels.detach().cpu(),
            normalized_images=images.detach().cpu(), images_01=bundle['images_01'][samples])
        result, stats = HFGradInvAttack(attack_config).reconstruct_from_bundle(model, observation)
    else:
        import flcore.attacks.gi_nas.attack as gi_module
        from flcore.attacks.gi_nas.attack import GINASAttack, GINASConfig
        if len(samples) == 1:
            upstream_skip = gi_module.skip
            def singleton_skip(*args, **kwargs):
                generator = upstream_skip(*args, **kwargs)
                # Generator-only padding keeps its 1x1 BatchNorm valid; the
                # victim and gradient loss still see exactly one candidate.
                generator.register_forward_pre_hook(
                    lambda _, inputs: (torch.cat((inputs[0], torch.zeros_like(inputs[0]))),))
                generator.register_forward_hook(lambda _, inputs, output: output[:1])
                return generator
            gi_module.skip = singleton_skip
        attack_config = GINASConfig(search_size=args.search_size, iterations=args.iterations,
            seed=0, report_every=100, batchnorm_mode='running')
        config = asdict(attack_config)
        result, stats = GINASAttack(attack_config).reconstruct(model, gradient, labels,
            tuple(bundle['image_shape']), bundle['normalization_mean'], bundle['normalization_std'])
    reference = bundle['images_01'][samples]
    assert result.shape == reference.shape
    assert torch.isfinite(result).all()
    assert all(torch.equal(before, after) for before, after in zip(weights, model.parameters()))
    _, metrics = evaluate_reconstruction(reference, result, labels, model,
        bundle['normalization_mean'], bundle['normalization_std'])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_image(reference, output / 'original_grid.png', nrow=len(samples))
    save_image(result, output / 'reconstructed_grid.png', nrow=len(samples))
    torch.save(result, output / 'reconstructed.pt')
    report = dict(diagnostic_only=True, observation='exact_gradient_no_correction', attack=args.attack,
        samples=samples, batch_size=len(samples), config=config, stats=stats, metrics=metrics,
        logical_private_images=len(set(samples)),
        hardware=dict(torch_version=torch.__version__,
            device=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'),
        control_note=('generator has a fixed zero-noise padding sample; victim loss has exactly one candidate'
                      if args.attack == 'gi_nas' and len(samples) == 1 else
                      'repeated target with separately optimized candidates'
                      if len(set(samples)) == 1 and len(samples) > 1 else 'distinct-image gradient control'),
        bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        passed=max(metrics['mse_per_image']) < .01 and min(metrics['ssim_per_image']) > .6)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('Exact-gradient control:', {key:metrics[key] for key in ('mse_mean','psnr_db_mean','ssim_mean')}, flush=True)
    print('CONTROL_PASSED',report['passed'],flush=True)


if __name__ == '__main__':
    main()
