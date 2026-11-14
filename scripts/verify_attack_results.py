"""Replay saved tensors and scoring; require all 24 cases before publication."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'safe_FL/system'))
from flcore.attacks.batchnorm import configure_attack_batchnorm
from flcore.attacks.gi_nas.bundle import build_victim_model, evaluate_reconstruction, load_gradient_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    base = ROOT / 'safe_FL/results/attacks/uniform_20261106'
    common = torch.load(ROOT / 'tmp/attack_benchmark_20261106/common/reference.pt',
                        map_location='cpu', weights_only=False)
    methods = ['FedAvg', 'DPFedAvg', 'HEFedAvg', 'LDP-FL', 'AdapLDP-FL', 'SCAFFOLD', 'FedPVR', 'Adap-CTA']
    attacks = ['geiping', 'gi_nas', 'hf_gradinv']
    checked, pending = [], []
    for method in methods:
        bundle_file = base / 'captures' / method / 'gradient_bundle.pt'
        bundle = load_gradient_bundle(bundle_file) if bundle_file.exists() else None
        if bundle is not None:
            assert bundle['metadata']['algorithm'] == method
            assert tuple(bundle['normalization_mean']) == (.5, .5, .5)
            assert tuple(bundle['normalization_std']) == (.5, .5, .5)
            common['images_01'] = (common['normalized_images'] * .5 + .5).clamp(0, 1)
            for key in ('images_01', 'normalized_images', 'labels'):
                assert torch.equal(bundle[key], common[key]), (method, key)
            bundle_hash = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
            capture = json.loads((bundle_file.parent / 'capture_provenance.json').read_text())
            assert capture['algorithm'] == method and capture['bundle_sha256'] == bundle_hash
            assert tuple(capture[k] for k in ('round_completed', 'batch_size', 'local_epochs',
                'learning_rate', 'num_batches')) == (800, 64, 5, .01, 4)
            input_file = base / 'inputs' / (method + '.pt')
            assert hashlib.sha256(input_file.read_bytes()).hexdigest() == capture['input_sha256']
            source = torch.load(input_file, map_location='cpu', weights_only=False, mmap=True)
            assert source['algorithm'] == method and source['round_completed'] == 800
            assert source['controls_available']
            assert source['model_state_dict'].keys() == bundle['model_state_dict'].keys()
            assert all(torch.equal(value, bundle['model_state_dict'][name])
                       for name, value in source['model_state_dict'].items())
            if method in ('SCAFFOLD', 'FedPVR', 'Adap-CTA'):
                assert source['client']['id'] == 2
                assert len(source['sc']) == len(source['client']['cc']) == 62
            del source
        for attack in attacks:
            folder = base / 'results' / method / attack
            report_file = folder / 'metrics.json'
            if not report_file.exists() or bundle is None:
                pending.append(method + '/' + attack)
                continue
            report = json.loads(report_file.read_text())
            expected_hash = report.get('bundle_sha256', report.get('provenance', {}).get('bundle_sha256'))
            assert expected_hash == bundle_hash and report['seed'] == 0
            mode = report.get('batchnorm_mode', report.get('stats', {}).get('batchnorm_mode'))
            assert mode == 'batch'
            config = report['config']
            if attack == 'geiping':
                assert config['max_iterations'] == 4800 and config['optim'] == 'adam'
                assert config['lr'] == .1 and config['signed'] is False
            elif attack == 'gi_nas':
                assert config['search_size'] == 5000 and config['iterations'] == 30000
                assert config['learning_rate'] == .001
                assert report['label_mode'] == 'known_ground_truth_multiset'
                assert report['searched_candidates'] == 5000
                assert report['loss_history'][-1]['iteration'] == 30000
            else:
                control = json.loads((base / 'controls/hf_gradinv.json').read_text())
                assert control['passed'] and control['attack'] == 'hf_gradinv'
                assert report['implementation_source_sha256'] == control['implementation_source_sha256']
                assert report['objective_schedule_iterations'] == config['iterations']
                for key in ('iterations', 'stages', 'learning_rate', 'restarts', 'tv_weight',
                    'gradient_dropout', 'signed_gradients', 'learning_rate_decay',
                    'input_parameterization', 'quantized_gradients'):
                    assert config[key] == control['config'][key], (method, attack, key)
                assert report['label_mode'] == 'inferred'
            raw = torch.load(folder / 'reconstructed.pt', map_location='cpu', weights_only=False)
            assert raw.shape == (64, 3, 32, 32) and torch.isfinite(raw).all()
            assert float(raw.min()) >= 0 and float(raw.max()) <= 1
            for filename, tensor in [('original_grid.png', common['images_01']),
                                     ('reconstructed_grid.png', raw)]:
                png = np.array(Image.open(folder / filename).convert('RGB'))
                assert png.shape == (36, 2178, 3)
                tiles = np.stack([png[2:34, 2+34*i:34+34*i] for i in range(64)])
                expected = tensor.mul(255).add(.5).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
                assert np.array_equal(tiles, expected), (method, attack, filename)
            model = build_victim_model(bundle['model_name'], bundle['num_classes'])
            model.load_state_dict(bundle['model_state_dict'], strict=True)
            configure_attack_batchnorm(model, 'batch')
            _, replay = evaluate_reconstruction(common['images_01'], raw, common['labels'], model,
                bundle['normalization_mean'], bundle['normalization_std'])
            saved = report['metrics']
            assert replay['reconstruction_assignment'] == saved['reconstruction_assignment']
            errors = {}
            for key in ('mse_mean', 'psnr_db_mean', 'ssim_mean', 'recognition_rate', 'classifier_confidence_mean'):
                errors[key] = abs(replay[key] - saved[key])
                tolerance = .002 if key == 'classifier_confidence_mean' else 1e-5
                assert errors[key] < tolerance, (method, attack, key, errors[key])
            checked.append(dict(method=method, attack=attack, bundle_sha256=bundle_hash,
                report_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest(), replay_errors=errors,
                artifact_sha256={name:hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in ('reconstructed.pt', 'original_grid.png', 'reconstructed_grid.png')}))
            print('PASS: raw tensors, pixels, protocol and scoring:', method, attack, flush=True)
    complete = len(checked) == 24 and not pending
    if not args.allow_incomplete:
        assert complete, pending
        for name in ('geiping', 'gi_nas', 'hf_gradinv'):
            assert json.loads((base / 'controls' / (name + '.json')).read_text())['passed']
    result = dict(complete=complete, checked_cases=len(checked), checked_image_pairs=64*len(checked),
                  pending=pending, cases=checked)
    (base / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=complete, checked_cases=len(checked), pending_count=len(pending))))


if __name__ == '__main__':
    main()
