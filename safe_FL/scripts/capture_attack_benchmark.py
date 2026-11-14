"""Capture uniform probes through each method's existing client training code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--rng-state', type=Path, required=True)
    parser.add_argument('--reference-bundle', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    for name in ("inputs", "rng_state", "reference_bundle", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    system = Path(__file__).resolve().parents[1] / 'system'
    sys.path.insert(0, str(system))
    os.environ['FL_DATA_ROOT'] = str(args.data_root.resolve())
    torch.set_num_threads(2)
    # Load project utilities before the vendored GI-NAS flat imports.
    from argparse import Namespace
    from flcore.clients.clientavg import clientAVG
    from flcore.clients.clientdpavg import clientdpavg
    from flcore.clients.clientheavg import clientHEAVG
    from flcore.clients.client_privacy import clientLDP, clientAdapLDP
    from flcore.clients.clientscaffold import clientSCAFFOLD
    from flcore.clients.clientS_scaffold import clientS_SCAFFOLD
    from flcore.clients.save_client import exp_client
    from flcore.attacks.observation import save_upload_observation
    from flcore.attacks.gi_nas.bundle import build_victim_model, load_gradient_bundle

    classes = {'FedAvg': clientAVG, 'DPFedAvg': clientdpavg, 'HEFedAvg': clientHEAVG,
               'LDP-FL': clientLDP, 'AdapLDP-FL': clientAdapLDP, 'SCAFFOLD': clientSCAFFOLD,
               'FedPVR': clientS_SCAFFOLD, 'Adap-CTA': exp_client}
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    rng = torch.load(args.rng_state, map_location='cpu', weights_only=False)
    reference = torch.load(args.reference_bundle, map_location='cpu', weights_only=False)
    assert reference['format'] in ('fl-paper-reference-batch', 'fl-paper-gradient-observation-bundle')
    records = []
    for entry in manifest['inputs']:
        method = entry['algorithm']
        if args.methods and method not in args.methods:
            continue
        if not entry['ready']:
            records.append(dict(algorithm=method, status='missing_controls'))
            continue
        source = args.inputs / entry['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
        saved = torch.load(source, map_location='cpu', weights_only=False)
        assert saved['algorithm'] == method and saved['controls_available']
        output = args.output_dir / method
        output.mkdir(parents=True, exist_ok=True)
        model = build_victim_model('resnet', 10)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model.to('cuda')
        client_args = Namespace(model=model, model_name='resnet', algorithm=method,
            dataset='Cifar10', NIID=.2, device='cuda', device_id=0, num_classes=10,
            save_folder_name=str(output), total_layer=62, batch_size=64,
            local_learning_rate=.01, local_epochs=5, learning_rate_decay=False,
            learning_rate_decay_gamma=.99, reconstruction_eval=False,
            noise_multiplier=saved['protocol']['noise_multiplier'], resume_state=str(source))
        client = classes[method](client_args, 2, 314, 105, train_slow=False, send_slow=False)
        if method in ('Adap-CTA', 'SCAFFOLD', 'FedPVR'):
            assert len(saved['sc']) == len(saved['client']['cc']) == 62
            client.client_c = [v.to('cuda') for v in saved['client']['cc']]
            if saved['client'].get('decision') is not None:
                client.decision_l = int(saved['client']['decision'])
            client.set_parameters(model, [v.to('cuda') for v in saved['sc']])
        else:
            client.set_parameters(model)
        client.model.to('cuda')
        before = {name: v.detach().cpu().clone() for name, v in client.model.state_dict().items()}
        assert all(torch.equal(before[name], v) for name, v in saved['model_state_dict'].items())
        first = {}
        def record_images(module, inputs):
            if 'images' not in first:
                first['images'] = inputs[0].detach().clone()
        def record_labels(module, inputs):
            if 'labels' not in first:
                first['labels'] = inputs[1].detach().clone()
        hooks = [client.model.register_forward_pre_hook(record_images),
                 client.loss.register_forward_pre_hook(record_labels)]
        np.random.set_state(rng['numpy_rng'])
        random.setstate(rng['python_rng'])
        torch.set_rng_state(rng['torch_rng'])
        torch.cuda.manual_seed_all(0)
        if method == 'FedPVR':
            client.train([6] * 10, 6)
        else:
            client.train()
        for hook in hooks:
            hook.remove()
        assert torch.equal(first['images'].cpu(), reference['normalized_images'])
        assert torch.equal(first['labels'].cpu(), reference['labels'])
        assert len(client.load_train_data()) == 4
        observation = output / 'observation.pt'
        save_upload_observation(observation, client.model, before, first['images'], first['labels'],
            .01, 4, 5, 800, 2, algorithm=method,
            mechanism=f'{method}_actual_selected_client_upload')
        bundle = output / 'gradient_bundle.pt'
        subprocess.run([sys.executable, str(system / 'attack_make_bundle.py'),
                        str(observation), str(bundle)], cwd=system, check=True)
        captured = load_gradient_bundle(bundle)
        assert captured['metadata']['algorithm'] == method
        assert all(torch.isfinite(v).all() for v in captured['target_gradient'])
        receipt = dict(algorithm=method, input_sha256=entry['sha256'],
            bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
            round_completed=800, batch_size=64, local_epochs=5, learning_rate=.01,
            num_batches=4, identical_reference_pixels_and_labels=True,
            client_training_class=type(client).__name__)
        (output / 'capture_provenance.json').write_text(json.dumps(receipt, indent=2))
        records.append(dict(algorithm=method, status='captured', **{k:v for k,v in receipt.items() if k!='algorithm'}))
        (args.output_dir / 'capture_status.json').write_text(json.dumps(records, indent=2))
        print('PASS: actual upload, common original pixels, correct method tag:', method, flush=True)
        del client, model, saved, captured, first, before
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
