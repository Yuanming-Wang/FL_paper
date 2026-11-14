"""Package verified local models for the uniform eight-method attack rerun."""

import hashlib
import json
from pathlib import Path

import h5py
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / 'tmp/attack_benchmark_20261106/inputs'
    output.mkdir(parents=True, exist_ok=True)
    models = root / 'safe_FL/logs/models/Cifar10/resnet'
    results = root / 'safe_FL/logs/results'
    methods = ['FedAvg', 'DPFedAvg', 'HEFedAvg', 'LDP-FL', 'AdapLDP-FL', 'SCAFFOLD', 'FedPVR', 'Adap-CTA']
    rows = []
    for method in methods:
        if method == 'Adap-CTA':
            state_path = root / 'tmp/adapcta_hf_20261106/client2_resume.pt'
            state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
            assert state['round_completed'] == 800 and state['client']['id'] == 2
            model_state = state['global_model']
            protocol = dict(noise_multiplier=0., batch_size=64, local_epochs=5, local_learning_rate=.01)
            payload = dict(algorithm=method, model_state_dict=model_state, client=state['client'],
                           sc=state['sc'], protocol=protocol, round_completed=800,
                           source_sha256=state['source_sha256'], model_source_sha256=sha256(state_path))
        else:
            prefix = '20261022_partitionfix' if method == 'AdapLDP-FL' else '20261022_required'
            stem = f'{prefix}_Cifar10_resnet_d0.2_{method}_seed0_lr0p01'
            model_path = models / f'{method}_{stem}_0_server.pt'
            curves_path = results / f'Cifar10_resnet_{method}_{stem}_0.h5'
            model = torch.load(model_path, map_location='cpu', weights_only=False, mmap=True)
            model_state = model.state_dict()
            with h5py.File(curves_path) as curves:
                assert len(curves['rs_test_acc']) == 800
                assert str(curves.attrs['algorithm']) == method
                protocol = {key: float(curves.attrs[key]) for key in
                            ('noise_multiplier', 'batch_size', 'local_epochs', 'local_learning_rate')}
                source = str(curves.attrs['source_sha256'])
                accuracy = float(curves['rs_test_acc'][-1])
                metadata = str(curves.attrs.get('baseline_metadata', '{}'))
            assert protocol['batch_size'] == 64 and protocol['local_epochs'] == 5
            assert protocol['local_learning_rate'] == .01
            # The stored AdapLDP scalar only multiplies noise; the recorded noise is zero.
            assert method != 'AdapLDP-FL' or protocol['noise_multiplier'] == 0
            payload = dict(algorithm=method, model_state_dict=model_state, protocol=protocol,
                           round_completed=800, source_sha256=source, final_accuracy=accuracy,
                           baseline_metadata=metadata, model_source=str(model_path.relative_to(root)),
                           model_source_sha256=sha256(model_path))
        ready = method not in ('SCAFFOLD', 'FedPVR')
        payload['controls_available'] = ready
        file = output / f'{method}.pt'
        torch.save(payload, file)
        row = dict(algorithm=method, file=file.name, sha256=sha256(file), bytes=file.stat().st_size,
                   ready=ready, missing=[] if ready else ['round-800 server and client-2 controls'])
        rows.append(row)
        print(method, 'ready' if ready else 'controls missing', row['bytes'], flush=True)
    manifest = dict(expected_methods=methods, inputs=rows,
                    common_rng_source='client2_resume.pt', client_id=2,
                    batch_size=64, local_epochs=5, local_learning_rate=.01)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    assert len(rows) == 8 and len({row['algorithm'] for row in rows}) == 8
    print('PASS: eight trained models packaged; missing controls remain explicit', flush=True)


if __name__ == '__main__':
    main()
