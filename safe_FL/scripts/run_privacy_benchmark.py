"""Run one method's three attacks with the uniform validated settings."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True,
        choices=['FedAvg','DPFedAvg','HEFedAvg','LDP-FL','AdapLDP-FL','SCAFFOLD','FedPVR','Adap-CTA'])
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--geiping-control', type=Path, required=True)
    parser.add_argument('--ginas-control', type=Path, required=True)
    parser.add_argument('--hf-control', type=Path)
    parser.add_argument('--hf-iterations', type=int, default=3000)
    parser.add_argument('--attacks', nargs='+', choices=('geiping', 'gi_nas', 'hf_gradinv'),
                        default=['geiping', 'gi_nas', 'hf_gradinv'])
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    controls = [args.geiping_control, args.ginas_control]
    if 'hf_gradinv' in args.attacks:
        controls.append(args.hf_control or args.ginas_control.with_name('hf_gradinv.json'))
    for control in controls:
        assert json.loads(control.read_text())['passed'], f'control not validated: {control}'
    hf_options, hf_expected = [], {}
    if 'hf_gradinv' in args.attacks:
        validation = json.loads(controls[-1].read_text())
        assert validation['attack'] == 'hf_gradinv'
        config = validation['config']
        assert config['iterations'] == args.hf_iterations and config['stages'] == 4
        hf_expected = {key: config[key] for key in ('iterations', 'stages', 'learning_rate',
            'restarts', 'tv_weight', 'gradient_dropout', 'signed_gradients',
            'learning_rate_decay', 'input_parameterization', 'quantized_gradients')}
        hf_options = ['--iterations', str(config['iterations']), '--stages', '4',
            '--report-every', '250', '--lr', str(config['learning_rate']),
            '--restarts', str(config['restarts']), '--tv-weight', str(config['tv_weight']),
            '--gradient-dropout', str(config['gradient_dropout']),
            '--input-parameterization', config['input_parameterization'],
            '--gradient-update', 'quantized' if config['quantized_gradients'] else 'adam']
        if config['learning_rate_decay']:
            hf_options.append('--lr-decay')
        if config['signed_gradients']:
            hf_options.append('--signed-gradients')
    system = Path(__file__).resolve().parents[1] / 'system'
    if 'hf_gradinv' in args.attacks:
        hf_source = system / 'flcore/attacks/hf_gradinv/attack.py'
        assert hashlib.sha256(hf_source.read_bytes()).hexdigest() == validation['implementation_source_sha256']
    bundle = args.captures / args.method / 'gradient_bundle.pt'
    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    capture = json.loads((bundle.parent / 'capture_provenance.json').read_text())
    assert capture['algorithm'] == args.method and capture['bundle_sha256'] == bundle_hash
    assert capture['identical_reference_pixels_and_labels']
    output = args.output_root / args.method
    output.mkdir(parents=True, exist_ok=True)
    shared = ['--bundle',str(bundle),'--device','cuda','--seed','0',
              '--allow-observed-update','--batchnorm-mode','batch']
    jobs = [
        ('geiping', 'run_geiping_attack.py',
         ['--iterations','4800','--optimizer','adam','--lr','0.1','--unsigned','--restarts','1'],
         dict(max_iterations=4800, optim='adam', signed=False, lr=.1)),
        ('gi_nas', 'run_gi_nas.py',
         ['--search-size','5000','--iterations','30000','--lr','0.001','--report-every','1000'],
         dict(search_size=5000, iterations=30000, learning_rate=.001)),
        ('hf_gradinv', 'run_hf_gradinv.py', hf_options, hf_expected),
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.device
    status = dict(algorithm=args.method, bundle_sha256=bundle_hash, jobs=[])
    def save_status():
        temporary = output / 'status.json.partial'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / 'status.json')
    for name, script, options, expected in jobs:
        if name not in args.attacks:
            continue
        folder = output / name
        folder.mkdir(exist_ok=True)
        report_file = folder / 'metrics.json'
        row = dict(attack=name, started=time.time())
        status['jobs'].append(row)
        if not report_file.exists():
            command = [sys.executable,'-u',str(system/script),*shared,
                       '--output-dir',str(folder),*options]
            row['command'] = command
            with (folder / 'run.log').open('w') as log:
                process = subprocess.Popen(command, cwd=system, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                save_status()
                # ponytail: one C500 context per 32 GiB VM; larger RAM permits
                # overlap. Wait for context teardown before launching another.
                row['exit_code'] = process.wait()
            save_status()
            if row['exit_code']:
                raise RuntimeError(f'{args.method}/{name} failed: {row["exit_code"]}')
        report = json.loads(report_file.read_text())
        observed_hash = report.get('bundle_sha256', report.get('provenance',{}).get('bundle_sha256'))
        assert observed_hash == bundle_hash and report['seed'] == 0
        mode = report.get('batchnorm_mode', report.get('stats',{}).get('batchnorm_mode'))
        assert mode == 'batch'
        for key, value in expected.items():
            assert report['config'][key] == value, (name, key)
        if name == 'hf_gradinv':
            assert report['implementation_source_sha256'] == validation['implementation_source_sha256']
            assert report['objective_schedule_iterations'] == args.hf_iterations
        elif name == 'gi_nas':
            assert report['searched_candidates'] == 5000
            assert report['loss_history'][-1]['iteration'] == 30000
        assert sorted(report['metrics']['reconstruction_assignment']) == list(range(64))
        assert all(math.isfinite(report['metrics'][key]) for key in
                   ('mse_mean','psnr_db_mean','ssim_mean','recognition_rate','classifier_confidence_mean'))
        row.update(finished=time.time(), verified=True,
                   metrics_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest())
        save_status()
        print('PASS:', args.method, name, flush=True)
    status['complete'] = set(args.attacks) == {job[0] for job in jobs}
    save_status()


if __name__ == '__main__':
    main()
