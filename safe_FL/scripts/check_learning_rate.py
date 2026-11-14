"""Check lr=0.01 configuration and reject complete results from another rate; no GPU needed."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py

root = Path(__file__).resolve().parents[1]
curves = list((root.parent/'results/training/main_table').glob('*/*/*/*/curves.h5'))
assert curves, 'No recorded training curves found'
for path in curves:
    with h5py.File(path) as handle:
        assert float(handle.attrs['local_learning_rate']) == .01, path
for name in ('main.py', 'main_collect.py'):
    tree = ast.parse((root/'system'/name).read_text(encoding='utf-8-sig'))
    argument = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and any(isinstance(a, ast.Constant) and a.value == '--local_learning_rate' for a in node.args))
    assert next(ast.literal_eval(k.value) for k in argument.keywords if k.arg == 'default') == .01
for path in (root/'scripts').glob('*.sh'):
    assert '-lr 0.001' not in path.read_text(), path

# An otherwise complete result from lr=0.001 must fail the lr=0.01 queue's audit.
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'result.h5'
    for rate, expected_code in ((.001, 1), (.01, 0)):
        with h5py.File(path, 'w') as handle:
            handle.attrs.update(dict(algorithm='FedAvg', dataset='check', model='linear',
                distribution=2., total_layer=2, rounds_requested=2, num_clients=2,
                join_ratio=1., batch_size=2, local_epochs=1, privacy_protocol='check',
                evaluation_protocol='global_model_all_test_examples_after_aggregation',
                local_learning_rate=rate))
            handle['rs_test_acc'] = [.2, .3]
            handle['rs_test_loss'] = [2., 1.]
        check = subprocess.run([sys.executable, str(root/'scripts/audit_h5.py'), str(path),
            '--learning-rate', '0.01'], capture_output=True, text=True)
        assert check.returncode == expected_code, check.stdout + check.stderr
        if expected_code:
            assert 'local learning rate does not match' in check.stdout
print(f'PASS: lr=0.01 defaults, {len(curves)} recorded runs, and wrong-rate result rejection')
