from pathlib import Path
import re
import h5py
import matplotlib.pyplot as plt

root = Path('/data/safe_FL/system/results')
rows = []
for path in sorted(root.glob('Cifar10_resnet_Adap-CTA_C10_resnet_Adap-CTA_sweep_l*_final_0.h5')):
    match = re.search(r'_sweep_l(\d+)_final_0\.h5$', path.name)
    if not match:
        continue
    with h5py.File(path, 'r') as handle:
        accuracy = float(handle['rs_test_acc'][:].max())
        privacy = float(handle['round_privacy_cost'][:].sum())
        total = float(handle['round_total_cost'][:].sum())
    rows.append((int(match.group(1)), accuracy, privacy, total))
if not rows:
    raise SystemExit('no sweep result files found')
rows.sort()
out = Path('/data/safe_FL/queue_logs/final_queue7')
out.mkdir(parents=True, exist_ok=True)
(out / 'pareto.csv').write_text('decision_l,accuracy,privacy_cost,total_cost\n' + ''.join(
    f'{level},{accuracy:.8f},{privacy:.8f},{total:.8f}\n'
    for level, accuracy, privacy, total in rows
))
for xlabel, column, filename in (
    ('privacy cost', 2, 'accuracy_privacy_pareto.png'),
    ('total cost', 3, 'accuracy_total_pareto.png'),
):
    plt.figure(figsize=(6, 4))
    xs = [row[column] for row in rows]
    ys = [row[1] for row in rows]
    plt.plot(xs, ys, 'o-')
    for row in rows:
        plt.annotate(str(row[0]), (row[column], row[1]), textcoords='offset points', xytext=(4, 4))
    plt.xlabel(xlabel)
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.savefig(out / filename, dpi=180)
    plt.close()
