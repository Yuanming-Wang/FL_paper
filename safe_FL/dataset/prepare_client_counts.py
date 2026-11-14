"""Repartition the existing CIFAR corpus without moving samples across train/test."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from prepare_cifar_partitions import partition

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-clients', type=int, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
arrays = {}
for split in ('train', 'test'):
    rows = [np.load(p, allow_pickle=True)['data'].item() for p in sorted((args.source / split).glob('*.npz'))]
    arrays[split] = (np.concatenate([row['x'] for row in rows]), np.concatenate([row['y'] for row in rows]))
boundary = len(arrays['train'][1])
assignments = partition(np.concatenate([arrays[s][1] for s in ('train', 'test')]), args.num_clients, 10, 0.2, 0)
counts = {}
fingerprint = hashlib.sha256()
for split in ('train', 'test'):
    buckets = [ids[ids < boundary].tolist() if split == 'train' else (ids[ids >= boundary] - boundary).tolist() for ids in assignments]
    for empty in (i for i, ids in enumerate(buckets) if len(ids) < 2):
        while len(buckets[empty]) < 2:
            donor = max(range(len(buckets)), key=lambda i: len(buckets[i]))
            buckets[empty].append(buckets[donor].pop())
    indices = [i for ids in buckets for i in ids]
    assert len(indices) == len(set(indices)) == len(arrays[split][1])
    (args.output / split).mkdir(parents=True)
    counts[split] = [len(ids) for ids in buckets]
    x, y = arrays[split]
    for client, ids in enumerate(buckets):
        fingerprint.update(np.asarray(ids, dtype=np.int64).tobytes())
        np.savez_compressed(args.output / split / f'{client}.npz', data={'x': x[ids], 'y': y[ids]})
(args.output / 'config.json').write_text(json.dumps(dict(num_clients=args.num_clients, alpha=0.2,
    seed=0, partition='dirichlet_labels', train_test_preserved=True, source=str(args.source),
    assignment_sha256=fingerprint.hexdigest(), sample_counts=counts), indent=2))
print('prepared', args.output, flush=True)
