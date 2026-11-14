"""Rebuild the exact local partitions from public CIFAR-10 and pixel hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['dataset'] == 'Cifar10' and len(manifest['clients']) == 100
    CIFAR10.url = 'https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz'
    wanted = {key for splits in manifest['clients'].values() for split in splits.values()
              for key in zip(split['images_sha256'], split['labels'])}
    found = {}
    for train in (True, False):
        data = CIFAR10(root=args.public_root, train=train, download=True)
        for image, label in zip(data.data, data.targets):
            image = image.transpose(2, 0, 1).copy()
            key = hashlib.sha256(image.tobytes()).hexdigest(), int(label)
            if key in wanted:
                found[key] = image
    assert wanted <= found.keys(), 'reference pixels missing from the official public dataset'
    count = 0
    for client, splits in manifest['clients'].items():
        for name, split in splits.items():
            keys = list(zip(split['images_sha256'], split['labels']))
            pixels = np.stack([found[key] for key in keys])
            x = (pixels.astype(np.float32) / np.float32(255) - np.float32(.5)) / np.float32(.5)
            y = np.asarray(split['labels'], dtype=np.int64)
            assert hashlib.sha256(x.tobytes()).hexdigest() == split['x_sha256']
            assert hashlib.sha256(y.tobytes()).hexdigest() == split['y_sha256']
            folder = args.output_root / 'Cifar10/Non-IID-0.2' / name
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f'{client}.npz', data=dict(x=x, y=y))
            count += len(y)
    assert count == manifest['total_samples']
    receipt = dict(clients=100, verified_partitions=200, samples=count,
        tensors_exact=True, public_archive_md5=CIFAR10.tgz_md5,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'verification.json').write_text(json.dumps(receipt, indent=2))
    print('PASS:', receipt, flush=True)


if __name__ == '__main__':
    main()
