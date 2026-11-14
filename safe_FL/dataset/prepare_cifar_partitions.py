"""Prepare deterministic PFLlib-style CIFAR client partitions."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_cifar(raw_root: Path, dataset: str):
    if dataset == "Cifar10":
        source = raw_root / "cifar-10-batches-py"
        batches = [pickle.load((source / f"data_batch_{i}").open("rb"), encoding="bytes") for i in range(1, 6)]
        test = pickle.load((source / "test_batch").open("rb"), encoding="bytes")
        label_key = b"labels"
    else:
        source = raw_root / "cifar-100-python"
        batches = [pickle.load((source / "train").open("rb"), encoding="bytes")]
        test = pickle.load((source / "test").open("rb"), encoding="bytes")
        label_key = b"fine_labels"

    images = np.concatenate([np.asarray(batch[b"data"]) for batch in batches] + [np.asarray(test[b"data"])])
    labels = np.concatenate([np.asarray(batch[label_key]) for batch in batches] + [np.asarray(test[label_key])])
    images = images.reshape(-1, 3, 32, 32).astype(np.float32)
    return images / 127.5 - 1.0, labels


def partition(labels, num_clients, num_classes, alpha, seed):
    rng = np.random.default_rng(seed)
    if alpha > 1:
        indices = rng.permutation(len(labels))
        return np.array_split(indices, num_clients)

    buckets = [[] for _ in range(num_clients)]
    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(indices, cuts)):
            buckets[client].extend(chunk.tolist())
    # A Dirichlet draw can leave clients empty, which the FL loader cannot train.
    for empty in (i for i, bucket in enumerate(buckets) if len(bucket) < 2):
        while len(buckets[empty]) < 2:
            donor = max(range(num_clients), key=lambda i: len(buckets[i]))
            buckets[empty].append(buckets[donor].pop())
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in buckets]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cifar10", "Cifar100"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "Cifar10" else 100
    images, labels = load_cifar(args.raw_root, args.dataset)
    alpha = 2.0 if args.iid else args.alpha
    clients = partition(labels, args.num_clients, num_classes, alpha, args.seed)
    out = args.output_root / ("IID" if args.iid else f"Non-IID-{args.alpha}")
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)

    for client, indices in enumerate(clients):
        rng = np.random.default_rng(args.seed + client)
        indices = indices[rng.permutation(len(indices))]
        cut = max(1, int(len(indices) * 0.75))
        sets = {"train": indices[:cut], "test": indices[cut:]}
        if len(sets["test"]) == 0:
            sets["test"] = sets["train"][-1:]
        for split, selected in sets.items():
            data = {"x": images[selected], "y": labels[selected]}
            np.savez_compressed(out / split / f"{client}.npz", data=data)
    print(f"prepared {args.dataset} {out} clients={len(clients)}")


if __name__ == "__main__":
    main()
