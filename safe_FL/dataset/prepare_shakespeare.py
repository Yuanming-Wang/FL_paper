"""Prepare genuine IID/Dirichlet Shakespeare partitions, retaining LEAF's split."""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from utils.language_utils import ALL_LETTERS
from prepare_cifar_partitions import partition


CHAR_TO_ID = {char: index for index, char in enumerate(ALL_LETTERS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    leaf_root = args.dataset_root / "utils" / "LEAF"
    sources = {split: leaf_root / split / f"all_data_niid_2_keep_0_{split}_9.json"
               for split in ("train", "test")}
    leaf = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in sources.items()}
    users = [u for u in leaf["train"]["users"] if u in leaf["test"]["user_data"]][:args.num_clients]
    if len(users) != args.num_clients:
        raise ValueError("not enough common LEAF roles")
    arrays = {}
    for split in ("train", "test"):
        x = [sample for user in users for sample in leaf[split]["user_data"][user]["x"]]
        y = [label for user in users for label in leaf[split]["user_data"][user]["y"]]
        x = np.asarray([[CHAR_TO_ID[c] for c in sample] for sample in x], dtype=np.uint8)
        y = np.asarray([CHAR_TO_ID[c] for c in y], dtype=np.int64)
        if x.shape != (len(y), 80):
            raise ValueError(f"invalid {split} sequence shape: {x.shape}")
        arrays[split] = (x, y)
    boundary = len(arrays["train"][1])
    labels = np.concatenate([arrays[s][1] for s in ("train", "test")])
    for alpha in (2.0, 0.2, 0.5, 0.8):
        distribution = "IID" if alpha > 1 else f"Non-IID-{alpha}"
        output = args.dataset_root / "Shakespeare" / distribution
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        assignments = partition(labels, args.num_clients, 80, alpha, args.seed)
        fingerprint = hashlib.sha256()
        counts = {}
        for split in ("train", "test"):
            buckets = [ids[ids < boundary].tolist() if split == "train"
                       else (ids[ids >= boundary] - boundary).tolist() for ids in assignments]
            for empty in (i for i, ids in enumerate(buckets) if not ids):
                donor = max(range(len(buckets)), key=lambda i: len(buckets[i]))
                buckets[empty].append(buckets[donor].pop())
            flattened = [i for ids in buckets for i in ids]
            assert len(flattened) == len(set(flattened)) == len(arrays[split][1])
            counts[split] = [len(ids) for ids in buckets]
            (output / split).mkdir(parents=True)
            for client, ids in enumerate(buckets):
                indices = np.asarray(ids, dtype=np.int64)
                fingerprint.update(indices.tobytes())
                x, y = arrays[split]
                np.savez_compressed(output / split / f"{client}.npz", data={"x": x[indices], "y": y[indices]})
        metadata = dict(num_clients=args.num_clients, num_classes=80, seq_len=80, seed=args.seed,
                        partition="iid" if alpha > 1 else "dirichlet_labels", alpha=None if alpha > 1 else alpha,
                        official_train_test_preserved=True, source_roles=users, sample_counts=counts,
                        assignment_sha256=fingerprint.hexdigest(),
                        source_sha256={s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in sources.items()})
        (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"prepared {distribution}: train={sum(counts['train'])} test={sum(counts['test'])}", flush=True)


if __name__ == "__main__":
    main()
