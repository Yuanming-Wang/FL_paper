"""Package round-800 CIFAR-10/ResNet checkpoints for the eight-method attack benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch


ROOT = Path(__file__).resolve().parents[2]
METHODS = ["FedAvg", "DPFedAvg", "HEFedAvg", "LDP-FL", "AdapLDP-FL", "SCAFFOLD", "FedPVR", "Adap-CTA"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Full server checkpoints named <method>.pt, including client control states")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "results/training")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the packaged inputs")
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if args.client_id < 0 or len(set(args.methods)) != len(args.methods):
        parser.error("Client ID must be nonnegative and methods must be unique")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory to preserve existing inputs: {args.output_dir}")
    # Check all sources before creating any output files.
    for method in args.methods:
        for path in (args.checkpoint_dir / f"{method}.pt",
                     args.training_dir / f"main_table/Cifar10/resnet/d=0.2/{method}/curves.h5"):
            if not path.is_file():
                raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True)
    rows = []
    for method in args.methods:
        source = args.checkpoint_dir / f"{method}.pt"
        state = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        assert state["round_completed"] == 800, source
        model_state = state["global_model"]
        client = state.get("client")
        if client is None:
            client = next((item for item in state.get("clients", []) if item["id"] == args.client_id), None)
        if client is not None:
            assert client["id"] == args.client_id, source
        sc = state.get("sc", [])
        needs_controls = method in ("SCAFFOLD", "FedPVR", "Adap-CTA")
        ready = not needs_controls or (client is not None and len(sc) == len(client.get("cc", [])) == 62)
        curves_path = args.training_dir / f"main_table/Cifar10/resnet/d=0.2/{method}/curves.h5"
        with h5py.File(curves_path) as curves:
            assert len(curves["rs_test_acc"]) == 800 and str(curves.attrs["algorithm"]) == method
            protocol = {key: float(curves.attrs[key]) for key in (
                "noise_multiplier", "batch_size", "local_epochs", "local_learning_rate")}
            assert (protocol["batch_size"], protocol["local_epochs"], protocol["local_learning_rate"]) == (64, 5, .01)
            payload = dict(algorithm=method, model_state_dict=model_state, protocol=protocol,
                           client=client, sc=sc, round_completed=800, controls_available=bool(ready),
                           source_sha256=state.get("source_sha256", str(curves.attrs["source_sha256"])),
                           final_accuracy=float(curves["rs_test_acc"][-1]),
                           model_source=source.name, model_source_sha256=sha256(source))
        path = args.output_dir / f"{method}.pt"
        torch.save(payload, path)
        rows.append(dict(algorithm=method, file=path.name, sha256=sha256(path), bytes=path.stat().st_size,
                         ready=bool(ready), missing=[] if ready else ["round-800 server and client controls"]))
        print(method, "ready" if ready else "controls missing", flush=True)
    manifest = dict(expected_methods=args.methods, inputs=rows, client_id=args.client_id,
                    batch_size=64, local_epochs=5, local_learning_rate=.01)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged {len(rows)} checkpoint inputs: {args.output_dir}")


if __name__ == "__main__":
    main()
