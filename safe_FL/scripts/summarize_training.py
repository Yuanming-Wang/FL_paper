"""Build and verify a portable index from the recorded training curves."""

import argparse
import csv
import hashlib
import io
from pathlib import Path

import h5py
import numpy as np

from plot_training_results import load_curve


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/training")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify the saved index without rewriting it")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.output or data / "summary.csv"
    rows = []
    for path in sorted((data / "main_table").glob("*/*/*/*/curves.h5"), key=lambda path: path.as_posix()):
        dataset, model, distribution, method, _ = path.relative_to(data / "main_table").parts
        rounds, percent = load_curve(path, "accuracy")
        accuracy = percent / 100
        with h5py.File(path) as handle:
            arrays = {name: np.asarray(handle[name], dtype=float) for name in (
                "rs_test_loss", "round_model_cost", "round_privacy_cost",
                "round_energy_cost", "round_total_cost")}
            if any(values.shape != accuracy.shape or not np.isfinite(values).all()
                   for values in arrays.values()):
                raise ValueError(f"Incomplete or nonfinite training data: {path}")
            if not np.allclose(arrays["round_total_cost"], arrays["round_model_cost"]
                               + arrays["round_privacy_cost"] + arrays["round_energy_cost"]):
                raise ValueError(f"Cost component sum mismatch: {path}")
            target = float(handle.attrs["target_accuracy"])
            reached = np.flatnonzero(accuracy >= target)
            row = dict(dataset=dataset, model=model, distribution=distribution, algorithm=method,
                       seed=int(handle.attrs["seed"]), rounds=len(rounds),
                       learning_rate=float(handle.attrs["local_learning_rate"]),
                       final_accuracy=float(accuracy[-1]), best_accuracy=float(accuracy.max()),
                       best_round=int(rounds[accuracy.argmax()]), target_accuracy=target,
                       convergence_round=int(rounds[reached[0]]) if len(reached) else -1,
                       final_test_loss=float(arrays["rs_test_loss"][-1]))
        row.update({name + "_cost": float(arrays["round_" + name + "_cost"].sum())
                    for name in ("model", "privacy", "energy", "total")})
        for column, source in (("curves_h5", path), ("training_log", path.with_name("training.log"))):
            if not source.is_file():
                raise FileNotFoundError(source)
            row[column] = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.as_posix()
            row[column + "_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows.append(row)
    if not rows:
        parser.error(f"No training curves found in {data / 'main_table'}")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != text.getvalue():
            raise ValueError(f"Training index is missing or stale; rerun this script without --check: {output}")
        print(f"PASS: {len(rows)} training records, paths and SHA-256 values match the saved index")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text.getvalue(), encoding="utf-8", newline="\n")
        print(f"Saved {len(rows)} verified training records: {output}")


if __name__ == "__main__":
    main()
