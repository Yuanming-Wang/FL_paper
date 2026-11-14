"""Redraw the saved attack runs; run this file to regenerate and check the figure."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/attacks"
OUTPUT = SOURCE / "figures"
METHODS = ["FedAvg", "DPFedAvg", "HEFedAvg", "LDP-FL", "AdapLDP-FL", "SCAFFOLD", "FedPVR", "Adap-CTA"]
ATTACKS = ["geiping", "gi_nas", "hf_gradinv"]
SAMPLES = [29, 24, 41]  # Fixed before reconstruction quality was inspected.
VERIFICATION = json.loads((SOURCE / "verification.json").read_text())
assert VERIFICATION['complete'] and VERIFICATION['checked_cases'] == 24, "Full artifact replay must pass before publication"
VERIFIED = {(case['method'], case['attack']): case for case in VERIFICATION['cases']}


def read_grid(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    method, attack = path.parent.parent.name, path.parent.name
    expected = VERIFIED[method, attack]['artifact_sha256'][path.name]
    assert digest == expected, path
    pixels = np.asarray(Image.open(path).convert("RGB"))
    assert pixels.shape == (36, 2178, 3), (path, pixels.shape)
    tiles = np.stack([pixels[2:34, 2 + 34 * i:34 + 34 * i] for i in range(64)])
    return tiles, digest


def main():
    baseline_original, _ = read_grid(SOURCE / "reconstructions/FedAvg/geiping/original_grid.png")
    common_originals = baseline_original[SAMPLES]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
    width, height = 7.16, 3.94
    fig = plt.figure(figsize=(width, height), facecolor="white")
    panel_width, panel_gap, label_width = 1.71, 0.09, 0.55
    row_height, row_gap, block_height = 0.355, 0.04, 1.99
    sources = []
    metric_rows = []
    max_mse_difference = 0.0
    for row, method in enumerate(METHODS):
        x = 0.02 + (row % 4) * (panel_width + panel_gap)
        top = height - 0.12 - (row // 4) * block_height
        title = method
        fig.text((x + panel_width / 2) / width, top / height,
                 f"({chr(97 + row)}) {title}", ha="center", va="center", fontsize=8.5,
                 fontweight="bold" if method == "Adap-CTA" else "normal")
        strips = []
        reference_hash = None
        for attack in ATTACKS:
            folder = SOURCE / "reconstructions" / method / attack
            samples = SAMPLES
            original, original_hash = read_grid(folder / "original_grid.png")
            reconstructed, reconstructed_hash = read_grid(folder / "reconstructed_grid.png")
            report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            assert hashlib.sha256((folder / "metrics.json").read_bytes()).hexdigest() == VERIFIED[method, attack]['report_sha256']
            metrics = report["metrics"]
            assignment = metrics["reconstruction_assignment"]
            assert sorted(assignment) == list(range(64)), folder
            assert reference_hash in (None, original_hash), folder
            reference_hash = original_hash
            assert np.array_equal(common_originals, original[samples]), folder
            aligned = reconstructed[assignment]
            # PNG export rounds each channel by <= 0.5/255; check all matched pairs.
            difference = (original.astype(float) - aligned.astype(float)) / 255
            measured = np.mean(difference ** 2, axis=(1, 2, 3))
            rounding_bound = 2 * np.mean(np.abs(difference), axis=(1, 2, 3)) / 255 + 1 / 255 ** 2
            errors = np.abs(measured - np.asarray(metrics["mse_per_image"]))
            assert np.all(errors <= rounding_bound + 1e-7), (folder, errors.max())
            max_mse_difference = max(max_mse_difference, float(errors.max()))
            if not strips:
                strips.append(original[samples])
            strips.append(aligned[samples])
            metric_rows.append({"method": method, "attack": attack,
                                **{key: metrics[key] for key in ("mse_mean", "psnr_db_mean", "ssim_mean",
                                    "recognition_rate", "classifier_confidence_mean")}})
            sources.append({"method": method, "attack": attack,
                            "directory": folder.relative_to(ROOT).as_posix(),
                            "original_sha256": original_hash, "reconstruction_sha256": reconstructed_hash,
                            "reference_indices_zero_based": samples,
                            "reconstruction_indices_zero_based": [assignment[i] for i in samples]})
        for line, tiles in enumerate(strips):
            y = top - 0.21 - row_height - line * (row_height + row_gap)
            fig.text((x + label_width - 0.045) / width, (y + row_height / 2) / height,
                     ["Original", "Geiping", "GI-NAS", "HF-GradInv"][line],
                     ha="right", va="center", fontsize=7)
            strip = np.full((32, 104, 3), 255, dtype=np.uint8)
            for i, tile in enumerate(tiles):
                strip[:, i * 36:i * 36 + 32] = tile
            ax = fig.add_axes([(x + label_width) / width, y / height,
                               (panel_width - label_width) / width, row_height / height])
            ax.imshow(strip, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
    provenance = {"source_root": SOURCE.relative_to(ROOT).as_posix(),
                  "verification_sha256": hashlib.sha256((SOURCE / 'verification.json').read_bytes()).hexdigest(),
                  "note": "All 24 runs use a common upload-probe protocol. Every panel displays the same three references, with reconstructions following each saved greedy batch-MSE assignment.",
                  "display": "Unmodified PNG pixels, nearest-neighbor display; pairs follow saved MSE assignment.",
                  "checked_image_pairs": 64 * len(sources), "max_png_mse_difference": max_mse_difference,
                  "reference_indices_zero_based": SAMPLES,
                  "metric_rows": metric_rows, "sources": sources}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(OUTPUT / f"Attack_reconstructions.{suffix}", dpi=300)
    plt.close(fig)
    (SOURCE / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2))
    with (SOURCE / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"PASS: checked {64 * len(sources)} pairs across {len(sources)} runs; wrote {OUTPUT / 'Attack_reconstructions.pdf'}")
    print("PASS: all eight method panels and all three attacks use identical original pixels and column order:", SAMPLES)


if __name__ == "__main__":
    main()
