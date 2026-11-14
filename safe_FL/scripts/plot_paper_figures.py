"""Recreate manuscript-style figures using only complete training results."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from plot_training_results import COLORS, METHODS, load_curve

PLOT_METHODS = tuple(method for method in METHODS if method != "FedCEO")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DISTRIBUTIONS = ("d=0.2", "d=0.5", "d=0.8", "IID")
DISPLAY = {"Cifar10": "CIFAR-10", "Cifar100": "CIFAR-100", "resnet": "ResNet-18",
           "VGG11": "VGG-11", "lstm": "LSTM", "Transformer": "Transformer"}
MARKERS = ("o", "s", "^", "v", "D", ">", "<", "P", "X", "*")
COST_ORDER = ("Adap-CTA", "FedPVR", "FedMut", "SCAFFOLD", "LDP-FL", "FedAvg",
              "HEFedAvg", "DPFedAvg", "AdapLDP-FL")
COST_COLORS = ("#4472c4", "#ffe091", "#e9362b")  # energy, privacy, model loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_curve(values, window=15):
    """Return a centered display-only moving average; source values stay unchanged."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("display smoothing window must be odd")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def cost_summary(accuracy, costs, target, to_target):
    """Include the first threshold-reaching round; NR has no invented cost."""
    reached = np.flatnonzero(accuracy >= target)
    stop = int(reached[0]) + 1 if len(reached) else None
    end = stop if to_target else len(accuracy)
    return {"target_round": stop, "rounds_summed": end,
            "components": None if end is None else costs[:end].sum(axis=0).tolist()}


def self_check():
    assert len(display_curve(np.arange(20, dtype=float), 5)) == 20
    assert np.allclose(display_curve(np.ones(20), 5), 1.0)
    acc = np.array([0.2, 0.6, 0.4])
    costs = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    assert cost_summary(acc, costs, 0.6, True) == {
        "target_round": 2, "rounds_summed": 2, "components": [5., 7., 9.]}
    assert cost_summary(acc, costs, 0.2, True)["components"] == [1., 2., 3.]
    assert cost_summary(acc, costs, 0.8, True)["components"] is None
    assert cost_summary(acc, costs, 0.8, False)["components"] == [12., 15., 18.]
    print("PASS: threshold crossing, inclusive prefix sum, NR, and full-horizon cost")


def load_results(root):
    results = {}
    for path in sorted((root / "main_table").glob("*/*/*/*/curves.h5")):
        dataset, model, distribution, method, _ = path.relative_to(root / "main_table").parts
        x, percent = load_curve(path, "accuracy")
        with h5py.File(path) as f:
            accuracy = percent / 100
            target = float(f.attrs["target_accuracy"])
            names = ("round_energy_cost", "round_privacy_cost", "round_model_cost")
            costs = np.column_stack([f[name][:] for name in names])
            if costs.shape != (800, 3) or not np.isfinite(costs).all() or (costs < 0).any():
                raise ValueError(f"Invalid cost arrays: {path}")
            if not np.allclose(costs.sum(axis=1), f["round_total_cost"][:]):
                raise ValueError(f"Cost sum mismatch: {path}")
            expected = f["round_selected_count"][:] * np.maximum(0, target - accuracy)
            if f.attrs.get("cost_timing") != "post_aggregation" or not np.allclose(costs[:, 2], expected):
                raise ValueError(f"Costs do not use post-aggregation accuracy: {path}")
            meta = {k: str(f.attrs.get(k, "")) for k in (
                "source_sha256", "data_sha256", "evaluation_protocol", "privacy_protocol",
                "noise_multiplier", "baseline_implementation", "control_protocol")}
        results[(dataset, model, distribution, method)] = {
            "x": x, "accuracy": accuracy, "costs": costs, "target": target,
            "path": path, "source_path": path.relative_to(root).as_posix(),
            "sha256": digest(path), "metadata": meta}
    expected_groups = [(d, m, s) for d in ("Cifar10", "Cifar100")
                       for m in ("resnet", "VGG11") for s in DISTRIBUTIONS]
    expected_groups += [("Shakespeare", m, "d=0.2") for m in ("lstm", "Transformer")]
    expected = {(*group, method) for group in expected_groups for method in METHODS}
    if set(results) != expected:
        raise ValueError(f"Main-table coverage differs: missing={expected - set(results)}")
    return results


def style(i):
    is_primary = PLOT_METHODS[i] == "Adap-CTA"
    return {"color": COLORS[i], "marker": "D" if is_primary else None,
            "markersize": 2.8 if is_primary else 0,
            "markevery": 80 if is_primary else None,
            "linewidth": 1.25 if is_primary else 0.8,
            "linestyle": "-" if PLOT_METHODS[i] == "Adap-CTA" else ("-", "--", "-.", ":")[i % 4]}


def save(fig, output, name, group_keys, results, entries, original_figure):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    sources = sorted({results[(*key, method)]["source_path"]
                      for key in group_keys for method in PLOT_METHODS})
    entries.append({"name": name, "original_figure": original_figure,
                    "pdf": name + ".pdf", "png": name + ".png", "sources": sources})


def learning_figure(keys, shape, title, name, results, output, entries):
    rows, cols = shape
    fig, axes = plt.subplots(rows, cols, figsize=(7.4 if cols == 2 else 5.0,
                                                 2.65 * rows + 1.0 if cols == 2 else 4.1),
                             squeeze=False)
    upper = min(1.0, math.ceil(max(results[(*key, m)]["accuracy"].max()
                                  for key in keys for m in METHODS) * 10 + 0.2) / 10)
    for panel, (ax, key) in enumerate(zip(axes.flat, keys)):
        for i, method in enumerate(PLOT_METHODS):
            data = results[(*key, method)]
            ax.plot(data["x"], display_curve(data["accuracy"]), label=method,
                    **style(i))
        ax.set(xlim=(0, 800), ylim=(0, upper), xlabel="Training Round", ylabel="Global Accuracy")
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.grid(linestyle="--", color="#b0b0b0", linewidth=0.45, alpha=0.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        panel_title = key[2] if len({k[:2] for k in keys}) == 1 else (
            f"{DISPLAY.get(key[0], key[0])} / {DISPLAY.get(key[1], key[1])} / {key[2]}")
        ax.set_title(panel_title, pad=5, fontsize=11)
        ax.tick_params(labelsize=8.5, direction="out", width=0.8, length=3.5)
    handles = [Line2D([], [], label=m, **style(i)) for i, m in enumerate(PLOT_METHODS)]
    fig.legend(handles=handles, loc="upper center", ncol=5 if cols == 2 else 3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.095 if cols == 2 else 0.13, right=0.985,
                        top=0.84 if rows < 3 else 0.91,
                        bottom=0.13 if cols == 1 else (0.10 if rows < 3 else 0.06),
                        hspace=0.48, wspace=0.24)
    save(fig, output, name, keys, results, entries, "Fig. 3")


def cost_figure(results, output, entries, to_target):
    keys = [("Cifar10", "resnet", s) for s in DISTRIBUTIONS]
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.8))
    for panel, (ax, key) in enumerate(zip(axes.flat, keys)):
        cumulative = np.zeros(len(COST_ORDER))
        values = []
        for method in COST_ORDER:
            data = results[(*key, method)]
            summary = cost_summary(data["accuracy"], data["costs"], 0.6, to_target)
            records.append({"distribution": key[2], "algorithm": method, **summary})
            values.append(summary)
        x = np.arange(len(COST_ORDER))
        for component, color in enumerate(COST_COLORS):
            heights = np.array([v["components"][component] / 1000
                                if v["components"] is not None else 0 for v in values])
            ax.bar(x, heights, width=0.63, bottom=cumulative, color=color,
                   edgecolor="white", linewidth=0.25)
            cumulative += heights
        ax.set_ylim(0, max(cumulative.max() * 1.20, 1))
        for i, value in enumerate(values):
            if value["components"] is None:
                ax.text(i, 0.025, "NR", ha="center", va="bottom", rotation=90,
                        transform=ax.get_xaxis_transform(), color="#666666", fontsize=9)
            elif to_target:
                ax.annotate(f"r={value['target_round']}", (i, cumulative[i]),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        if to_target and not cumulative.any():
                ax.text(0.5, 0.58, "No method reached 60% within 800 rounds",
                    transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
        ax.set_title(key[2], pad=5, fontsize=11)
        ax.set_xticks(x, COST_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(r"Total Cost ($10^3$)")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", color="#b0b0b0", linewidth=0.45, alpha=0.55)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.52, f"({chr(97 + panel)}) {key[2]}", transform=ax.transAxes,
                ha="center", fontsize=11)
    title = ("Cost to 60% global accuracy" if to_target else "Cumulative cost over all 800 rounds")
    handles = [Patch(color=c, label=label) for c, label in zip(COST_COLORS,
               ("Energy cost", "Privacy cost", "Model accuracy loss cost"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.01))
    note = ("NR: target not reached within 800 rounds; r: first target-reaching round."
            if to_target else "Fixed 800-round horizon; this is not a time-to-target comparison.")
    fig.text(0.5, 0.015, note, ha="center", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.25, hspace=0.95, wspace=0.23)
    name = "Fig04_cost_to_target" if to_target else "Fig04_supplement_cost_800_rounds"
    save(fig, output, name, keys, results, entries, "Fig. 4")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=workspace / "results/training")
    parser.add_argument("--output-dir", type=Path, default=workspace / "results/training/figures/paper")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_check()
    source = args.data_dir.resolve()
    results = load_results(source)
    print(f"PASS: {len(results)} complete curves and post-aggregation cost arrays")
    if args.check:
        return
    output = args.output_dir.resolve()
    if output == source or output == source / "main_table" or source / "main_table" in output.parents:
        raise ValueError("Output must not replace the training curves")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 11,
                         "axes.titlesize": 13, "legend.fontsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    entries = []
    configs = [
        ("Cifar10", "resnet", "Fig03a_Cifar10_ResNet18"),
        ("Cifar100", "VGG11", "Fig03b_Cifar100_VGG11"),
        ("Cifar10", "VGG11", "Supplement_Cifar10_VGG11"),
        ("Cifar100", "resnet", "Supplement_Cifar100_ResNet18"),
    ]
    for dataset, model, name in configs:
        learning_figure([(dataset, model, s) for s in DISTRIBUTIONS], (2, 2),
                        f"{DISPLAY[model]} / {DISPLAY[dataset]}", name, results, output, entries)
    for model, name in (("lstm", "Fig03c_Shakespeare_LSTM"),
                        ("Transformer", "Fig03d_Shakespeare_Transformer")):
        learning_figure([("Shakespeare", model, "d=0.2")], (1, 1),
                        f"{DISPLAY[model]} / Shakespeare", name, results, output, entries)
    combined = [("Cifar10", "resnet", s) for s in DISTRIBUTIONS]
    combined += [("Cifar100", "VGG11", s) for s in DISTRIBUTIONS]
    combined += [("Shakespeare", m, "d=0.2") for m in ("lstm", "Transformer")]
    learning_figure(combined, (5, 2), "Learning curves of Adap-CTA and eight baselines",
                    "Fig03_learning_curves_combined", results, output, entries)
    target_costs = cost_figure(results, output, entries, True)
    full_costs = cost_figure(results, output, entries, False)
    manifest = {"input": source.relative_to(workspace).as_posix() if source.is_relative_to(workspace) else source.as_posix(),
                "curve_protocol": "raw sources; one seed; 1-800 displayed rounds",
                "display_smoothing": "15-round centered moving average with transparent raw curve",
                "lr": 0.01, "partial_reruns_used": False, "figures": entries,
                "cost_component_order": ["energy", "privacy", "model_accuracy_loss"],
                "costs_to_target": target_costs, "costs_over_800_rounds": full_costs,
                "sources": [{"key": list(key), "path": v["source_path"], "sha256": v["sha256"],
                             "metadata": v["metadata"]} for key, v in results.items()
                             if key[3] in PLOT_METHODS]}
    for value in results.values():
        assert digest(value["path"]) == value["sha256"], "Source result changed while plotting"
    (output / "figure_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    assert len(entries) == 9
    print(f"Saved {len(entries)} figure sets (PNG/PDF): {output}")


if __name__ == "__main__":
    main()
