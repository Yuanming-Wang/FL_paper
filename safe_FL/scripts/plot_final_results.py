"""Plot complete 800-round results from final_logs; leave source data unchanged."""
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "FedAvg", "DPFedAvg", "HEFedAvg", "LDP-FL", "SCAFFOLD",
    "FedMut", "FedPVR", "FedCEO", "AdapLDP-FL", "Adap-CTA",
)
COLORS = (
    # Okabe-Ito-inspired colors: distinguishable in print and under common
    # color-vision deficiencies; Adap-CTA is highlighted separately below.
    "#0072B2", "#56B4E9", "#E69F00", "#7F7F7F", "#CC79A7",
    "#009E73", "#A6761D", "#F0E442", "#D55E00", "#000000",
)
METRICS = {
    "accuracy": ("rs_test_acc", "Test accuracy (%)", 100),
    "loss": ("rs_test_loss", "Test loss", 1),
    "cost": ("round_total_cost", "Total cost per round", 1),
}
PAPER_YMAX = {
    ("Cifar10", "resnet"): 0.8,
    ("Cifar10", "VGG11"): 0.9,
    ("Cifar100", "resnet"): 0.4,
    ("Cifar100", "VGG11"): 0.5,
    ("Shakespeare", "lstm"): 0.4,
    ("Shakespeare", "Transformer"): 0.5,
}


def load_curve(path, metric):
    key, _, scale = METRICS[metric]
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[key], dtype=float)
        rounds = np.asarray(handle["eval_rounds"])
        if values.shape != (800,) or not np.isfinite(values).all():
            raise ValueError(f"Expected 800 finite values of {key}: {path}")
        if np.array_equal(rounds, np.arange(800)):
            rounds = rounds + 1
        elif not np.array_equal(rounds, np.arange(1, 801)):
            raise ValueError(f"Unexpected round numbering: {path}")
        for name, expected in {
            "local_learning_rate": 0.01, "num_clients": 100,
            "join_ratio": 0.1, "batch_size": 64, "local_epochs": 5, "seed": 0,
        }.items():
            if not np.isclose(float(handle.attrs.get(name, np.nan)), expected):
                raise ValueError(f"Protocol mismatch for {name}: {path}")
        if metric == "accuracy" and ((values < 0).any() or (values > 1).any()):
            raise ValueError(f"Accuracy outside [0, 1]: {path}")
    return rounds, values * scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "final_logs")
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--model", default="resnet")
    parser.add_argument("--distribution", default="d=0.2")
    parser.add_argument("--metric", choices=METRICS, default="accuracy")
    parser.add_argument("--all", action="store_true", help="Plot all 18 configurations")
    parser.add_argument("--check", action="store_true", help="Validate inputs without plotting")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the figure output directory")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    base = data_dir / "main_table"
    groups = defaultdict(dict)
    for path in sorted(base.glob("*/*/*/*/curves.h5")):
        dataset, model, distribution, method, _ = path.relative_to(base).parts
        group = (dataset, model, distribution)
        if not args.all and group != (args.dataset, args.model, args.distribution):
            continue
        groups[group][method] = path
    if not groups:
        parser.error("No matching complete results; check dataset/model/distribution and --data-dir")

    # Validate the entire selection before writing figures, so a missing method
    # cannot silently produce a partial comparison.
    curves = {}
    for group, paths in groups.items():
        if set(paths) != set(METHODS):
            raise ValueError(f"Unexpected methods in {group}: {sorted(paths)}")
        curves[group] = {method: load_curve(paths[method], args.metric) for method in METHODS}
    if args.check:
        print(f"PASS: {len(groups)} configurations, {len(groups) * len(METHODS)} complete curves")
        return

    output = (args.output_dir.resolve() if args.output_dir else data_dir / "figures" / args.metric)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 17,
        "axes.titlesize": 19,
        "legend.fontsize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plot_methods = tuple(method for method in METHODS
                         if not (args.metric == "accuracy" and method == "FedCEO"))
    plot_colors = {method: COLORS[i] for i, method in enumerate(plot_methods)}
    for (dataset, model, distribution), series in curves.items():
        # IEEE two-column width (7.16 in); compact enough for manuscript insertion.
        fig, ax = plt.subplots(figsize=(7.16, 4.6))
        for i, method in enumerate(plot_methods):
            x, y = series[method]
            if args.metric == "accuracy":
                y = y / 100.0
            is_primary = method == "Adap-CTA"
            ax.plot(x, y, label=method, color=plot_colors[method],
                    linestyle="-" if method == "Adap-CTA" else ("-", "--", "-.", ":")[i % 4],
                    linewidth=2.2 if is_primary else 1.1,
                    marker="D" if is_primary else None,
                    markersize=3.4 if is_primary else 0,
                    markevery=80 if is_primary else None,
                    alpha=1 if method == "Adap-CTA" else 0.85)
        display_model = {"resnet": "ResNet-18", "VGG11": "VGG-11",
                         "lstm": "LSTM", "Transformer": "Transformer"}.get(model, model)
        display_dataset = {"Cifar10": "CIFAR-10", "Cifar100": "CIFAR-100"}.get(dataset, dataset)
        ax.set_title(f"{display_model} / {display_dataset}", pad=6)
        ax.set_xlabel("Training round")
        ax.set_ylabel("Global accuracy" if args.metric == "accuracy" else METRICS[args.metric][1])
        ax.set_xlim(0, 800)
        ax.set_xticks(np.arange(0, 801, 100))
        if args.metric == "accuracy":
            ymax = PAPER_YMAX.get((dataset, model), 1.0)
            ax.set_ylim(0, ymax)
            ax.set_yticks(np.arange(0, ymax + 0.001, 0.1))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1f}"))
            ax.text(0.02, 0.96, distribution, transform=ax.transAxes,
                    ha="left", va="top", fontsize=16)
        ax.grid(True, linestyle="--", color="#b0b0b0", linewidth=0.55, alpha=0.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(direction="out", width=1.0, length=4.5, pad=3)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=5,
                   frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.01),
                   handlelength=1.3, columnspacing=0.7, labelspacing=0.35,
                   handletextpad=0.4)
        fig.tight_layout(rect=(0.03, 0.04, 0.99, 0.83))
        filename = f"{dataset}_{model}_{distribution.replace('=', '')}_{args.metric}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{extension}", dpi=300)
        plt.close(fig)
    print(f"Saved {len(curves)} plots (PNG and PDF): {output}")


if __name__ == "__main__":
    main()
