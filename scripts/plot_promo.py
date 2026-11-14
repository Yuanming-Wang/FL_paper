"""Render a research publicity graphic from recorded results (no new dependencies)."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "results/training/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
baselines = {"FedAvg", "DPFedAvg", "HEFedAvg", "LDP-FL", "AdapLDP-FL", "SCAFFOLD", "FedMut", "FedPVR"}
image_runs = defaultdict(dict)
for row in rows:
    if row["dataset"] in {"Cifar10", "Cifar100"} and row["algorithm"] in baselines | {"Adap-CTA"}:
        assert row["seed"] == "0" and row["rounds"] == "800", row
        key = row["dataset"], row["model"], row["distribution"]
        assert row["algorithm"] not in image_runs[key], (key, row["algorithm"])
        image_runs[key][row["algorithm"]] = row
assert len(image_runs) == 16
for runs in image_runs.values():
    assert set(runs) == baselines | {"Adap-CTA"}
gains = {key: 100 * (float(runs["Adap-CTA"]["final_accuracy"]) -
         max(float(runs[name]["final_accuracy"]) for name in baselines))
         for key, runs in image_runs.items()}
non_iid = [gain for key, gain in gains.items() if key[2] != "IID"]
wins = sum(gain > 0 for gain in non_iid)
runs = image_runs["Cifar10", "resnet", "d=0.2"]
reduction = 100 * (1 - float(runs["Adap-CTA"]["total_cost"]) / float(runs["FedAvg"]["total_cost"]))
proof = json.loads((ROOT / "formal/verification.json").read_text(encoding="utf-8"))
assert proof["passed"] and proof["all_theorems_audited"]
assert (wins, len(non_iid), round(max(gains.values()), 1), round(reduction, 1), proof["audited_theorem_count"]) == (12, 12, 7.8, 36.2, 43)

BG, NAVY, TEAL, MUTED = "#F8FAF8", "#142B3D", "#087F8C", "#61727A"
LIGHT, LINE, GOLD = "#DFE8E8", "#D6E1E2", "#BC9258"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "svg.hashsalt": "AdapCTA-promo", "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 16), ylim=(0, 9), facecolor=BG)
ax.axis("off")
texts = []


def text(x, y, value, size=12, color=NAVY, weight="normal", align="left", **kwargs):
    result = ax.text(x, y, value, ha=align, va="center", fontsize=size, color=color,
                     weight=weight, **kwargs)
    texts.append(result)
    return result


def box(x, y, w, h, fill=BG, edge=LINE, radius=.12, lw=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fill, edgecolor=edge,
                 linewidth=lw, boxstyle=f"round,pad=0,rounding_size={radius}"))


# Editorial identity and headline; space on the right is reserved for the method art.
ax.plot([.65, 1.15], [8.42, 8.42], color=TEAL, lw=3, solid_capstyle="round")
text(1.31, 8.42, "FEDERATED LEARNING  /  RESEARCH SPOTLIGHT", size=9, color=MUTED, weight="bold")
text(15.35, 8.42, "CODE  /  RESULTS  /  LEAN PROOFS", size=9, color=TEAL, weight="bold", align="right")
text(.58, 7.32, "Adap-CTA", size=67, weight="bold")
text(.65, 6.18, "Adaptive correction.", size=25, weight="bold")
text(.65, 5.66, "Stronger federated learning.", size=25, weight="bold")
text(.67, 4.82, "Optimize the long-term trade-off between", size=13, color=MUTED)
text(.67, 4.47, "model accuracy, privacy risk, and energy.", size=13, color=MUTED)
for x, width, caption in [(.67, 1.75, "8 BASELINES"), (2.62, 2.67, "3 INVERSION ATTACKS")]:
    box(x, 3.70, width, .40, edge="#BDCDCE", radius=.2)
    text(x + width / 2, 3.90, caption, size=8.5, color=TEAL, weight="bold", align="center")
text(.67, 3.11, r"$\widetilde{g}_i = g_i + P_{\ell_i}(c_s-c_i)$", size=23)
text(.68, 2.67, "Control correction, tailored to each client.", size=11, color=MUTED)

# Original scientific illustration: a shared state and three different suffix masks.
text(11.83, 7.59, "ADAPTIVE CORRECTION COVERAGE", size=11, color=TEAL, weight="bold", align="center")
for x, caption, accent in [(9.62, "Model loss", NAVY), (11.82, "Privacy", TEAL), (13.92, "Energy", GOLD)]:
    ax.add_patch(Circle((x - .57, 7.04), .038, facecolor=accent, edgecolor="none"))
    text(x, 7.04, caption, size=10.5, color=MUTED, align="center")

# A light halo anchors the global coordinator without a process-box layout.
ax.add_patch(Circle((11.82, 5.97), .78, facecolor="#EAF2F0", edgecolor="none"))
box(11.18, 5.63, 1.28, .83, fill=NAVY, edge=NAVY, radius=.12)
for y in [5.81, 6.04, 6.27]:
    ax.plot([11.37, 12.24], [y, y], color="#D8EBE7", lw=2.0, solid_capstyle="round")
    ax.add_patch(Circle((11.40, y), .029, facecolor="#74C9C1", edgecolor="none"))
text(11.82, 5.02, "Global model + control", size=10.5, weight="bold", align="center")

clients = [(8.92, 3.79, "Client 1", 2, [NAVY, NAVY, NAVY, TEAL, GOLD, GOLD]),
           (11.82, 3.26, "Client i", 4, [TEAL, TEAL, TEAL, TEAL, GOLD, NAVY]),
           (14.67, 3.79, "Client N", 3, [GOLD, GOLD, GOLD, GOLD, NAVY, TEAL])]
for x, y, caption, selected, mixture in clients:
    start = (11.30, 5.58) if x < 10 else (12.34, 5.58) if x > 13 else (11.82, 4.78)
    end = (x, y + .90)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=10,
                 color="#91AFB0", lw=1.25, connectionstyle="arc3,rad=0.10" if x < 10 else "arc3,rad=-0.10" if x > 13 else "arc3,rad=0"))
    text(x, y + .59, caption, size=11, weight="bold", align="center")
    for k in range(6):
        active = k >= 6 - selected
        box(x - .81 + k * .28, y + .01, .23, .36,
            fill=TEAL if active else LIGHT, edge=TEAL if active else "#CEDDDD", radius=.045, lw=.7)
    for k, color in enumerate(mixture):
        ax.add_patch(Circle((x - .29 + k * .115, y - .27), .031,
                     facecolor=color, edgecolor="none"))
text(11.85, 2.67, "Different clients. Different correction coverage.", size=10.5, color=MUTED, align="center")

# Four verified headline results, deliberately separated from the method illustration.
box(.65, .82, 14.7, 1.56, fill=NAVY, edge=NAVY, radius=.17)
metrics = [(.98, f"{wins} / {len(non_iid)}", "Non-IID image settings won"),
           (4.72, f"+{max(gains.values()):.1f} pp", "Peak accuracy gain"),
           (8.43, f"{reduction:.1f}%", "Lower modeled total cost"),
           (12.11, str(proof["audited_theorem_count"]), "Lean-checked theorems")]
for index, (x, value, caption) in enumerate(metrics):
    if index:
        ax.plot([x - .30, x - .30], [1.12, 2.06], color="#3C5360", lw=.8)
    text(x, 1.82, value, size=35, color="#BCE8DF", weight="bold")
    text(x + .025, 1.21, caption, size=10.5, color="#EDF4F2")
text(.67, .49, "Accuracy: eight baselines, seed 0. Cost vs. FedAvg: CIFAR-10 / ResNet-18, d = 0.2, 800 rounds.", size=8, color=MUTED)
text(.67, .21, "REPRODUCIBLE CODE  +  EXPERIMENT ARTIFACTS  +  MACHINE-CHECKED PROOFS", size=7.6, color=MUTED, weight="bold")
text(15.33, .21, "github.com/Yuanming-Wang/FL_paper", size=8.5, color=TEAL, weight="bold", align="right")

# Small executable checks: verify typography stays in frame and the output is valid SVG.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for obj in texts:
    bounds = obj.get_window_extent(renderer)
    assert bounds.x0 >= 0 and bounds.y0 >= 0, obj.get_text()
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, obj.get_text()
assets = ROOT / "assets"
assets.mkdir(exist_ok=True)
fig.savefig(assets / "adapcta_promo.png", dpi=240, facecolor=BG)
fig.savefig(assets / "adapcta_promo.svg", facecolor=BG,
            metadata={"Title": "Adap-CTA research spotlight", "Date": None})
svg = assets / "adapcta_promo.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
ET.parse(svg)
plt.close(fig)
print("Created assets/adapcta_promo.png (3840 x 2160) and .svg; metrics, proof audit, and label bounds checked.")
