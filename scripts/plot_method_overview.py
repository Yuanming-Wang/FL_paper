"""Render the README's vector method overview: python scripts/plot_method_overview.py."""
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, GOLD = "#253D57", "#167D8D", "#B78532"
MUTED, LINE, GRAY = "#627487", "#D8E1E9", "#E3E9EF"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "Adap-CTA",
                     "font.size": 11, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(14.8, 6.3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 14.8), ylim=(0, 6.3))
ax.axis("off")
texts = []


def label(x, y, text, size=11, color=NAVY, weight="normal", align="left", **kwargs):
    obj = ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                  ha=align, va="center", **kwargs)
    texts.append(obj)
    return obj


def box(x, y, w, h, fill="white", edge=LINE, radius=.12, lw=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 facecolor=fill, edgecolor=edge, linewidth=lw))


def arrow(start, end, color=NAVY, style="-", width=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=13, linewidth=width, color=color, linestyle=style,
                 shrinkA=0, shrinkB=0))


label(.3, 5.95, "Adap-CTA", size=23, weight="bold")
label(2.6, 5.94, "Cost-aware correction for federated learning", size=13, color=MUTED)
label(14.5, 5.94, "METHOD OVERVIEW", size=9, color=MUTED, align="right")

panels = [(.3, 3.2, "Global coordination", NAVY),
          (4.0, 4.8, "Adaptive coverage decision", TEAL),
          (9.3, 5.2, "Client training", NAVY)]
for number, (x, w, title, color) in enumerate(panels, 1):
    box(x, 1.8, w, 3.65, fill="#F3FAFA" if number == 2 else "#F8FAFC",
        edge="#ADCED2" if number == 2 else LINE)
    ax.add_patch(Circle((x + .27, 5.12), .13, facecolor=color, edgecolor="none"))
    label(x + .27, 5.12, str(number), size=9, color="white", weight="bold", align="center")
    label(x + .51, 5.12, title, size=13, color=color, weight="bold")

# Server rack and the two objects broadcast to selected clients.
for y in [4.48, 4.25, 4.02]:
    box(1.19, y, 1.4, .18, edge=NAVY, radius=.03)
    ax.add_patch(Circle((1.33, y + .09), .025, facecolor=TEAL, edgecolor="none"))
    ax.plot([2.17, 2.42], [y + .09, y + .09], color=LINE, lw=2)
label(1.9, 3.77, "Central server", size=12, weight="bold", align="center")
for x, symbol, caption in [(.67, r"$w^r$", "Global model"),
                            (1.98, r"$c_s^r$", "Server control")]:
    box(x, 2.89, 1.16, .6)
    label(x + .58, 3.24, symbol, size=19, align="center")
    label(x + .58, 2.71, caption, size=9, color=MUTED, align="center")
label(1.9, 2.31, "Broadcast to selected clients", size=10.5, align="center")
label(1.9, 2.06, "Aggregate returned updates", size=10, color=MUTED, align="center")

# The decision combines the same three costs recorded by the implementation.
for x, title, symbol, accent in [
        (4.26, "Model loss", r"$J_{\mathrm{model}}$", NAVY),
        (5.77, "Privacy", r"$J_{\mathrm{priv}}$", TEAL),
        (7.28, "Energy", r"$J_{\mathrm{energy}}$", GOLD)]:
    box(x, 4.01, 1.25, .78, edge="#D4E5E7")
    label(x + .625, 4.53, symbol, size=15, color=accent, align="center")
    label(x + .625, 4.21, title, size=10, color=MUTED, align="center")
label(6.4, 3.69, "Accuracy / privacy / energy trade-off", size=11, align="center")
arrow((6.4, 3.43), (6.4, 3.14), color=TEAL, width=1.4)
box(4.6, 2.42, 3.6, .64, fill="white", edge="#ADCED2")
label(6.4, 2.80, r"Choose correction coverage $\ell_i^r$", size=13,
      color=TEAL, weight="bold", align="center")
label(6.4, 2.08, "Client-specific selection at each round", size=10,
      color=MUTED, align="center")

# Small data mixtures illustrate heterogeneity; colored suffixes show the actual mask.
label(10.53, 4.70, "Local data", size=9, color=MUTED, align="center")
label(12.72, 4.70, "Parameter blocks", size=9, color=MUTED, align="center")
for y, client, subscript, corrected, mixture in [
        (4.26, "Client 1", "1", 2, [NAVY, NAVY, NAVY, TEAL, GOLD, GOLD]),
        (3.65, "Client i", "i", 4, [TEAL, TEAL, TEAL, GOLD, GOLD, NAVY]),
        (3.04, "Client N", "N", 3, [GOLD, GOLD, GOLD, GOLD, NAVY, TEAL])]:
    label(9.58, y, client, size=10, weight="bold")
    for index, color in enumerate(mixture):
        ax.add_patch(Circle((10.44 + index % 3 * .13, y + .075 - index // 3 * .15),
                     .042, facecolor=color, edgecolor="none"))
    arrow((10.98, y), (11.36, y), color=MUTED, width=1)
    for index in range(6):
        selected = index >= 6 - corrected
        box(11.55 + index * .37, y - .14, .29, .28,
            fill=TEAL if selected else GRAY,
            edge=TEAL if selected else "#D2DCE5", radius=.035, lw=.7)
    label(14.12, y, rf"$\ell_{{{subscript}}}={corrected}$", size=10, color=TEAL, align="right")
label(11.9, 2.45, r"$\widetilde{g}_i = g_i + P_{\ell_i}(c_s-c_i)$", size=17, align="center")
label(11.9, 2.06, "Apply the selected control correction", size=10,
      color=MUTED, align="center")

arrow((3.56, 3.64), (3.94, 3.64))
arrow((8.86, 3.64), (9.24, 3.64), color=TEAL)

# A single return path makes the communication-round loop explicit.
loop = PlotPath([(11.9, 1.78), (11.9, 1.20), (1.9, 1.20), (1.9, 1.76)],
                [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO])
ax.add_patch(FancyArrowPatch(path=loop, arrowstyle="-|>", mutation_scale=13,
             linewidth=1.5, color=NAVY, joinstyle="round"))
label(6.9, 1.20, "Upload model updates + control changes", size=11, align="center",
      bbox=dict(facecolor="white", edgecolor="none", pad=7))
label(6.9, .84, "Aggregate, evaluate, and repeat across communication rounds", size=10,
      color=MUTED, align="center")
for x, color, text in [(3.05, TEAL, "Corrected parameter block"),
                        (7.35, GRAY, "Standard gradient step")]:
    box(x, .31, .2, .19, fill=color, edge=TEAL if color == TEAL else "#D2DCE5", radius=.025)
    label(x + .32, .405, text, size=9, color=MUTED)

# Keep every label within the exported canvas, including when font metrics change.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in texts:
    bounds = text.get_window_extent(renderer)
    assert bounds.x0 >= 0 and bounds.y0 >= 0, text.get_text()
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, text.get_text()
out = ROOT / "assets"
out.mkdir(exist_ok=True)
metadata = {"Title": "Adap-CTA method overview", "Description":
            "Global coordination, cost-aware correction-coverage decisions, and client training.",
            "Date": None}
fig.savefig(out / "adapcta_overview.svg", metadata=metadata, facecolor="white")
fig.savefig(out / "adapcta_overview.png", dpi=180, facecolor="white")
svg = out / "adapcta_overview.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
               encoding="utf-8")
ET.parse(svg)
plt.close(fig)
print("Rendered assets/adapcta_overview.svg and .png; label bounds and SVG XML passed.")
