# -*- coding: utf-8 -*-
"""Figure 1: three-column study-design framework (stage | analysis | output).

Grayscale, vector PDF + 600 dpi PNG.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

W, H = 100.0, 126.0
fig, ax = plt.subplots(figsize=(6.0, 7.6))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

LW = 0.7
DASH = (0, (3.5, 2.0))

LX, LWID = 3.5, 18.5
MX, MWID = 28.0, 42.0
RX, RWID = 76.0, 20.5
ROW_H, GAP, TOP = 11.0, 5.0, 107.0

ax.add_patch(FancyBboxPatch((1.5, 8.0), W - 3.0, 116.0,
                            boxstyle="round,pad=0,rounding_size=1.4",
                            linewidth=1.0, edgecolor="black", facecolor="white"))


def box(x, y, w, h, text, dashed=False, fs=6.8, bold=False, lw=LW):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                linewidth=lw, edgecolor="black", facecolor="white",
                                linestyle=DASH if dashed else "solid"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                linespacing=1.42, color="black")


def harrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                 arrowstyle="simple,head_length=0.42,head_width=0.36,tail_width=0.14",
                                 mutation_scale=11, facecolor="white",
                                 edgecolor="black", linewidth=LW, shrinkA=0, shrinkB=0))


def varrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                 arrowstyle="simple,head_length=0.38,head_width=0.34,tail_width=0.13",
                                 mutation_scale=11, facecolor="white",
                                 edgecolor="black", linewidth=LW, shrinkA=0, shrinkB=0))


ROWS = [
    ("Data",
     ["ShuttleSet + ShuttleSet22\n94 matches · 223 games\n7,585 rallies"],
     "Public\nstroke-level\ndatasets"),
    ("Timestamping",
     ["State\nscore", "Style\n18 freq.", "Micro\nrally-win\nprob."],
     "Leakage-free\nfeatures"),
    ("Q1\nBoundary",
     ["Time to AUC\n0.70–0.80\n= 11.1–16.3 rallies",
      "Within rally\n8.8 of 10.8 shots"],
     "Time-to-\ncriterion"),
    ("Q2\nGap",
     ["Whole-game\nAUC 0.9475", "In play 0.6597\nafter 10 rallies\n(gap 0.29)"],
     "Stance\ncomparison"),
    ("Q3\nFusion",
     ["Seven fusion\nstrategies", "Sequence models\nGRU / LSTM /\nTransformer"],
     "Two-stage +\nend-to-end"),
    ("Evaluation",
     ["One split (5-fold)\nAUC · accuracy\nBrier · ECE · net benefit"],
     "Common\nprotocol"),
]

# header
hy = TOP + 5.0
for x, w, txt in [(LX, LWID, "Stage"), (MX, MWID, "Analysis"), (RX, RWID, "Output")]:
    box(x, hy, w, 6.4, txt, dashed=True, fs=7.8, bold=True)

for i, (stage, items, right) in enumerate(ROWS):
    y = TOP - i * (ROW_H + GAP) - ROW_H
    cy = y + ROW_H / 2

    box(LX, y, LWID, ROW_H, stage, fs=6.9, bold=True)
    box(MX, y, MWID, ROW_H, "", dashed=True)

    n = len(items)
    if n == 1:
        box(MX + 1.8, y + 1.5, MWID - 3.6, ROW_H - 3.0, items[0], fs=6.3)
    elif n == 2:
        iw = (MWID - 3.6 - 2.0) / 2
        for k, t in enumerate(items):
            box(MX + 1.8 + k * (iw + 2.0), y + 1.5, iw, ROW_H - 3.0, t, fs=6.0)
    else:
        iw = (MWID - 3.6 - 2 * 1.6) / 3
        for k, t in enumerate(items):
            box(MX + 1.8 + k * (iw + 1.6), y + 1.5, iw, ROW_H - 3.0, t, fs=5.9)

    box(RX, y, RWID, ROW_H, right, fs=6.6)

    harrow(LX + LWID + 0.8, MX - 0.8, cy)
    harrow(RX - 0.8, MX + MWID + 0.8, cy)

    if i < len(ROWS) - 1:
        varrow(MX + MWID / 2, y - 0.6, y - GAP + 0.6)

fig.savefig(os.path.join(OUT, "Fig1_study_design.png"), dpi=600, facecolor="white", transparent=False)
fig.savefig(os.path.join(OUT, "Fig1_study_design.pdf"), facecolor="white")
plt.close(fig)

# force white-background RGB (journals do not accept an alpha channel)
from PIL import Image
p = os.path.join(OUT, "Fig1_study_design.png")
im = Image.open(p).convert("RGBA")
bg = Image.new("RGB", im.size, (255, 255, 255))
bg.paste(im, mask=im.split()[3])
bg.save(p, dpi=(600, 600), optimize=True)

print("wrote Fig1_study_design (png 600dpi RGB + pdf)")
