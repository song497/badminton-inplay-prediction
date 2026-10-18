# -*- coding: utf-8 -*-
"""Fig.1 研究框架图（英文、灰度、按 IJPAS 插图宽度 6.9 in 设计）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["savefig.dpi"] = 600
OUT = str(FIGURES)
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(6.9, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.05); ax.axis("off")

EDGE = "#333333"
F = ["#f2f2f2", "#e8e8e8", "#dcdcdc"]


def box(x0, y0, x1, y1, lines, fill, fs=7.4, bold_first=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.012,rounding_size=0.08",
                                linewidth=0.8, edgecolor=EDGE, facecolor=fill, zorder=2))
    n = len(lines)
    for i, t in enumerate(lines):
        y = y1 - (i + 0.60) * (y1 - y0) / (n + 0.20)
        ax.text((x0 + x1) / 2, y, t, ha="center", va="center",
                fontsize=fs + (0.5 if (bold_first and i == 0) else 0),
                fontweight="bold" if (bold_first and i == 0) else "normal",
                color="#111111", zorder=3)


def arrow(x0, y0, x1, y1, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw,
                                 color=EDGE, shrinkA=0, shrinkB=0, zorder=1))


# 数据层
box(1.15, 8.72, 8.85, 9.82,
    ["ShuttleSet + ShuttleSet22 public datasets",
     "94 matches  |  223 games  |  7,585 rallies (after de-duplication)"],
    F[0], fs=7.6, bold_first=True)
arrow(5.0, 8.72, 5.0, 8.48)

# 信息层
box(1.15, 7.05, 8.85, 8.48,
    ["Information classes and their earliest-computable timestamp",
     "State: score difference, momentum over 3 rallies, rallies completed",
     "Style: 18-dimensional stroke-frequency difference up to the time point",
     "Micro: stroke-level model \u2192 rally-win probabilities"],
    F[0], fs=7.0, bold_first=True)

# 三问
CW, GAP, X0 = 2.58, 0.44, 0.80
HY0, HY1, BY0, BY1 = 6.02, 6.92, 4.10, 5.97
HDR = [["Q1: When does the", "outcome become predictable?"],
       ["Q2: Does the post-hoc", "stance overstate it?"],
       ["Q3: Can fusion lift", "the bound?"]]
BODY = [["Seven information arms", "w = 5, 10, \u2026, 30", "Time to reach AUC 0.70\u20130.80"],
        ["In-play vs post-hoc", "Size of the gap", "Source of the gap"],
        ["Seven fusion strategies", "End-to-end GRU/LSTM", "/Transformer"]]
for i in range(3):
    x0 = X0 + i * (CW + GAP); x1 = x0 + CW; xc = (x0 + x1) / 2
    box(x0, HY0, x1, HY1, HDR[i], F[1], fs=7.4, bold_first=True)
    box(x0, BY0, x1, BY1, BODY[i], F[2], fs=6.9)
    arrow(xc, 7.05, xc, HY1 + 0.02, lw=0.75)
    arrow(xc, BY0, xc, 3.02)

# 评估层
box(1.15, 1.68, 8.85, 3.00,
    ["Common evaluation protocol",
     "Five-fold cross-validation grouped by match (fold mean); one common split",
     "AUC  |  accuracy  |  Brier score  |  ECE  |  net benefit"],
    F[0], fs=7.0, bold_first=True)

fig.tight_layout(pad=0.10)
fig.savefig(os.path.join(OUT, "Fig1_framework.png"))
print("Fig1 ok")
