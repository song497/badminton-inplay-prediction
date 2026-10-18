# -*- coding: utf-8 -*-
"""生成 5 张英文、黑白（灰度）图，对应 IJPAS 英文稿 Fig.1–Fig.5。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402

matplotlib.rcParams["font.family"] = "DejaVu Sans"   # 英文
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["savefig.dpi"] = 600

B = str(RESULTS)
OUTD = os.path.join(B, "en_figs")
os.makedirs(OUTD, exist_ok=True)

TA = json.load(open(os.path.join(B, "timealigned_learner.json"), encoding="utf-8"))
IP = json.load(open(os.path.join(B, "inplay_lr.json"), encoding="utf-8"))
WS = ["5", "10", "15", "20", "25", "30"]
WX = [int(w) for w in WS]

# ---------- Fig 2：七臂 AUC + 与 State 的差值 ----------
ARM = [("State", "state", dict(ls="-", marker="o", color="k")),
       ("Style", "style", dict(ls="--", marker="s", color="k")),
       ("Micro", "micro", dict(ls=":", marker="^", color="k")),
       ("State+Style", "state+style", dict(ls="-.", marker="D", color="0.35")),
       ("State+Micro", "state+micro", dict(ls="-", marker="v", color="0.55")),
       ("Style+Micro", "style+micro", dict(ls="--", marker="P", color="0.75")),
       ("All", "all", dict(ls=":", marker="x", color="0.15"))]
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8))
ax = axs[0]
for lab, key, st in ARM:
    ys = [TA["windows"][w]["auc"][key]["lr"] for w in WS]
    ax.plot(WX, ys, linewidth=1.0, markersize=3.2, label=lab, **st)
ax.set_xlabel("Prediction time point (rallies completed)")
ax.set_ylabel("Test-set AUC")
ax.set_xticks(WX); ax.set_ylim(0.60, 0.95)
ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower right")
ax.set_title("(a)", fontsize=9, loc="left")
ax = axs[1]
base = [TA["windows"][w]["auc"]["state"]["lr"] for w in WS]
for lab, key, st in ARM[1:]:
    ys = [TA["windows"][w]["auc"][key]["lr"] - b for w, b in zip(WS, base)]
    ax.plot(WX, ys, linewidth=1.0, markersize=3.2, label=lab, **st)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("Prediction time point (rallies completed)")
ax.set_ylabel("AUC difference from State arm")
ax.set_xticks(WX)
ax.set_title("(b)", fontsize=9, loc="left")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUTD, "Fig2_arms.png")); plt.close(fig)
print("Fig2 ok")

# ---------- Fig 3：ICC + Style AUC / 同义反复 ----------
SNR = json.load(open(os.path.join(B, "snr_summary.json"), encoding="utf-8"))
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8))
ax = axs[0]
icc = [SNR["by_cutoff"][w]["weighted_icc"] for w in WS]
sty = [TA["windows"][w]["auc"]["style"]["lr"] for w in WS]
ax.plot(WX, icc, "k-o", markersize=3.2, linewidth=1.0, label="Weighted ICC of stroke frequencies")
ax.set_xlabel("Prediction time point (rallies completed)")
ax.set_ylabel("ICC", color="k")
ax.set_xticks(WX)
ax2 = ax.twinx()
ax2.plot(WX, sty, "k--s", markersize=3.2, linewidth=1.0, label="AUC, style arm")
ax2.set_ylabel("AUC (style arm)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left")
ax.set_title("(a)", fontsize=9, loc="left")
ax = axs[1]
dec = [IP["taut"][w]["decided"] for w in WS]
ax.plot(WX, dec, "k-o", markersize=3.2, linewidth=1.0)
ax.set_xlabel("Prediction time point (rallies completed)")
ax.set_ylabel("Proportion of games in which the\nleading side eventually won")
ax.set_xticks(WX); ax.set_ylim(0.55, 0.95)
ax.set_title("(b)", fontsize=9, loc="left")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUTD, "Fig3_reliability.png")); plt.close(fig)
print("Fig3 ok")

# ---------- Fig 4：赛中 vs 赛后 ----------
POST = 0.9475
fig, ax = plt.subplots(figsize=(4.4, 3.0))
inplay = [IP["t7"][w]["tech"]["mean"] for w in WS]
ax.plot(WX, inplay, "k-o", markersize=3.5, linewidth=1.1, label="In-play (information up to the time point)")
ax.axhline(POST, color="k", linestyle="--", linewidth=1.1,
           label="Post-hoc (whole-game stroke frequencies)")
ax.set_xlabel("Prediction time point (rallies completed)")
ax.set_ylabel("Test-set AUC")
ax.set_xticks(WX); ax.set_ylim(0.60, 1.00)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUTD, "Fig4_inplay_vs_posthoc.png")); plt.close(fig)
print("Fig4 ok")

# ---------- Fig 5：回合内 ----------
WR = json.load(open(os.path.join(B, "within_rally_lstm_curve.json"), encoding="utf-8"))
ks, ys, es = [], [], []
for k in ("1", "2", "3", "5", "8", "12", "all"):
    if k in WR:
        ks.append(12.5 if k == "all" else int(k))
        ys.append(WR[k]["auc_mean"]); es.append(WR[k]["auc_std"])
order = sorted(range(len(ks)), key=lambda i: ks[i])
ks = [ks[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.errorbar(ks, ys, yerr=es, fmt="k-o", markersize=3.5, linewidth=1.1, capsize=2)
ax.axhline(0.5, color="k", linestyle=":", linewidth=0.9)
ax.set_xlabel("Shots observed within the rally (k)")
ax.set_ylabel("AUC for rally outcome")
ax.set_xticks(ks); ax.set_xticklabels([str(int(k)) if k < 12 else "all" for k in ks])
ax.set_ylim(0.45, 0.90)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUTD, "Fig5_within_rally.png")); plt.close(fig)
print("Fig5 ok")

print("saved to", OUTD)
