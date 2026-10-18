# -*- coding: utf-8 -*-
"""
重绘 图6（赛中预测 vs 赛后分类，逻辑回归版）与 图7（回合级前 k 拍，BiLSTM 固定划分版）。
数据来源：
  图6 ← results_inplay/inplay_lr.json
  图7 ← results_inplay/e5_fixed_k*.json（三段合并）
输出：results_inplay/fig_inplay_auc_lr.png / fig_within_rally_lstm.png
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
OUT = str(RESULTS)

# ============================================================ 图6
R = json.load(open(os.path.join(OUT, "inplay_lr.json"), encoding="utf-8"))
ws = [5, 10, 15, 20, 25, 30]
main_m = [R["t7"][str(w)]["main"]["mean"] for w in ws]
main_s = [R["t7"][str(w)]["main"]["std"] for w in ws]
tech_m = [R["t7"][str(w)]["tech"]["mean"] for w in ws]
state_m = [R["t7"][str(w)]["state"]["mean"] for w in ws]
posthoc = R["t7_full"]["tech"]["mean"]

fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=300)
ax.errorbar(ws, main_m, yerr=main_s, fmt="o-", capsize=4, label="赛中预测：技术频差+比分+势头")
ax.plot(ws, tech_m, "^-", alpha=0.75, label="仅技术频差（截至该时点）")
ax.plot(ws, state_m, "s--", alpha=0.75, label="仅比分+势头")
ax.axhline(posthoc, color="crimson", ls=":", lw=1.6,
           label="整局全量技术频差（赛后分类）= %.3f" % posthoc)
ax.axhline(0.5, color="gray", ls="-.", lw=1.0, label="随机基线 0.5")
ax.set_xlabel("预测时点：已完成回合数 $w$")
ax.set_ylabel("局级胜负 AUC（5 折按比赛分组 CV，逻辑回归）")
ax.set_title("赛中预测 vs 赛后分类")
ax.set_ylim(0.55, max(0.98, posthoc + 0.05))
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_inplay_auc_lr.png"))
fig.savefig(os.path.join(OUT, "fig_inplay_auc_lr.pdf"))
plt.close(fig)
print("图6 ->", "fig_inplay_auc_lr.png", "posthoc=%.4f" % posthoc)

# ============================================================ 图7
merged = {}
for p in glob.glob(os.path.join(OUT, "e5_fixed_k*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    merged.update(d.get("ks", {}))
KS = ["1", "2", "3", "5", "8", "12", "all"]
xs = list(range(len(KS)))
mean = [merged[k]["auc_mean"] for k in KS]
std = [merged[k]["auc_std"] for k in KS]

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
ax.errorbar(xs, mean, yerr=std, fmt="o-", capsize=4, label="回合内实时预测（前 k 拍，BiLSTM）")
for x, m in zip(xs, mean):
    ax.annotate("%.3f" % m, (x, m), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5)
ax.axhline(0.5, color="gray", ls="-.", lw=1.0, label="随机基线 0.5")
ax.set_xticks(xs)
ax.set_xticklabels(["前 %s 拍" % k if k != "all" else "全部拍" for k in KS], fontsize=9)
ax.set_xlabel("已观测拍数")
ax.set_ylabel("回合胜负 AUC（固定测试集，5 次独立运行）")
ax.set_title("回合内实时预测：看到第几拍才能判断这一分")
ax.set_ylim(0.46, 0.88)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_within_rally_lstm.png"))
fig.savefig(os.path.join(OUT, "fig_within_rally_lstm.pdf"))
plt.close(fig)
print("图7 ->", "fig_within_rally_lstm.png")
for k, m, s in zip(KS, mean, std):
    print("   k=%-4s %.4f ± %.4f" % (k, m, s))
json.dump({k: merged[k] for k in KS}, open(os.path.join(OUT, "within_rally_lstm_curve.json"), "w",
                                           encoding="utf-8"), ensure_ascii=False, indent=1)
