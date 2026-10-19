# -*- coding: utf-8 -*-
"""
重绘 图6（赛中预测 vs 赛后分类，逻辑回归版）与 图7（回合级前 k 拍，BiLSTM 固定划分版）。
数据来源：
  图6 ← results_inplay/inplay_lr.json
  图7 ← results_inplay/e5_fixed_k*.json（三段合并）
输出：results_inplay/fig_inplay_auc_lr.png / fig_within_rally_lstm.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
# ============================================================ 图7
merged = {}
for p in glob.glob(os.path.join(str(R2), "e5_fixed_k*.json")):
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
fig.savefig(os.path.join(str(FIGURES), "fig_within_rally_lstm.png"))
fig.savefig(os.path.join(str(FIGURES), "fig_within_rally_lstm.pdf"))
plt.close(fig)
print("图7 ->", "fig_within_rally_lstm.png")
for k, m, s in zip(KS, mean, std):
    print("   k=%-4s %.4f ± %.4f" % (k, m, s))
json.dump({k: merged[k] for k in KS}, open(os.path.join(str(R2), "within_rally_lstm_curve.json"), "w",
                                           encoding="utf-8"), ensure_ascii=False, indent=1)
