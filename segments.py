# -*- coding: utf-8 -*-
"""
追问：分类口径与预测口径差的 0.12 从哪来？

既然"估计精度"解释不了（w=30 噪声 28% vs 整局 25%，几乎没变），
就检验"尾部信息更值钱"：分段只用部分回合计算同一份技术特征，看各段的预测力。

段设置：
  first-m ：仅前 m 回合      （早期行为）
  last-m  ：仅后 m 回合      （后期行为，结局已基本塑造）
  whole   ：整局
对同一批局、同一模型（逻辑回归，R4 显示其最优）、同一 5 折 GroupKFold。

输出：results_inplay/segment_information.json、_facts/segment_information.md
"""
import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(ROOT))
import common as C  # noqa: E402

OUTD = str(RESULTS)
SEED, N_FOLDS = 42, 5
SEGMENTS = [3, 5, 10, 15, 20, 25, 30]
WMAX = 30

df, _ = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())

games = {}
for gid, g in df.groupby("game_id", sort=True):
    rallies = []
    for rid, rg in g.groupby("rally_id", sort=True):
        rg = rg.sort_values("time_proportion")
        s = rg[rg["is_target_turn"] == 1]
        r = rg[rg["is_target_turn"] == 0]
        rallies.append({"server_types": Counter(s["type"]), "receiver_types": Counter(r["type"]),
                        "server_shots": int(len(s)), "receiver_shots": int(len(r))})
    games[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rallies,
                       "label": int(g["is_target_win"].iloc[-1])}
POP = sorted([g for g, v in games.items() if len(v["rallies"]) >= WMAX])


def tech_of(rallies):
    sc, rc = Counter(), Counter(); st = rt = 0
    for r in rallies:
        sc.update(r["server_types"]); rc.update(r["receiver_types"])
        st += r["server_shots"]; rt += r["receiver_shots"]
    return np.array([(sc.get(t, 0) / st if st else 0.0) - (rc.get(t, 0) / rt if rt else 0.0)
                     for t in TYPES], np.float32)


def build(mode, m):
    rows = []
    for gid in POP:
        rl = games[gid]["rallies"]
        if mode == "first":
            seg = rl[:m]
        elif mode == "last":
            seg = rl[-m:]
        elif mode == "middle":
            mid = len(rl) // 2
            seg = rl[max(0, mid - m // 2): mid + m // 2]
        else:
            seg = rl
        rows.append({"match_id": games[gid]["match_id"], "label": games[gid]["label"],
                     "X": tech_of(seg)})
    return rows


def cv_auc(rows):
    y = np.array([r["label"] for r in rows]); g = np.array([r["match_id"] for r in rows])
    X = np.array([r["X"] for r in rows], np.float32)
    folds = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
        m.fit(X[tr], y[tr])
        folds.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return float(np.mean(folds)), float(np.std(folds)), folds


res = {"note": "同一批 183 局；技术频差特征；逻辑回归；5 折 GroupKFold", "population": len(POP)}
print("%-8s %-9s %-9s %-9s %-12s %-8s" % ("段长 m", "前 m 回合", "后 m 回合", "整局", "后-前", "配对 p"), flush=True)
whole_auc, whole_sd, whole_folds = cv_auc(build("whole", 0))
res["whole"] = {"auc": whole_auc, "std": whole_sd, "folds": whole_folds}
from scipy import stats as _st
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402
for m in SEGMENTS:
    a1, s1, f1 = cv_auc(build("first", m))
    a2, s2, f2 = cv_auc(build("last", m))
    d = np.array(f2) - np.array(f1)
    t, p = _st.ttest_rel(f2, f1)
    res[str(m)] = {"first": {"auc": a1, "std": s1, "folds": f1},
                   "last": {"auc": a2, "std": s2, "folds": f2},
                   "last_minus_first": {"mean": float(d.mean()), "p": float(p)}}
    print("%-8d %-9.4f %-9.4f %-9.4f %-12s %-8.3f"
          % (m, a1, a2, whole_auc, "%+.4f" % d.mean(), p), flush=True)

json.dump(res, open(os.path.join(OUTD, "segment_information.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

L = ["# 分段信息检验：分类与预测的 0.12 从哪来", "",
     "同一批 %d 局；特征为 18 维技术频差；逻辑回归；5 折按比赛分组。" % len(POP),
     "整局基准 AUC = **%.4f**（±%.4f）" % (whole_auc, whole_sd), "",
     "| 段长 m | 仅前 m 回合 | 仅后 m 回合 | 整局 |", "|---|---|---|---|"]
for m in SEGMENTS:
    e = res[str(m)]
    L.append("| %d | %.4f | %.4f | %.4f |" % (m, e["first"]["auc"], e["last"]["auc"], whole_auc))
L += ["", "**读法**：若「后 m 回合」的曲线明显高于「前 m 回合」且很快逼近整局值，",
      "说明比赛后段的技战术行为已被走势/结果塑造——这正是赛后分类口径虚高的来源。"]
print("\nwritten -> segment_information.json / _facts/segment_information.md")
