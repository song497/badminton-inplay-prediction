# -*- coding: utf-8 -*-
"""
按论文口径（未决子集重训 + 5折 + 折均值）计算：
每个时点「仅宏观 − 仅状态」的逐折差值，给出折均值差与 t 区间(95%, df=4)。
"""
import os, json
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import scipy.stats as st
import common as C
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402

OUT = str(RESULTS)


def load(df):
    o = {}
    for gid, g in df.groupby("game_id", sort=True):
        rl = []
        for rid, rg in g.groupby("rally_id", sort=True):
            rg = rg.sort_values("time_proportion")
            s = rg[rg["is_target_turn"] == 1]; r = rg[rg["is_target_turn"] == 0]
            rl.append({"server_types": Counter(s["type"]), "receiver_types": Counter(r["type"]),
                       "server_shots": int(len(s)), "receiver_shots": int(len(r)),
                       "win": int(rg["is_target_win"].iloc[0])})
        o[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rl,
                       "label": int(g["is_target_win"].iloc[-1])}
    return o


df, M = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())
G = load(df)
POP = sorted([g for g, x in G.items() if len(x["rallies"]) >= 30])


def row(gid, w):
    R = G[gid]["rallies"][:w]
    sc, rc = Counter(), Counter(); sN = rN = 0
    for x in R:
        sc.update(x["server_types"]); rc.update(x["receiver_types"])
        sN += x["server_shots"]; rN += x["receiver_shots"]
    tech = np.array([(sc.get(t, 0) / sN if sN else 0) - (rc.get(t, 0) / rN if rN else 0) for t in TYPES], np.float32)
    score = 2 * sum(x["win"] for x in R) - w
    k = min(3, w); mom = np.float32(sum(x["win"] for x in R[w - k:w]) if k else 0)
    return (G[gid]["match_id"], G[gid]["label"], tech, np.float32(score), mom, abs(score))


print("%-6s %-5s %-8s %-8s %-9s %-20s %s" % ("时点", "局数", "仅状态", "仅宏观", "折均差", "95%CI(折间t)", "显著"))
out = {}
for w in (5, 10, 15, 20, 25, 30):
    close = [row(g, w) for g in POP if row(g, w)[5] <= 2]
    y = np.array([c[1] for c in close]); grp = np.array([c[0] for c in close])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, grp))
    per = {}
    for name, idx in (("state", (3, 4)), ("tech", (2,))):
        X = np.array([np.concatenate([np.atleast_1d(c[i]) for i in idx]).astype(np.float32) for c in close])
        fa = []
        for tr, te in folds:
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)).fit(X[tr], y[tr])
            fa.append(roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1]))
        per[name] = np.array(fa)
    d = per["tech"] - per["state"]
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    ci = st.t.ppf(0.975, len(d) - 1) * se
    sig = "显著" if (m - ci > 0 or m + ci < 0) else "不显著"
    out[w] = {"n": len(close), "state": float(per["state"].mean()), "tech": float(per["tech"].mean()),
              "diff": float(m), "ci": [float(m - ci), float(m + ci)], "sig": sig}
    print("%-6s %-5d %-8.3f %-8.3f %+-9.3f [%+.3f, %+.3f]%s %s" % (
        "w=%d" % w, len(close), per["state"].mean(), per["tech"].mean(), m, m - ci, m + ci, " " * 3, sig))

lead_tech = sum(1 for w in out if out[w]["diff"] > 0)
print("\n宏观领先时点: %d/6 ；状态领先: %d/6" % (lead_tech, 6 - lead_tech))
print("显著时点数: %d/6" % sum(1 for w in out if out[w]["sig"] == "显著"))
json.dump(out, open(os.path.join(OUT, "close_subset_foldmean.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
