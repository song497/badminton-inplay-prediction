# -*- coding: utf-8 -*-
"""
赛中章的概率质量/决策价值指标（逻辑回归口径）：
各时点、各臂的 AUC / 准确率 / Brier / logloss / ECE / 净收益（pt=0.3,0.5）
输出：results_inplay/quality_lr.json 与 _facts/quality_lr.md
"""
import os, json
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import common as C
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

OUT = str(RESULTS)
SEED, NF = 42, 5
WS = [5, 10, 15, 20, 25, 30]
WMAX = max(WS)

df, META = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())


def load_games(df):
    out = {}
    for gid, g in df.groupby("game_id", sort=True):
        rl = []
        for rid, rg in g.groupby("rally_id", sort=True):
            rg = rg.sort_values("time_proportion")
            s = rg[rg["is_target_turn"] == 1]; r = rg[rg["is_target_turn"] == 0]
            rl.append({"rally_id": int(rid), "server_types": Counter(s["type"]),
                       "receiver_types": Counter(r["type"]), "server_shots": int(len(s)),
                       "receiver_shots": int(len(r)), "win": int(rg["is_target_win"].iloc[0])})
        out[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rl,
                         "label": int(g["is_target_win"].iloc[-1])}
    return out


GAMES = load_games(df)
POP = sorted([g for g, x in GAMES.items() if len(x["rallies"]) >= WMAX])
OOF = {}
for r in __import__("pandas").read_csv(os.path.join(OUT, "micro_oof_rally_probs.csv")).itertuples():
    OOF[(int(r.game_id), int(r.rally_id))] = float(r.prob)


def feats(gid, upto):
    g = GAMES[gid]; R = g["rallies"][:upto]
    sc, rc = Counter(), Counter(); st = rt = 0
    for x in R:
        sc.update(x["server_types"]); rc.update(x["receiver_types"])
        st += x["server_shots"]; rt += x["receiver_shots"]
    style = np.array([(sc.get(t, 0) / st if st else 0) - (rc.get(t, 0) / rt if rt else 0) for t in TYPES], np.float32)
    score = 2 * sum(x["win"] for x in R) - upto
    k = min(3, upto); mom = sum(x["win"] for x in R[upto - k:upto]) if k else 0
    state = np.array([score, mom, upto], np.float32)
    ps = [OOF.get((gid, x["rally_id"]), np.nan) for x in R]
    ps = [p for p in ps if not np.isnan(p)]
    mic = np.array([np.mean(ps), np.std(ps), np.max(ps), np.min(ps)], np.float32) if ps else np.zeros(4, np.float32)
    return state, style, mic


ARMS = {"state": ("state",), "style": ("style",), "micro": ("micro",),
        "style+micro": ("style", "micro"), "state+style": ("state", "style"),
        "state+micro": ("state", "micro"), "all": ("state", "style", "micro")}


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def nb(y, p, pt):
    m = p >= pt
    if m.sum() == 0:
        return 0.0
    return float((y[m] == 1).sum() / len(y) - (y[m] == 0).sum() / len(y) * (pt / (1 - pt)))


res = {"note": "逻辑回归；每臂每时点用 5 折 GroupKFold 的折外概率计算", "windows": {}}
for w in WS:
    rows = []
    for gid in POP:
        st, sty, mic = feats(gid, w)
        rows.append({"m": GAMES[gid]["match_id"], "y": GAMES[gid]["label"],
                     "state": st, "style": sty, "micro": mic})
    y = np.array([r["y"] for r in rows]); g = np.array([r["m"] for r in rows])
    folds = list(GroupKFold(n_splits=NF).split(np.zeros(len(y)), y, g))
    ent = {}
    for a, parts in ARMS.items():
        X = np.array([np.concatenate([np.atleast_1d(r[p]) for p in parts]) for r in rows], np.float32)
        oof = np.zeros(len(y)); oof[:] = np.nan
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        ent[a] = {"auc": float(roc_auc_score(y, oof)), "acc": float(np.mean((oof > 0.5) == y)),
                  "brier": float(np.mean((oof - y) ** 2)), "logloss": float(-np.mean(y * np.log(np.clip(oof, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - oof, 1e-9, 1)))),
                  "ece": ece(y, oof), "nb03": nb(y, oof, 0.3), "nb05": nb(y, oof, 0.5)}
    res["windows"][str(w)] = ent
    print("w=%-3d " % w + "  ".join("%s AUC%.3f Br%.3f ECE%.3f" % (a, ent[a]["auc"], ent[a]["brier"], ent[a]["ece"])
                                    for a in ("state", "style", "all")), flush=True)

json.dump(res, open(os.path.join(OUT, "quality_lr.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

L = ["# 赛后章概率质量与决策价值（逻辑回归，5 折折外）", "",
     "| 时点 | 臂 | AUC | 准确率 | Brier | logloss | ECE | 净收益(pt=.3) | 净收益(pt=.5) |",
     "|---|---|---|---|---|---|---|---|---|"]
for w in WS:
    for a in ARMS:
        e = res["windows"][str(w)][a]
        L.append("| w=%d | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %+.3f |"
                 % (w, a, e["auc"], e["acc"], e["brier"], e["logloss"], e["ece"], e["nb03"], e["nb05"]))
print("written")
