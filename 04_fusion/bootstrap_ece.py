# -*- coding: utf-8 -*-
"""
对「引入更多信息后期望校准误差上升」这一结论做自助法检验：
对 w=5/15/30，分别用 bootstrap 给出「仅状态」与「全部」臂 ECE 的 95% 区间，
以及两者差值 (全部 − 仅状态) 的区间，判断该结论是否稳健。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import common as C

OUT = str(RESULTS)
SEED, NF, B = 42, 5, 2000
WS = [5, 15, 30]
WMAX = max(WS)

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

df, META = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())
GAMES = load_games(df)
POP = sorted([g for g, x in GAMES.items() if len(x["rallies"]) >= WMAX])
OOF = {}
for r in __import__("pandas").read_csv(os.path.join(str(R2), "micro_oof_rally_probs.csv")).itertuples():
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

def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

rng = np.random.default_rng(SEED)
res = {}
for w in WS:
    rows = []
    for gid in POP:
        st, sty, mic = feats(gid, w)
        rows.append({"m": GAMES[gid]["match_id"], "y": GAMES[gid]["label"], "state": st, "all": np.concatenate([st, sty, mic])})
    y = np.array([r["y"] for r in rows]); g = np.array([r["m"] for r in rows])
    folds = list(GroupKFold(n_splits=NF).split(np.zeros(len(y)), y, g))
    oof = {}
    for arm in ("state", "all"):
        X = np.array([r[arm] for r in rows], np.float32)
        o = np.zeros(len(y))
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)).fit(X[tr], y[tr])
            o[te] = m.predict_proba(X[te])[:, 1]
        oof[arm] = o
    e_state, e_all, diff = [], [], []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        a = ece(y[idx], oof["state"][idx]); b = ece(y[idx], oof["all"][idx])
        e_state.append(a); e_all.append(b); diff.append(b - a)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    res[w] = {"ece_state": ece(y, oof["state"]), "ece_all": ece(y, oof["all"]),
              "state_ci": q(e_state), "all_ci": q(e_all),
              "diff": float(np.mean(diff)), "diff_ci": q(diff),
              "p_diff_gt0": float(np.mean(np.array(diff) > 0))}
    print("w=%-3d 仅状态 %.3f %s | 全部 %.3f %s | 差 %+.3f %s | P(全部>仅状态)=%.2f" % (
        w, res[w]["ece_state"], tuple(round(x, 3) for x in res[w]["state_ci"]),
        res[w]["ece_all"], tuple(round(x, 3) for x in res[w]["all_ci"]),
        res[w]["diff"], tuple(round(x, 3) for x in res[w]["diff_ci"]), res[w]["p_diff_gt0"]))

json.dump(res, open(os.path.join(OUT, "ece_bootstrap.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved ece_bootstrap.json")
