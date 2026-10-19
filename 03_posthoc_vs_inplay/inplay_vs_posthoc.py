# -*- coding: utf-8 -*-
"""
赛中预测一章的**逻辑回归版**全量复算（与表2/表5 的学习器统一）。
覆盖：表7（3臂×6时点+整局全量）、相对进程、表8（势头变体）、达标信息量、准确率阶梯、同义反复。
学习器：StandardScaler + LogisticRegression(L2, max_iter=5000)。
输出：results_inplay/inplay_lr.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json
import numpy as np
import scipy.stats as st
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import common as C

ROOT = str(ROOT)
OUT = str(R3)
SEED, N_FOLDS = 42, 5
WINDOWS = [5, 10, 15, 20, 25, 30]
WMAX = max(WINDOWS)

df, META = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())

def load_games(df):
    out = {}
    for gid, g in df.groupby("game_id", sort=True):
        rallies = []
        for rid, rg in g.groupby("rally_id", sort=True):
            rg = rg.sort_values("time_proportion")
            s = rg[rg["is_target_turn"] == 1]
            r = rg[rg["is_target_turn"] == 0]
            rallies.append({"rally_id": int(rid), "server_types": Counter(s["type"]),
                            "receiver_types": Counter(r["type"]),
                            "server_shots": int(len(s)), "receiver_shots": int(len(r)),
                            "win": int(rg["is_target_win"].iloc[0])})
        out[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rallies,
                         "label": int(g["is_target_win"].iloc[-1])}
    return out

GAMES = load_games(df)
POP = sorted([gid for gid, g in GAMES.items() if len(g["rallies"]) >= WMAX])   # 183
ALL = sorted(GAMES.keys())                                                    # 223
print("games all=%d pop=%d" % (len(ALL), len(POP)), flush=True)

def tech(rallies):
    sc, rc = Counter(), Counter()
    st = rt = 0
    for r in rallies:
        sc.update(r["server_types"]); rc.update(r["receiver_types"])
        st += r["server_shots"]; rt += r["receiver_shots"]
    return np.array([(sc.get(t, 0) / st if st else 0.0) - (rc.get(t, 0) / rt if rt else 0.0)
                     for t in TYPES], dtype=np.float32)

def mom_k(rallies, k):
    if not rallies:
        return 0.0
    kk = min(k, len(rallies))
    return float(sum(r["win"] for r in rallies[len(rallies) - kk:]))

def streak(rallies):
    c = 0
    for r in reversed(rallies):
        if r["win"] == 1:
            c += 1
        else:
            break
    return float(c)

def row_of(gid, upto):
    g = GAMES[gid]
    Rl = g["rallies"][:upto]
    score = 2 * sum(r["win"] for r in Rl) - upto
    return {"match_id": g["match_id"], "label": g["label"], "upto": int(upto),
            "tech": tech(Rl), "score": np.float32(score), "upto_f": np.float32(upto),
            "mom1": np.float32(mom_k(Rl, 1)), "mom3": np.float32(mom_k(Rl, 3)),
            "mom5": np.float32(mom_k(Rl, 5)), "streak": np.float32(streak(Rl))}

def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))

def cv_auc_folds(X, y, grp):
    fa = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, grp):
        m = mk().fit(X[tr], y[tr])
        fa.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return fa

def cat(r, parts):
    return np.concatenate([np.atleast_1d(r[p]) for p in parts]).astype(np.float32)

def describe(fa):
    return {"mean": float(np.mean(fa)), "std": float(np.std(fa, ddof=0)), "folds": [float(x) for x in fa]}

R = {"note": "赛中章逻辑回归版（与表2/表5 统一）；5 折 GroupKFold 按 match_id，seed 42"}

# ---------------- A. 表7
T7_ARMS = [("main", ("tech", "score", "mom3")), ("tech", ("tech",)), ("state", ("score", "mom3"))]
R["t7"] = {}
for w in WINDOWS:
    rows = [row_of(gid, w) for gid in POP]
    y = np.array([r["label"] for r in rows]); grp = np.array([r["match_id"] for r in rows])
    e = {}
    for name, parts in T7_ARMS:
        e[name] = describe(cv_auc_folds(np.array([cat(r, parts) for r in rows], np.float32), y, grp))
    X = np.array([cat(r, T7_ARMS[0][1]) for r in rows], np.float32)
    accs = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, grp):
        m = mk().fit(X[tr], y[tr])
        accs.append(float(np.mean((m.predict_proba(X[te])[:, 1] > 0.5) == y[te])))
    e["acc_main"] = describe(accs)
    R["t7"][str(w)] = e
    print("w=%-3d main %.4f | tech %.4f | state %.4f | acc %.3f"
          % (w, e["main"]["mean"], e["tech"]["mean"], e["state"]["mean"], e["acc_main"]["mean"]), flush=True)

# 整局全量（赛后分类口径，同 183 局同一折）
rows = [row_of(gid, len(GAMES[gid]["rallies"])) for gid in POP]
R["t7_full"] = {"tech": describe(cv_auc_folds(
    np.array([r["tech"] for r in rows], np.float32),
    np.array([r["label"] for r in rows]), np.array([r["match_id"] for r in rows])))}
print("整局全量 tech %.4f" % R["t7_full"]["tech"]["mean"], flush=True)

# ---------------- B. 相对进程（全部 223 局）
R["rel"] = {}
for pct in (25, 50, 75):
    rows = [row_of(gid, max(1, int(round(len(GAMES[gid]["rallies"]) * pct / 100.0)))) for gid in ALL]
    y = np.array([r["label"] for r in rows]); grp = np.array([r["match_id"] for r in rows])
    X = np.array([cat(r, ("tech", "score", "mom3")) for r in rows], np.float32)
    R["rel"][str(pct)] = describe(cv_auc_folds(X, y, grp))
    print("进程 %d%%: %.4f" % (pct, R["rel"][str(pct)]["mean"]), flush=True)

# ---------------- C. 表8（势头变体）
T8_ARMS = [("score_w", ("score", "upto_f")), ("mom1", ("score", "upto_f", "mom1")),
           ("mom3", ("score", "upto_f", "mom3")), ("mom5", ("score", "upto_f", "mom5")),
           ("streak", ("score", "upto_f", "streak"))]
R["t8"] = {}
for tag in ("10", "20", "30", "p50"):
    if tag == "p50":
        rows = [row_of(gid, max(1, int(round(len(GAMES[gid]["rallies"]) * 0.5)))) for gid in ALL]
    else:
        rows = [row_of(gid, int(tag)) for gid in POP]
    y = np.array([r["label"] for r in rows]); grp = np.array([r["match_id"] for r in rows])
    d = {name: describe(cv_auc_folds(np.array([cat(r, parts) for r in rows], np.float32), y, grp))
         for name, parts in T8_ARMS}
    R["t8"][tag] = d
    print("表8 %-4s " % tag + "  ".join("%s %.4f" % (k, v["mean"]) for k, v in d.items()), flush=True)

# ---------------- D. 达标信息量
def t2c_for(keyf):
    curves = np.array([R["t7"][str(w)][keyf]["folds"] for w in WINDOWS])
    out = {}
    for tgt in (0.70, 0.75, 0.80):
        need = []
        for f in range(curves.shape[1]):
            c = curves[:, f]
            for i in range(len(WINDOWS) - 1):
                if c[i] < tgt <= c[i + 1]:
                    need.append(WINDOWS[i] + (tgt - c[i]) * (WINDOWS[i + 1] - WINDOWS[i]) / (c[i + 1] - c[i]))
                    break
        if need:
            m = float(np.mean(need)); se = float(np.std(need, ddof=1) / np.sqrt(len(need)))
            ci = float(st.t.ppf(0.975, len(need) - 1) * se)
            out[str(tgt)] = {"mean": m, "ci": [m - ci, m + ci], "n_folds": len(need)}
        else:
            out[str(tgt)] = {"mean": None, "note": "w<=30 内未达到", "n_folds": 0}
    return out

R["t2c"] = {"main": t2c_for("main"), "state": t2c_for("state")}
print("t2c main:", json.dumps(R["t2c"]["main"], ensure_ascii=False), flush=True)
print("t2c state:", json.dumps(R["t2c"]["state"], ensure_ascii=False), flush=True)

# ---------------- E. 同义反复
R["taut"] = {}
for w in WINDOWS:
    decided, gaps, close = [], [], []
    for gid in POP:
        Rl = GAMES[gid]["rallies"][:w]
        sc = 2 * sum(r["win"] for r in Rl) - w
        gaps.append(abs(sc))
        if sc != 0:
            decided.append(1.0 if (sc > 0) == (GAMES[gid]["label"] == 1) else 0.0)
        if abs(sc) <= 2:
            close.append(row_of(gid, w))
    e = {"decided": float(np.mean(decided)), "mean_abs_gap": float(np.mean(gaps)),
         "n_non_tied": len(decided), "n_close": len(close), "n": len(POP)}
    yc = np.array([r["label"] for r in close]); gc = np.array([r["match_id"] for r in close])
    if len(set(gc)) >= N_FOLDS and len(set(yc)) == 2:
        for name, parts in (("state", ("score", "mom3")), ("tech", ("tech",))):
            e[name] = describe(cv_auc_folds(np.array([cat(r, parts) for r in close], np.float32), yc, gc))
    R["taut"][str(w)] = e
    print("taut w=%-3d decided %.3f close n=%d state %s tech %s"
          % (w, e["decided"], e["n_close"],
             ("%.4f" % e["state"]["mean"]) if "state" in e else "-",
             ("%.4f" % e["tech"]["mean"]) if "tech" in e else "-"), flush=True)

json.dump(R, open(os.path.join(OUT, "inplay_lr.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE ->", os.path.join(OUT, "inplay_lr.json"))
