# -*- coding: utf-8 -*-
"""
时点对齐融合（同时完成"局末收敛校验"）。

设计：所有臂都只用第 1..t 回合的信息，同模型（XGBoost）、同划分（5 折 GroupKFold 按 match_id）。
  6 个臂：
    state       当前比分差 + 势头 + 已打回合数（结果维度聚合，精确观测）
    style       累计 18 维技术频差（技术维度聚合，统计估计）
    state+style 两者拼接
    micro       微观 BiLSTM 对第 1..t 回合的折外胜率预测聚合（均值/标准差/最大/最小）
    state+micro
    all         全部

时点：绝对 w ∈ {5,10,15,20,25,30}（固定总体：回合数 ≥ 30 的局）+ 整局全量（全部局，用于收敛校验）

微观折外概率：每个外层折内训练 BiLSTM（内层再切验证集做早停），预测被留出局的逐回合胜率。
             概率会存盘，便于复用。

输出：
  results_inplay/micro_oof_rally_probs.csv
  results_inplay/timealigned_summary.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, sys, json, time
from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import common as C  # noqa: E402

ROOT = str(ROOT)
DATA = os.path.join(str(DATA), "dataset_blsr_v2.csv")
OUT = str(R2)
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_FOLDS = 5
BATCH = 32
PATIENCE = 15
EPOCHS = 100
WINDOWS = [5, 10, 15, 20, 25, 30]
WMAX = max(WINDOWS)
SMOKE = os.environ.get("SMOKE") == "1"
TAG = "_smoke" if SMOKE else ""
if SMOKE:
    EPOCHS, N_FOLDS, WINDOWS = 2, 3, [5, 30]
    WMAX = max(WINDOWS)

XGB_KW = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
              random_state=SEED, eval_metric="logloss", n_jobs=4)

print("loading data ...", flush=True)
df, META = C.load_blsr(C.BLSR_V2)
TYPES = sorted(df["type"].dropna().unique())

# ------------------------------------------------------------------ games
def load_games(df):
    out = {}
    for gid, g in df.groupby("game_id", sort=True):
        rallies = []
        for rid, rg in g.groupby("rally_id", sort=True):
            rg = rg.sort_values("time_proportion")
            s = rg[rg["is_target_turn"] == 1]
            r = rg[rg["is_target_turn"] == 0]
            rallies.append({
                "rally_id": int(rid),
                "server_types": Counter(s["type"]),
                "receiver_types": Counter(r["type"]),
                "server_shots": int(len(s)),
                "receiver_shots": int(len(r)),
                "win": int(rg["is_target_win"].iloc[0]),
            })
        out[int(gid)] = {"match_id": int(g["match_id"].iloc[0]),
                         "rallies": rallies,
                         "label": int(g["is_target_win"].iloc[-1])}
    return out

GAMES = load_games(df)
POP = sorted([gid for gid, g in GAMES.items() if len(g["rallies"]) >= WMAX])
ALL = sorted(GAMES.keys())
print("games: all=%d  window-population(>=%d rallies)=%d" % (len(ALL), WMAX, len(POP)), flush=True)

def feats_at(g, upto):
    """state(3), style(18), micro-aggregates(4, filled later) — all from rallies[:upto]."""
    rallies = g["rallies"][:upto]
    sc, rc = Counter(), Counter()
    st = rt = 0
    for r in rallies:
        sc.update(r["server_types"]); rc.update(r["receiver_types"])
        st += r["server_shots"]; rt += r["receiver_shots"]
    style = np.array([(sc.get(t, 0) / st if st else 0.0) - (rc.get(t, 0) / rt if rt else 0.0)
                      for t in TYPES], dtype=np.float32)
    score = 2 * sum(r["win"] for r in rallies) - upto
    k = min(3, upto)
    momentum = sum(r["win"] for r in rallies[upto - k:upto]) if k else 0
    state = np.array([score, momentum, upto], dtype=np.float32)
    return state, style

# ============================================================ micro OOF probabilities
OOF_CSV = os.path.join(OUT, "micro_oof_rally_probs%s.csv" % TAG)

def build_micro_oof():
    """5-fold GroupKFold on games (grouped by match_id); BiLSTM trained per fold."""
    if os.path.exists(OOF_CSV):
        print("reuse existing micro OOF probs:", OOF_CSV, flush=True)
        return pd.read_csv(OOF_CSV)
    gids = np.array(ALL)
    groups = np.array([GAMES[g]["match_id"] for g in gids])
    recs = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fi, (tr_i, te_i) in enumerate(gkf.split(gids, groups=groups), 1):
        tr_m = sorted(set(groups[tr_i]))
        # 内层验证：从训练比赛里切 20% 做早停
        rng = np.random.RandomState(SEED + fi)
        perm = rng.permutation(len(tr_m))
        n_va = max(1, int(round(0.2 * len(tr_m))))
        va_m = [tr_m[i] for i in perm[:n_va]]
        tr_m2 = [tr_m[i] for i in perm[n_va:]]
        te_m = sorted(set(groups[te_i]))
        print("  fold %d/%d: train %d / inner-val %d matches, predict %d matches"
              % (fi, N_FOLDS, len(tr_m2), len(va_m), len(te_m)), flush=True)
        tr = C.build_rally_sequences(df, tr_m2)
        va = C.build_rally_sequences(df, va_m)
        te = C.build_rally_sequences(df, te_m)
        C.set_seed(SEED + fi)
        model = C.compile_model(C.build_bilstm(META["num_types"]))
        C.train_with_early_stopping(
            model, C.make_micro_dataset(tr, BATCH, shuffle=True, seed=SEED + fi),
            C.make_micro_dataset(va, BATCH, shuffle=False), epochs=EPOCHS, patience=PATIENCE,
            verbose=0, steps_per_epoch=C.steps_for(tr, BATCH),
            validation_steps=C.steps_for(va, BATCH))
        p = C.predict_micro(model, te, BATCH)
        for s, pi in zip(te, p):
            recs.append({"game_id": s["game_id"], "rally_id": s["rally_id"],
                         "match_id": s["match_id"], "prob": float(pi)})
    out = pd.DataFrame(recs)
    out.to_csv(OOF_CSV, index=False)
    print("saved micro OOF probs:", OOF_CSV, len(out), flush=True)
    return out

t0 = time.time()
oof = build_micro_oof()
print("micro OOF done in %.1f min" % ((time.time() - t0) / 60), flush=True)
OOF = {(int(r.game_id), int(r.rally_id)): float(r.prob) for r in oof.itertuples()}

def micro_agg(g, upto):
    ps = [OOF.get((gid_of(g), r["rally_id"]), np.nan) for r in g["rallies"][:upto]]
    ps = [p for p in ps if not np.isnan(p)]
    if not ps:
        return np.zeros(4, np.float32)
    return np.array([np.mean(ps), np.std(ps), np.max(ps), np.min(ps)], dtype=np.float32)

_GID = {}

def gid_of(g):
    return g["_gid"]

# ============================================================ evaluation
def build_rows(gids, cutoff_fn):
    rows = []
    for gid in gids:
        g = GAMES[gid]
        g = dict(g); g["_gid"] = gid
        n = int(cutoff_fn(gid, g))
        n = max(1, min(n, len(g["rallies"])))
        state, style = feats_at(g, n)
        mic = micro_agg(g, n)
        rows.append({"gid": gid, "match_id": g["match_id"], "label": g["label"],
                     "state": state, "style": style, "micro": mic, "upto": n})
    return rows

ARMS = {
    "state": lambda r: r["state"],
    "style": lambda r: r["style"],
    "state+style": lambda r: np.concatenate([r["state"], r["style"]]),
    "micro": lambda r: r["micro"],
    "state+micro": lambda r: np.concatenate([r["state"], r["micro"]]),
    "all": lambda r: np.concatenate([r["state"], r["style"], r["micro"]]),
}

def cv_arms(rows):
    """Same folds for every arm -> paired per-fold AUCs."""
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["match_id"] for r in rows])
    Xs = {a: np.array([f(r) for r in rows], np.float32) for a, f in ARMS.items()}
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.zeros(len(y)), y, groups))
    res = {a: [] for a in ARMS}
    for tr, te in folds:
        for a in ARMS:
            X = Xs[a]
            m = xgb.XGBClassifier(**XGB_KW)
            m.fit(X[tr], y[tr])
            res[a].append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return res, Xs, y, groups, folds

def run_cutoff(name, gids, cutoff_fn):
    rows = build_rows(gids, cutoff_fn)
    res, *_ = cv_arms(rows)
    out = {"cutoff": name, "n_games": len(rows),
           "auc": {a: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
                   for a, v in res.items()}}
    # 配对差值（相对 state）
    for a in ARMS:
        if a == "state":
            continue
        d = np.array(res[a]) - np.array(res["state"])
        out["auc"][a]["minus_state"] = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                                        "folds": d.tolist()}
    line = "%-10s n=%3d | " % (name, len(rows))
    for a in ARMS:
        line += "%s %.4f  " % (a, np.mean(res[a]))
    print(line, flush=True)
    return out

print("\n=== absolute windows (fixed population) ===", flush=True)
summary = {"design": "time-aligned fusion; all arms use rallies[1..t]; 5-fold GroupKFold by match_id",
           "windows": [], "full": None}
for w in WINDOWS:
    summary["windows"].append(run_cutoff("w=%d" % w, POP, lambda gid, g, w=w: w))

print("\n=== full game (all games; convergence check + micro-only game-level) ===", flush=True)
summary["full"] = run_cutoff("full", ALL, lambda gid, g: len(g["rallies"]))

json.dump(summary, open(os.path.join(OUT, "timealigned_summary%s.json" % TAG), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwritten ->", os.path.join(OUT, "timealigned_summary%s.json" % TAG), flush=True)
