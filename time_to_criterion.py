# -*- coding: utf-8 -*-
"""
R1 + R4 + R6 + R7 合并分析。

R1 达标信息量（time-to-confidence）：达到 AUC 0.70/0.75/0.80 所需的最小时点，
   并用**按比赛重抽样的 bootstrap** 给出置信区间。
R4 边界是否模型无关：把 XGBoost 换成随机森林 / 逻辑回归，看时点曲线与排序是否一致。
R6 绝对时点 vs 相对进程：两种时点定义的曲线对比。
R7 最简状态量：只用"比分差"（1 维）能逼近完整状态臂（3 维）多少。

输出：results_inplay/t2c_summary.json、_facts/t2c.md
"""
import os, json
from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402

ROOT = str(ROOT)
OUTD = os.path.join(ROOT, "results_inplay")
DATA = os.path.join(ROOT, "data", "dataset_blsr_v2.csv")
SEED, N_FOLDS, NBOOT = 42, 5, 2000
WINDOWS = [5, 10, 15, 20, 25, 30]
REL = [0.25, 0.50, 0.75]
WMAX = max(WINDOWS)
THRESH = [0.70, 0.75, 0.80]

df = pd.read_csv(DATA)
TYPES = sorted(df["type"].dropna().unique())
games = {}
for gid, g in df.groupby("game_id", sort=True):
    rallies = []
    for rid, rg in g.groupby("rally_id", sort=True):
        rg = rg.sort_values("time_proportion")
        s = rg[rg["is_target_turn"] == 1]; r = rg[rg["is_target_turn"] == 0]
        rallies.append({"rally_id": int(rid), "server_types": Counter(s["type"]),
                        "receiver_types": Counter(r["type"]),
                        "server_shots": int(len(s)), "receiver_shots": int(len(r)),
                        "win": int(rg["is_target_win"].iloc[0])})
    games[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rallies,
                       "label": int(g["is_target_win"].iloc[-1])}

MICRO = {}
_o = pd.read_csv(os.path.join(OUTD, "micro_oof_rally_probs.csv"))
for r in _o.itertuples():
    MICRO[(int(r.game_id), int(r.rally_id))] = float(r.prob)


def feats(gid, upto):
    g = games[gid]
    rl = g["rallies"][:upto]
    sc, rc = Counter(), Counter(); st = rt = 0
    for x in rl:
        sc.update(x["server_types"]); rc.update(x["receiver_types"])
        st += x["server_shots"]; rt += x["receiver_shots"]
    style = np.array([(sc.get(t, 0) / st if st else 0.0) - (rc.get(t, 0) / rt if rt else 0.0)
                      for t in TYPES], np.float32)
    score = 2 * sum(x["win"] for x in rl) - len(rl)
    momentum = sum(x["win"] for x in rl[-min(3, len(rl)):]) if rl else 0
    ps = [MICRO.get((gid, x["rally_id"]), np.nan) for x in rl]
    ps = [p for p in ps if not np.isnan(p)]
    mic = (np.array([np.mean(ps), np.std(ps), np.max(ps), np.min(ps)], np.float32)
           if ps else np.zeros(4, np.float32))
    return (np.array([score], np.float32),
            np.array([score, momentum, upto], np.float32), style, mic)


ARMS = {"min_state": 0, "state": 1, "style": 2, "style+micro": None, "all": None}


def build(gids, upto_fn):
    rows = []
    for gid in gids:
        n = max(1, min(int(upto_fn(gid)), len(games[gid]["rallies"])))
        ms, st, sty, mi = feats(gid, n)
        rows.append({"gid": gid, "match_id": games[gid]["match_id"], "label": games[gid]["label"],
                     "min_state": ms, "state": st, "style": sty, "mic": mi})
    return rows


def mat(rows, arm):
    if arm == "min_state":
        return np.array([r["min_state"] for r in rows], np.float32)
    if arm == "state":
        return np.array([r["state"] for r in rows], np.float32)
    if arm == "style":
        return np.array([r["style"] for r in rows], np.float32)
    if arm == "style+micro":
        return np.array([np.concatenate([r["style"], r["mic"]]) for r in rows], np.float32)
    return np.array([np.concatenate([r["state"], r["style"], r["mic"]]) for r in rows], np.float32)


def make_model(kind):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=600, max_depth=8, random_state=SEED, n_jobs=4)
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=SEED,
                             eval_metric="logloss", n_jobs=4)


def oof(rows, arm, kind="xgb"):
    y = np.array([r["label"] for r in rows]); g = np.array([r["match_id"] for r in rows])
    X = mat(rows, arm)
    p = np.full(len(y), np.nan)
    fold_auc = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = make_model(kind); m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        fold_auc.append(float(roc_auc_score(y[te], p[te])))
    return y, g, p, fold_auc


def boot_curve(y, g, probs_by_w, nboot=NBOOT, seed=SEED):
    """按比赛重抽样，得到各时点 AUC 的 bootstrap 分布"""
    rng = np.random.RandomState(seed)
    matches = np.unique(g)
    idx_by_match = {m: np.where(g == m)[0] for m in matches}
    out = {w: [] for w in probs_by_w}
    for _ in range(nboot):
        pick = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in pick])
        for w, p in probs_by_w.items():
            yy = y[idx]
            if len(set(yy)) < 2:
                continue
            out[w].append(roc_auc_score(yy, p[idx]))
    return {w: np.array(v) for w, v in out.items()}


def crossing(curve, thr, grid):
    """curve: dict grid->auc ; 返回首次达到 thr 的网格点（线性插值），未达到返回 nan"""
    xs = sorted(grid)
    for i in range(len(xs)):
        if curve[xs[i]] >= thr:
            if i == 0:
                return float(xs[0])
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = curve[x0], curve[x1]
            return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)
    return float("nan")


res = {}

# ---------------- R1：绝对时点曲线 + bootstrap 置信区间
POP = sorted([g for g, v in games.items() if len(v["rallies"]) >= WMAX])
rows_w = {w: build(POP, lambda gid, w=w: w) for w in WINDOWS}
probs, fold_aucs = {}, {}
for arm in ARMS:
    probs[arm], fold_aucs[arm] = {}, {}
    for w in WINDOWS:
        y, g, p, fa = oof(rows_w[w], arm)
        probs[arm][w] = p
        fold_aucs[arm][w] = fa
    y = np.array([r["label"] for r in rows_w[WINDOWS[0]]])
    g = np.array([r["match_id"] for r in rows_w[WINDOWS[0]]])
    # 主口径：折 AUC 均值（与论文其他表一致）+ 逐折插值的达标信息量
    foldmean = {w: float(np.mean(fold_aucs[arm][w])) for w in WINDOWS}
    per_fold_t2c = {thr: [] for thr in THRESH}
    for fi in range(N_FOLDS):
        cur = {w: fold_aucs[arm][w][fi] for w in WINDOWS}
        for thr in THRESH:
            c = crossing(cur, thr, WINDOWS)
            if c == c:
                per_fold_t2c[thr].append(c)
    bc = boot_curve(y, g, probs[arm])
    res[arm] = {"auc_foldmean": foldmean,
                "auc_pooled": {w: float(roc_auc_score(y, probs[arm][w])) for w in WINDOWS},
                "auc_ci_boot": {w: [float(np.percentile(bc[w], 2.5)), float(np.percentile(bc[w], 97.5))]
                                for w in WINDOWS},
                "t2c_foldmean": {}, "t2c_foldmean_ci": {}, "t2c_pooled": {}}
    for thr in THRESH:
        pt = crossing(foldmean, thr, WINDOWS)
        arr = np.array(per_fold_t2c[thr])
        res[arm]["t2c_foldmean"][str(thr)] = pt
        if len(arr) >= 2:
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            from scipy import stats as _st
            tc = _st.t.ppf(0.975, len(arr) - 1)
            res[arm]["t2c_foldmean_ci"][str(thr)] = [float(arr.mean() - tc * se), float(arr.mean() + tc * se)]
        else:
            res[arm]["t2c_foldmean_ci"][str(thr)] = None
        res[arm]["t2c_pooled"][str(thr)] = crossing(
            {w: float(roc_auc_score(y, probs[arm][w])) for w in WINDOWS}, thr, WINDOWS)
    print("R1 %-12s " % arm + " ".join("w=%d:%.3f" % (w, foldmean[w]) for w in WINDOWS), flush=True)
    print("    达标信息量(折均值) " + str({k: (round(v, 1) if v == v else None)
                                          for k, v in res[arm]["t2c_foldmean"].items()})
          + "  CI " + str({k: ([round(x, 1) for x in v] if v else None)
                           for k, v in res[arm]["t2c_foldmean_ci"].items()}), flush=True)

# ---------------- R7：最简状态量
print("\nR7 最简状态量（仅比分差，1 维）vs 完整状态臂（3 维）", flush=True)
res["min_state_vs_state"] = {}
for w in WINDOWS:
    y, g, p1, _f1 = oof(rows_w[w], "min_state")
    y2, g2, p3, _f2 = oof(rows_w[w], "state")
    res["min_state_vs_state"][str(w)] = {"min_state": float(roc_auc_score(y, p1)),
                                         "state": float(roc_auc_score(y2, p3))}
    print("  w=%-3d 仅比分 %.4f ｜ 完整状态 %.4f" % (w, res["min_state_vs_state"][str(w)]["min_state"],
                                                     res["min_state_vs_state"][str(w)]["state"]), flush=True)

# ---------------- R4：模型无关性
print("\nR4 模型无关性（同一批特征，换学习器）", flush=True)
res["model_agnostic"] = {}
for kind in ("xgb", "rf", "lr"):
    res["model_agnostic"][kind] = {}
    for arm in ("state", "style"):
        curve = {}
        for w in WINDOWS:
            y, g, p, _f = oof(rows_w[w], arm, kind)
            curve[w] = float(roc_auc_score(y, p))
        res["model_agnostic"][kind][arm] = curve
        print("  %-4s %-6s " % (kind, arm) + " ".join("%.3f" % curve[w] for w in WINDOWS), flush=True)

# ---------------- R6：相对进程
print("\nR6 相对进程曲线", flush=True)
res["relative"] = {}
for arm in ("state", "style"):
    r = {}
    for q in REL:
        rows = build(sorted(games), lambda gid, q=q: max(1, int(round(q * len(games[gid]["rallies"])))))
        y, g, p, _f = oof(rows, arm)
        r[str(q)] = float(roc_auc_score(y, p))
    res["relative"][arm] = r
    print("  %-6s " % arm + " ".join("%s:%.4f" % (k, v) for k, v in r.items()), flush=True)

json.dump(res, open(os.path.join(OUTD, "t2c_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwritten -> t2c_summary.json")
