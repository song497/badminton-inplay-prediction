# -*- coding: utf-8 -*-
"""
时间对齐七臂实验的**学习器对照**版本：同一批特征、同一折，分别用
  逻辑回归（表2 中宏观特征上的最优学习器） vs XGBoost（原时间对齐实验所用的学习器）

目的：检验论文核心对比"状态类积累、风格类不积累"是否依赖学习器选择。
输出：results_inplay/timealigned_learner.json + _facts/timealigned_learner.md
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import common as C

ROOT = str(ROOT)
OUT = str(R2)
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
POP = sorted([gid for gid, g in GAMES.items() if len(g["rallies"]) >= WMAX])

OOF = {}
p = os.path.join(OUT, "micro_oof_rally_probs.csv")
for r in pd.read_csv(p).itertuples():
    OOF[(int(r.game_id), int(r.rally_id))] = float(r.prob)
print("games=%d pop=%d oof=%d" % (len(GAMES), len(POP), len(OOF)), flush=True)

def feats_at(g, upto):
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

def micro_agg(gid, g, upto):
    ps = [OOF.get((gid, r["rally_id"]), np.nan) for r in g["rallies"][:upto]]
    ps = [x for x in ps if not np.isnan(x)]
    if not ps:
        return np.zeros(4, np.float32)
    return np.array([np.mean(ps), np.std(ps), np.max(ps), np.min(ps)], dtype=np.float32)

ARMS = {
    "state": lambda r: r["state"],
    "style": lambda r: r["style"],
    "micro": lambda r: r["micro"],
    "style+micro": lambda r: np.concatenate([r["style"], r["micro"]]),
    "state+style": lambda r: np.concatenate([r["state"], r["style"]]),
    "state+micro": lambda r: np.concatenate([r["state"], r["micro"]]),
    "all": lambda r: np.concatenate([r["state"], r["style"], r["micro"]]),
}
CN = {"state": "仅状态", "style": "仅风格", "micro": "仅微观", "style+micro": "风格+微观(原式融合)",
      "state+style": "状态+风格", "state+micro": "状态+微观", "all": "全部"}

def mk_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))

def mk_xgb():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             random_state=SEED, eval_metric="logloss", n_jobs=4)

res = {"note": "同一批183局、同一5折、同一特征；学习器对照 LR vs XGB",
       "n_games": len(POP), "windows": {}}

for w in WINDOWS:
    rows = []
    for gid in POP:
        g = GAMES[gid]
        st, sty = feats_at(g, w)
        rows.append({"match_id": g["match_id"], "label": g["label"],
                     "state": st, "style": sty, "micro": micro_agg(gid, g, w)})
    y = np.array([r["label"] for r in rows])
    grp = np.array([r["match_id"] for r in rows])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, grp))
    Xs = {a: np.array([f(r) for r in rows], np.float32) for a, f in ARMS.items()}

    entry = {"auc": {}, "folds": {}}
    for lname, mk in (("lr", mk_lr), ("xgb", mk_xgb)):
        for a in ARMS:
            X = Xs[a]
            fa = []
            for tr, te in folds:
                m = mk().fit(X[tr], y[tr])
                fa.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
            entry["auc"].setdefault(a, {})[lname] = float(np.mean(fa))
            entry["folds"].setdefault(a, {})[lname] = fa
    res["windows"][str(w)] = entry
    print("w=%-3d " % w + "  ".join("%s L%.3f/X%.3f" % (CN[a], entry["auc"][a]["lr"],
                                                         entry["auc"][a]["xgb"]) for a in ARMS), flush=True)

json.dump(res, open(os.path.join(OUT, "timealigned_learner.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------------- 报告
L = ["# 时间对齐七臂：学习器对照（逻辑回归 vs XGBoost）", "",
     "同一批 %d 局、同一 5 折 GroupKFold、同一批特征；唯一差别是学习器。" % len(POP), ""]
for metric, title in (("state", "状态类如何积累"), ("style", "风格类如何积累"),
                      ("micro", "微观类如何积累")):
    L += ["## %s（%s）" % (title, CN[metric]), "",
          "| 时点 | 逻辑回归 | XGBoost | 差 |", "|---|---|---|---|"]
    for w in WINDOWS:
        e = res["windows"][str(w)]["auc"][metric]
        L.append("| w=%d | %.4f | %.4f | %+.4f |" % (w, e["lr"], e["xgb"], e["lr"] - e["xgb"]))
    L.append("")

L += ["## 全部七臂（各时点的 LR / XGB）", "",
      "| 时点 | " + " | ".join(CN[a] for a in ARMS) + " |",
      "|" + "---|" * (len(ARMS) + 1)]
for w in WINDOWS:
    cells = ["%.3f / %.3f" % (res["windows"][str(w)]["auc"][a]["lr"],
                              res["windows"][str(w)]["auc"][a]["xgb"]) for a in ARMS]
    L.append("| w=%d | %s |" % (w, " | ".join(cells)))
L.append("")
L.append("（每格为 逻辑回归 / XGBoost）")

L += ["", "## 核心对比：状态 vs 风格（同一学习器下）", "",
      "| 时点 | LR 状态 | LR 风格 | LR 差 | XGB 状态 | XGB 风格 | XGB 差 |",
      "|---|---|---|---|---|---|---|"]
for w in WINDOWS:
    a = res["windows"][str(w)]["auc"]
    L.append("| w=%d | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |"
             % (w, a["state"]["lr"], a["style"]["lr"], a["state"]["lr"] - a["style"]["lr"],
                a["state"]["xgb"], a["style"]["xgb"], a["state"]["xgb"] - a["style"]["xgb"]))

L += ["", "## 融合是否带来增量（状态+风格 / 风格+微观 相对 仅状态）", "",
      "| 时点 | LR: 状态+风格−状态 | LR: 风格+微观−状态 | XGB: 状态+风格−状态 | XGB: 风格+微观−状态 |",
      "|---|---|---|---|---|"]
for w in WINDOWS:
    a = res["windows"][str(w)]["auc"]
    L.append("| w=%d | %+.4f | %+.4f | %+.4f | %+.4f |"
             % (w, a["state+style"]["lr"] - a["state"]["lr"],
                a["style+micro"]["lr"] - a["state"]["lr"],
                a["state+style"]["xgb"] - a["state"]["xgb"],
                a["style+micro"]["xgb"] - a["state"]["xgb"]))
print("\nwritten -> _facts/timealigned_learner.md")
