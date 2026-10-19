# -*- coding: utf-8 -*-
"""
Rebuild the dataset after removing the 8 duplicated matches, then re-split.

Fixes:
  F1 (data leakage): 8 matches appear twice (raw\\set and raw\\set22), 2 of them
      straddling train/test. Here we verify row-level identity of each pair,
      drop the higher match_id copy, and rebuild BLSR + macro features.
  E1: the split is regenerated and its real game counts are written out.

Outputs (all under D:\\data\\experiments_v2\\data\\):
  dataset_blsr_v2.csv            deduplicated shot-level data
  macro_features_v2.csv          18-dim macro features, regenerated
  train/val/test_match_ids_v2.txt
  dataset_report.json            all counts + the per-pair identity proof
"""
import pandas as pd, numpy as np, os, json, hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

SRC = str(DATA)
OUT = str(DATA)
os.makedirs(OUT, exist_ok=True)

DUP_PAIRS = [(3, 50), (5, 51), (6, 52), (10, 61), (13, 62), (32, 84), (33, 85), (35, 86)]
DROP = [b for a, b in DUP_PAIRS]

report = {"dup_pairs": DUP_PAIRS, "pair_identity": {}}

df = pd.read_csv(os.path.join(SRC, "dataset_blsr_all.csv"))

# ---------- 1. prove each pair is row-identical (all substantive columns) ----------
# game_id / rally_id / match_id are global counters and necessarily differ; every
# other column (features AND label) must match row-for-row.
ID_COLS = ["match_id", "game_id", "rally_id"]
cols = [c for c in df.columns if c not in ID_COLS]
for a, b in DUP_PAIRS:
    ga = df[df["match_id"] == a].reset_index(drop=True)
    gb = df[df["match_id"] == b].reset_index(drop=True)
    ga_k = ga.sort_values(["game_id", "rally_id", "time_proportion"]).reset_index(drop=True)
    gb_k = gb.sort_values(["game_id", "rally_id", "time_proportion"]).reset_index(drop=True)
    same_shape = ga_k.shape == gb_k.shape
    same_vals = bool(same_shape and (ga_k[cols] == gb_k[cols]).all().all())
    report["pair_identity"]["%d-%d" % (a, b)] = {
        "rows_a": int(len(ga)), "rows_b": int(len(gb)),
        "games_a": int(ga["game_id"].nunique()), "games_b": int(gb["game_id"].nunique()),
        "rallys_a": int(ga[["game_id", "rally_id"]].drop_duplicates().shape[0]),
        "identical_all_columns": same_vals,
    }
    if not same_vals:
        raise SystemExit("PAIR %d-%d NOT IDENTICAL -> refuse to auto-dedupe" % (a, b))
print("all 8 pairs proven row-identical on every column except match_id")

# ---------- 2. drop duplicates ----------
keep = df[~df["match_id"].isin(DROP)].copy()
report["raw"] = {"matches": int(df.match_id.nunique()), "games": int(df.game_id.nunique()),
                 "rallies": int(df[["game_id", "rally_id"]].drop_duplicates().shape[0])}
report["dedup"] = {"matches": int(keep.match_id.nunique()), "games": int(keep.game_id.nunique()),
                   "rallies": int(keep[["game_id", "rally_id"]].drop_duplicates().shape[0])}
keep.to_csv(os.path.join(OUT, "dataset_blsr_v2.csv"), index=False)
print("dedup -> matches=%d games=%d rallies=%d" % (
    report["dedup"]["matches"], report["dedup"]["games"], report["dedup"]["rallies"]))

# ---------- 3. regenerate macro features (same recipe as generate_macro_from_blsr.py) ----------
all_types = sorted(keep["type"].dropna().unique())
rows = []
for game_id, g in keep.groupby("game_id"):
    s, r = g[g["is_target_turn"] == 1], g[g["is_target_turn"] == 0]
    sc, rc = s["type"].value_counts(), r["type"].value_counts()
    feats = [sc.get(t, 0) / len(s) - rc.get(t, 0) / len(r) for t in all_types]
    rows.append([game_id, g["match_id"].iloc[0]] + feats + [g["is_target_win"].iloc[-1]])
macro = pd.DataFrame(rows, columns=["game_id", "match_id"] + all_types + ["label"])
macro.to_csv(os.path.join(OUT, "macro_features_v2.csv"), index=False, encoding="utf-8-sig")
report["macro"] = {"games": int(len(macro)), "features": len(all_types),
                   "label_1": int((macro["label"] == 1).sum()), "label_0": int((macro["label"] == 0).sum())}
print("macro features -> %d games x %d features (pos %d / neg %d)" % (
    len(macro), len(all_types), report["macro"]["label_1"], report["macro"]["label_0"]))

# ---------- 4. new split: test 30% of matches, then val 20% of the remainder ----------
matches = np.unique(macro["match_id"].values)
prop = macro.groupby("match_id")["label"].mean()
binl = (prop >= 0.5).astype(int)
tr_all, te = train_test_split(matches, test_size=0.3, random_state=42, stratify=binl.loc[matches])
tr, va = train_test_split(tr_all, test_size=0.20, random_state=42, stratify=binl.loc[tr_all])
tr, va, te = (set(int(x) for x in v) for v in (tr, va, te))
assert not (tr & va) and not (tr & te) and not (va & te)
for name, s in (("train", tr), ("val", va), ("test", te)):
    np.savetxt(os.path.join(OUT, "%s_match_ids_v2.txt" % name), np.array(sorted(s)), fmt="%d")

def gcount(s, col="game_id"):
    return int(macro[macro["match_id"].isin(s)][col].nunique())

report["split"] = {n: {"matches": len(s), "games": gcount(s),
                       "label_1": int((macro[macro["match_id"].isin(s)]["label"] == 1).sum()),
                       "label_0": int((macro[macro["match_id"].isin(s)]["label"] == 0).sum())}
                   for n, s in (("train", tr), ("val", va), ("test", te))}
print("split -> train %s | val %s | test %s" % (
    report["split"]["train"], report["split"]["val"], report["split"]["test"]))

# leakage re-check on the new split
shot_g2m = keep.drop_duplicates(["game_id", "match_id"]).set_index("game_id")["match_id"]
assert set(shot_g2m[shot_g2m.isin(tr)].index) & set(shot_g2m[shot_g2m.isin(te)].index) == set(), "leak!"

# ---------- 5. macro baselines on the new split (updated Table 1) ----------
fcols = [c for c in macro.columns if c not in ("game_id", "match_id", "label")]
mid = macro["match_id"].values
mask_tr = np.isin(mid, list(tr)); mask_te = np.isin(mid, list(te))
X, y = macro[fcols].values, macro["label"].values
rf = RandomForestClassifier(n_estimators=600, max_depth=8, random_state=42, n_jobs=-1).fit(X[mask_tr], y[mask_tr])
xg = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42,
                       eval_metric="logloss", use_label_encoder=False).fit(X[mask_tr], y[mask_tr])
report["macro_baseline_new_split"] = {
    "RF_test_AUC": float(roc_auc_score(y[mask_te], rf.predict_proba(X[mask_te])[:, 1])),
    "XGB_test_AUC": float(roc_auc_score(y[mask_te], xg.predict_proba(X[mask_te])[:, 1])),
    "train_games": int(mask_tr.sum()), "test_games": int(mask_te.sum()),
}
print("new-split macro baseline: RF %.4f | XGB %.4f" % (
    report["macro_baseline_new_split"]["RF_test_AUC"], report["macro_baseline_new_split"]["XGB_test_AUC"]))

with open(os.path.join(OUT, "dataset_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nreport ->", os.path.join(OUT, "dataset_report.json"))
