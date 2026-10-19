# -*- coding: utf-8 -*-
"""
experiments_v2 / run_fusion.py
==============================
Two-stage fusion: a macro XGBoost model (18 technical-frequency differences per
game) fused with the micro CNN+BiGRU+attention rally probabilities.

6 methods x 2 conditions
------------------------
  methods : wa, rf_concat, xgb_concat, stacking_lr, stacking_rf, stacking_xgb
  conds   : fusion (with location) and fusion_no_location

Protocol (the core v2 fix)
--------------------------
The original train_fusion.py fitted the base micro model on the training split
and then used its *in-sample* training predictions (plus an in-sample macro
prediction) to fit the meta-model -- the meta-model therefore saw over-fitted,
leakage-contaminated inputs, and the macro/微 split was regenerated inside the
script with train_test_split.

Here every base-model prediction handed to a meta-model is **out-of-fold**:

  * 5-fold ``GroupKFold`` grouped by ``match_id`` over the training split only;
  * each fold's base model is early-stopped on an inner match-level holdout of
    that fold (the outer validation split stays untouched);
  * the micro per-rally probabilities are turned into per-game statistics by
    aggregating the OOF predictions;
  * the macro XGBoost probabilities come from the *same* GroupKFold;
  * alpha (weighted average) and all meta-models are fitted on the OOF training
    games only; the test split is read exactly once, at the very end.

env / CLI
---------
  N_RUNS (default 5)  EPOCHS (default 100)  V2_FUSION_FOLDS (default 5)
  python run_fusion.py [--conditions fusion] [--n-runs 1] [--epochs 2]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import common as C

METHODS = ["wa", "rf_concat", "xgb_concat", "lr_concat",
           "stacking_lr", "stacking_rf", "stacking_xgb"]
CONDITIONS = ["fusion_lrmacro", "fusion_lrmacro_no_location"]
ORIGIN = {
    "fusion_lrmacro": "train_fusion.py recipe; macro branch = LogisticRegression; base micro = BiLSTM",
    "fusion_lrmacro_no_location": "same as fusion_lrmacro, drop_location=True",
}
BATCH = 32
PATIENCE = 15
MICRO_FEATS = ["micro_mean", "micro_std", "micro_max", "micro_min"]

# --- hyper-parameters copied verbatim from train_fusion.py -------------------
XGB_MACRO = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                 random_state=42, eval_metric="logloss", use_label_encoder=False)
RF_CONCAT = dict(n_estimators=600, max_depth=8, random_state=42)
XGB_CONCAT = dict(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
LR_META = dict()
RF_META = dict(n_estimators=100, random_state=42)
XGB_META = dict(n_estimators=100, max_depth=4, random_state=42)
ALPHAS = np.arange(0, 1.05, 0.05)

def macro_probs_oof(X, y, groups, seed: int, n_folds: int):
    """OOF macro probabilities + full-train model for the test split."""
    oof = np.full(len(y), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_folds)
    for tr_idx, ho_idx in gkf.split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=seed))
        m.fit(X[tr_idx], y[tr_idx])
        oof[ho_idx] = m.predict_proba(X[ho_idx])[:, 1]
    full = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, random_state=seed))
    full.fit(X, y)
    return oof, full

def micro_probs_oof(seqs_tr, factory, seed: int, n_folds: int, epochs: int,
                    verbose: int = 0):
    """OOF per-rally probabilities over the training split (GroupKFold on match_id).

    Returns (oof_rally_df, oof_game_stats, wall_seconds).
    """
    match = np.array([s["match_id"] for s in seqs_tr])
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full(len(seqs_tr), np.nan, dtype=float)
    t0 = time.time()
    for k, (tr_idx, ho_idx) in enumerate(gkf.split(np.zeros(len(seqs_tr)), groups=match)):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        try:
            sub_tr, sub_va = next(inner.split(np.zeros(len(tr_idx)),
                                              groups=match[tr_idx]))
        except ValueError:                       # too few matches: fall back
            sub_tr = np.arange(len(tr_idx))
            sub_va = np.arange(len(tr_idx))
        fold_tr = [seqs_tr[i] for i in tr_idx[sub_tr]]
        fold_va = [seqs_tr[i] for i in tr_idx[sub_va]]
        C.set_seed(seed * 100 + k)
        model = C.compile_model(factory())
        C.train_with_early_stopping(
            model,
            C.make_micro_dataset(fold_tr, BATCH, shuffle=True, seed=seed * 100 + k),
            C.make_micro_dataset(fold_va, BATCH, shuffle=False),
            epochs=epochs, patience=PATIENCE, verbose=verbose,
            steps_per_epoch=C.steps_for(fold_tr, BATCH),
            validation_steps=C.steps_for(fold_va, BATCH))
        ho = [seqs_tr[i] for i in ho_idx]
        oof[ho_idx] = C.predict_micro(model, ho, BATCH)
        print("    fold %d/%d: base on %d rallies (inner val %d) -> predict %d rallies"
              % (k + 1, n_folds, len(fold_tr), len(fold_va), len(ho)), flush=True)
    wall = time.time() - t0
    if np.isnan(oof).any():
        raise RuntimeError("OOF micro predictions incomplete")

    df = pd.DataFrame({
        "game_id": [s["game_id"] for s in seqs_tr],
        "match_id": [s["match_id"] for s in seqs_tr],
        "label": [s["label"] for s in seqs_tr],
        "prob": oof,
    })
    stats = df.groupby("game_id")["prob"].agg(micro_mean="mean", micro_std="std",
                                             micro_max="max", micro_min="min").reset_index()
    stats["micro_std"] = stats["micro_std"].fillna(0.0)
    return df, stats, wall

def micro_game_stats(seqs, probs) -> pd.DataFrame:
    df = pd.DataFrame({"game_id": [s["game_id"] for s in seqs],
                       "prob": probs})
    stats = df.groupby("game_id")["prob"].agg(micro_mean="mean", micro_std="std",
                                             micro_max="max", micro_min="min").reset_index()
    stats["micro_std"] = stats["micro_std"].fillna(0.0)
    return stats

def fusion_methods(X_tr, y_tr, X_te, y_te, seed: int, want: list):
    """Fit each meta-model on OOF training data, score once on test."""
    out = {}
    if "wa" in want:
        macro_tr, micro_tr = X_tr[:, 0], X_tr[:, 1]
        best_alpha, best_auc = 0.0, -np.inf
        for alpha in ALPHAS:
            auc = roc_auc_score(y_tr, alpha * macro_tr + (1 - alpha) * micro_tr)
            if auc > best_auc:
                best_auc, best_alpha = auc, float(alpha)
        p_te = best_alpha * X_te[:, 0] + (1 - best_alpha) * X_te[:, 1]
        out["wa"] = dict(alpha=best_alpha, oof_auc=float(best_auc),
                         test_auc=float(roc_auc_score(y_te, p_te)))
    if "rf_concat" in want:
        m = RandomForestClassifier(**{**RF_CONCAT, "random_state": seed})
        m.fit(X_tr, y_tr)
        out["rf_concat"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])))
    if "xgb_concat" in want:
        m = xgb.XGBClassifier(**{**XGB_CONCAT, "random_state": seed})
        m.fit(X_tr, y_tr)
        out["xgb_concat"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])))
    if "lr_concat" in want:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=seed))
        m.fit(X_tr, y_tr)
        out["lr_concat"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])))
    if "stacking_lr" in want:
        m = LogisticRegression(**LR_META)
        m.fit(X_tr[:, :2], y_tr)
        out["stacking_lr"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te[:, :2])[:, 1])))
    if "stacking_rf" in want:
        m = RandomForestClassifier(**{**RF_META, "random_state": seed})
        m.fit(X_tr[:, :2], y_tr)
        out["stacking_rf"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te[:, :2])[:, 1])))
    if "stacking_xgb" in want:
        m = xgb.XGBClassifier(**{**XGB_META, "random_state": seed})
        m.fit(X_tr, y_tr)
        out["stacking_xgb"] = dict(test_auc=float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])))
    return out

def run_condition(condition: str, n_runs: int, epochs: int, n_folds: int,
                  tag: str | None) -> dict:
    drop_loc = condition.endswith("_no_location")
    base_arch = "cnn" if "_cnn" in condition else "bilstm"
    print("\n" + "=" * 74)
    print("CONDITION: %s   (%s)" % (condition, ORIGIN[condition]))
    print("base micro model: %s | drop_location=%s" % (base_arch, drop_loc))
    print("=" * 74)

    ids = C.load_split_ids()
    macro, macro_cols = C.load_macro()
    macro_train = macro[macro["match_id"].isin(ids["train"])]
    macro_val = macro[macro["match_id"].isin(ids["val"])]
    macro_test = macro[macro["match_id"].isin(ids["test"])]
    Xm_tr = macro_train[macro_cols].to_numpy(np.float32)
    ym_tr = macro_train["label"].to_numpy(float)
    gm_tr = macro_train["match_id"].to_numpy()
    Xm_te = macro_test[macro_cols].to_numpy(np.float32)
    ym_te = macro_test["label"].to_numpy(float)
    print("games: train %d / val %d / test %d | macro dim %d"
          % (len(macro_train), len(macro_val), len(macro_test), len(macro_cols)))

    df, meta = C.load_blsr(C.BLSR_V2, drop_location=drop_loc)
    seqs_tr = C.build_rally_sequences(df, ids["train"])
    seqs_va = C.build_rally_sequences(df, ids["val"])
    seqs_te = C.build_rally_sequences(df, ids["test"])
    if base_arch == "bilstm":
        factory = lambda: C.build_bilstm(meta["num_types"])  # noqa: E731
    else:
        factory = lambda: C.build_cnn_gru_attention(  # noqa: E731
            meta["num_types"], meta["num_areas"])
    print("rallies: train %d / val %d / test %d | num_types=%d num_areas=%d"
          % (len(seqs_tr), len(seqs_va), len(seqs_te), meta["num_types"], meta["num_areas"]))
    print("GroupKFold folds = %d (grouped by match_id, training split only)" % n_folds)

    results = []
    for run in range(n_runs):
        seed = 42 + run
        print("\n--- %s | run %d/%d | seed %d ---" % (condition, run + 1, n_runs, seed))
        t0 = time.time()
        C.set_seed(seed)

        # 1) macro OOF (train split) + final macro model (train -> test)
        oof_macro, macro_full = macro_probs_oof(Xm_tr, ym_tr, gm_tr, seed, n_folds)
        macro_oof_auc = C.safe_auc(ym_tr, oof_macro)
        macro_test_auc = C.safe_auc(ym_te, macro_full.predict_proba(Xm_te)[:, 1])
        print("  macro LR: OOF(train) AUC=%.4f | test AUC=%.4f"
              % (macro_oof_auc, macro_test_auc))
        macro_oof_df = pd.DataFrame({"game_id": macro_train["game_id"].to_numpy(),
                                     "macro_prob": oof_macro})
        macro_te_df = pd.DataFrame({
            "game_id": macro_test["game_id"].to_numpy(),
            "macro_prob": macro_full.predict_proba(Xm_te)[:, 1]})

        # 2) micro OOF (train split)
        oof_rally, micro_oof_stats, oof_wall = micro_probs_oof(
            seqs_tr, factory, seed, n_folds, epochs)

        # 3) final base model: train split, early-stopped on the *outer* val split
        print("  final base model (train split, early stop on outer val split)...", flush=True)
        tb = time.time()
        C.set_seed(seed)
        base = C.compile_model(factory())
        fit = C.train_with_early_stopping(
            base,
            C.make_micro_dataset(seqs_tr, BATCH, shuffle=True, seed=seed),
            C.make_micro_dataset(seqs_va, BATCH, shuffle=False),
            epochs=epochs, patience=PATIENCE, verbose=2,
            steps_per_epoch=C.steps_for(seqs_tr, BATCH),
            validation_steps=C.steps_for(seqs_va, BATCH))
        base_wall = time.time() - tb
        p_tr = C.predict_micro(base, seqs_tr, BATCH)          # in-sample, report only
        p_va = C.predict_micro(base, seqs_va, BATCH)
        p_te = C.predict_micro(base, seqs_te, BATCH)
        base_train_auc = C.safe_auc([s["label"] for s in seqs_tr], p_tr)
        base_val_auc = C.safe_auc([s["label"] for s in seqs_va], p_va)
        base_test_auc = C.safe_auc([s["label"] for s in seqs_te], p_te)
        print("  base micro: train(in-sample)=%.4f val=%.4f test=%.4f (test read once)"
              % (base_train_auc, base_val_auc, base_test_auc))

        # 4) build OOF fusion matrices
        tr_fusion = (macro_oof_df
                     .merge(micro_oof_stats, on="game_id")
                     .merge(macro_train[["game_id", "label"]], on="game_id"))
        te_stats = micro_game_stats(seqs_te, p_te)
        va_stats = micro_game_stats(seqs_va, p_va)
        te_fusion = (macro_te_df
                     .merge(te_stats, on="game_id")
                     .merge(macro_test[["game_id", "label"]], on="game_id"))
        cols = ["macro_prob"] + MICRO_FEATS
        X_meta = tr_fusion[cols].to_numpy(float)
        y_meta = tr_fusion["label"].to_numpy(float)
        X_te = te_fusion[cols].to_numpy(float)
        y_te = te_fusion["label"].to_numpy(float)
        assert len(X_meta) == len(macro_train), "OOF fusion matrix must cover all train games"
        assert not np.isnan(X_meta).any(), "NaN in OOF features"

        # diagnostic: what the *original* in-sample protocol would have scored on
        # the same training games (kept only to quantify the optimistic bias)
        leaked = pd.DataFrame({"game_id": [s["game_id"] for s in seqs_tr], "prob": p_tr})
        leaked_stats = leaked.groupby("game_id")["prob"].agg(
            micro_mean="mean", micro_std="std", micro_max="max", micro_min="min").reset_index()
        insample = (macro_oof_df.merge(leaked_stats, on="game_id")
                    .merge(macro_train[["game_id", "label"]], on="game_id"))
        y_in = insample["label"].to_numpy(float)
        insample_auc = float(roc_auc_score(
            y_in, 0.5 * insample["macro_prob"].to_numpy(float)
            + 0.5 * insample["micro_mean"].to_numpy(float)))
        oof_auc_same = float(roc_auc_score(y_meta, 0.5 * X_meta[:, 0] + 0.5 * X_meta[:, 1]))

        # 4b) 保存概率，便于后续统计复算（无需重训）
        _pdir = os.path.join(C.RESULTS_DIR, "probs")
        os.makedirs(_pdir, exist_ok=True)
        np.savez_compressed(
            os.path.join(_pdir, "%s__run%d.npz" % (condition, run + 1)),
            macro_oof=oof_macro, y_meta=y_meta,
            macro_test=macro_full.predict_proba(Xm_te)[:, 1],
            micro_test=p_te, y_te=y_te,
            micro_mean_te=X_te[:, 1], micro_mean_tr=X_meta[:, 1])

        # 5) meta-models on OOF, scored once on test
        meth = fusion_methods(X_meta, y_meta, X_te, y_te, seed, METHODS)

        key_cols = dict(tr_fusion_games=len(tr_fusion), test_games=len(te_fusion),
                        X_meta_cols=cols)
        rec = C.base_record(condition, run, seed, epochs, C.BLSR_V2, extra={
            "origin_script": ORIGIN[condition],
            "n_runs": n_runs,
            "n_folds": n_folds,
            "protocol": "5-fold GroupKFold(match_id) OOF base predictions; "
                        "alpha and meta-models fitted on OOF train only",
            # generic schema fields (collect.py reads these for every record).
            # "train_auc" is the base micro model's in-sample train AUC and
            # "test_auc" is the base micro model's single test evaluation --
            # they describe the BASE model, not any fused method.  The fused
            # results live in method_<name> below.
            "train_auc": base_train_auc,
            "val_auc_best": fit["val_auc_best"],
            "test_auc": base_test_auc,
            "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"],
            "base_micro_train_auc_insample": base_train_auc,
            "base_micro_val_auc": base_val_auc,
            "base_micro_test_auc": base_test_auc,
            "base_micro_best_epoch": fit["best_epoch"],
            "base_micro_epochs_run": fit["epochs_run"],
            "macro_oof_auc_train": macro_oof_auc,
            "macro_test_auc": macro_test_auc,
            "diag_insample_wa_train_auc": insample_auc,
            "diag_oof_wa_train_auc": oof_auc_same,
            "diag_optimism_gap": insample_auc - oof_auc_same,
            "meta_features": cols,
            **key_cols,
            **{k: v["test_auc"] for k, v in meth.items()},
            "fusion_detail": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items()} for k, v in meth.items()},
            "wall_seconds": round(time.time() - t0, 2),
            "oof_seconds": round(oof_wall, 2),
            "final_base_seconds": round(base_wall, 2),
            "seconds_per_epoch": round((oof_wall + base_wall)
                                       / max(1, (n_folds + 1) * fit["epochs_run"]), 3),
        })
        for m in METHODS:
            rec["method_%s" % m] = meth[m]["test_auc"]
        json_path = C.save_record(rec, tag)
        npz_path = C.save_curves(fit, "%s__base_cnn" % condition, run, tag)
        print("  methods: " + ", ".join("%s=%.4f" % (m, meth[m]["test_auc"]) for m in METHODS))
        # NOTE: no per-run "best method" selection is printed or stored -- picking a
        # method by its test score would be selective reporting.  Method comparison
        # is done in collect.py on the mean over the N_RUNS runs.
        print("RESULT %s run%d: %.1fs (all 6 method test AUCs listed above)"
              % (condition, run + 1, rec["wall_seconds"]))
        print("saved: %s | %s" % (os.path.basename(json_path), os.path.basename(npz_path)))
        results.append(rec)

    print("\nSUMMARY %s:" % condition)
    for m in METHODS:
        vals = np.array([r["method_%s" % m] for r in results], dtype=float)
        print("  %-14s test AUC %.4f +/- %.4f  (per run %s)"
              % (m, vals.mean(), vals.std(), [round(x, 4) for x in vals]))
    return {"condition": condition,
            "methods": {m: [r["method_%s" % m] for r in results] for m in METHODS}}

def main():
    ap = C.cli_parser(__doc__)
    args = ap.parse_args()
    conds = C.resolve_conditions(CONDITIONS, args.conditions)
    if args.list:
        print("\n".join(CONDITIONS))
        return
    cfg = C.config(args.n_runs, args.epochs)
    n_folds = C.env_int("V2_FUSION_FOLDS", 5)
    C.ensure_dirs()
    tee = C.start_log("run_fusion%s" % (("_" + args.tag) if args.tag else ""))
    try:
        print("run_fusion.py | conditions=%s | N_RUNS=%d EPOCHS=%d folds=%d"
              % (conds, cfg["n_runs"], cfg["epochs"], n_folds))
        summary = [run_condition(c, cfg["n_runs"], cfg["epochs"], n_folds, args.tag)
                   for c in conds]
        print("\n" + "=" * 74)
        for s in summary:
            print(s["condition"])
            for m in METHODS:
                vals = np.array(s["methods"][m], dtype=float)
                print("  %-14s %.4f +/- %.4f" % (m, vals.mean(), vals.std()))
        print("=" * 74)
    finally:
        C.stop_log(tee)

if __name__ == "__main__":
    main()
