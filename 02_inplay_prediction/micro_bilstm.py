# -*- coding: utf-8 -*-
"""
experiments_v2 / run_micro.py
=============================
Five micro (rally-level) conditions, each trained ``N_RUNS`` times with an
independent seed, early-stopped on the validation split, and evaluated on the
test split exactly once.

conditions
----------
  cnn              CNN + BiGRU + attention          (微观模型/CNN.py 115-169)
  bilstm           BiLSTM                           (微观模型/BiLSTM.py 84-96)
  transformer      Transformer                      (微观模型/Transformer.py 74-115)
  cnn_no_location  same CNN, 3 location cols zeroed (消融实验/CNN_no_location.py)
  cnn_lindan       same CNN on the private dataset  (消融实验/CNN_lindan.py)

env / CLI
---------
  N_RUNS (default 5)   EPOCHS (default 100)
  python run_micro.py [--conditions cnn,bilstm] [--n-runs 1] [--epochs 2]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os
import sys
import time

import numpy as np
import pandas as pd

import common as C

CONDITIONS = ["cnn", "bilstm", "transformer", "cnn_no_location", "cnn_lindan",
              "bilstm_no_location", "bilstm_lindan"]
ORIGIN = {
    "cnn": r"D:\data\微观模型\CNN.py (lines 115-169)",
    "bilstm": r"D:\data\微观模型\BiLSTM.py (lines 84-96)",
    "transformer": r"D:\data\微观模型\Transformer.py (lines 74-115)",
    "cnn_no_location": r"D:\data\消融实验\CNN_no_location.py (location cols = 0)",
    "cnn_lindan": r"D:\data\消融实验\CNN_lindan.py (private data)",
    "bilstm_no_location": r"BiLSTM.py + location cols = 0 (ablation with the BEST micro model)",
    "bilstm_lindan": r"BiLSTM.py on private LinDan data (cross-dataset check)",
}
BATCH = 32
PATIENCE = 15

def build_private_split(df: pd.DataFrame, seed: int = 42):
    """3-way match-level split for the private dataset.

    v2 change: CNN_lindan.py used a 2-way ``train_test_split(test_size=0.3)``
    with no validation set.  We keep the same 70 % training share and the same
    stratified-by-match-mean-label recipe, and carve a 15 % validation set out
    of the remainder so EarlyStopping works (stratified, seed-fixed).
    """
    from sklearn.model_selection import train_test_split
    matches = np.unique(df["match_id"].values)
    prop = df.groupby("match_id")["is_target_win"].mean()
    binl = (prop >= 0.5).astype(int).loc[matches]
    tr, rest = train_test_split(matches, test_size=0.30, random_state=seed,
                                stratify=binl)
    if rest.size >= 4 and binl.loc[rest].nunique() > 1:
        va, te = train_test_split(rest, test_size=0.5, random_state=seed,
                                  stratify=binl.loc[rest])
    else:
        va, te = rest[:max(1, len(rest) // 2)], rest[max(1, len(rest) // 2):]
    return ([int(x) for x in tr], [int(x) for x in va], [int(x) for x in te])

def prepare(condition: str):
    """Returns (train/val/test sequences, model factory, data path, meta).

    Condition names are `<arch>[_no_location|_lindan]`, so the ablation and the
    cross-dataset check can be run with ANY architecture (e.g. bilstm_no_location
    uses the current best micro model instead of the CNN).
    """
    is_lindan = condition.endswith("_lindan")
    is_noloc = condition.endswith("_no_location")
    arch = condition.replace("_no_location", "").replace("_lindan", "")

    if is_lindan:
        df, meta = C.load_blsr(C.LINDAN_BLSR, drop_location=False)
        tr_ids, va_ids, te_ids = build_private_split(df, seed=42)
        meta["split_source"] = "v2 stratified 70/15/15 over match_id (seed 42)"
    else:
        df, meta = C.load_blsr(C.BLSR_V2, drop_location=is_noloc)
        ids = C.load_split_ids()
        tr_ids, va_ids, te_ids = ids["train"], ids["val"], ids["test"]
        meta["split_source"] = "shared v2 split (train/val/test_match_ids_v2.txt)"

    seqs = (C.build_rally_sequences(df, tr_ids),
            C.build_rally_sequences(df, va_ids),
            C.build_rally_sequences(df, te_ids))

    if arch == "cnn":
        factory = lambda: C.build_cnn_gru_attention(  # noqa: E731
            meta["num_types"], meta["num_areas"])
    elif arch == "bilstm":
        factory = lambda: C.build_bilstm(meta["num_types"])  # noqa: E731
    elif arch == "transformer":
        factory = lambda: C.build_transformer(meta["num_types"])  # noqa: E731
    else:
        raise ValueError(condition)

    data_path = C.LINDAN_BLSR if is_lindan else C.BLSR_V2
    meta["split"] = {"train_matches": len(tr_ids), "val_matches": len(va_ids),
                     "test_matches": len(te_ids),
                     "train_rallies": len(seqs[0]), "val_rallies": len(seqs[1]),
                     "test_rallies": len(seqs[2])}
    return seqs, factory, data_path, meta

def run_condition(condition: str, n_runs: int, epochs: int, tag: str | None) -> dict:
    print("\n" + "=" * 74)
    print("CONDITION: %s   (%s)" % (condition, ORIGIN[condition]))
    print("=" * 74)
    (tr, va, te), factory, data_path, meta = prepare(condition)
    print("num_types=%d num_areas=%d | %s" % (meta["num_types"], meta["num_areas"],
                                              meta["split_source"]))
    print("split: %s" % meta["split"])

    y_test = np.array([s["label"] for s in te])
    test_game = np.array([s["game_id"] for s in te])
    results = []

    for run in range(n_runs):
        seed = 42 + run
        C.set_seed(seed)
        print("\n--- %s | run %d/%d | seed %d | epochs<=%d ---"
              % (condition, run + 1, n_runs, seed, epochs))
        model = C.compile_model(factory())
        if run == 0:
            print("params=%d" % model.count_params(), flush=True)
        t0 = time.time()
        fit = C.train_with_early_stopping(
            model,
            C.make_micro_dataset(tr, BATCH, shuffle=True, seed=seed),
            C.make_micro_dataset(va, BATCH, shuffle=False),
            epochs=epochs, patience=PATIENCE, verbose=2,
            steps_per_epoch=C.steps_for(tr, BATCH),
            validation_steps=C.steps_for(va, BATCH))
        train_seconds = time.time() - t0

        # ---- the one and only test evaluation -----------------------------
        t1 = time.time()
        p_te = C.predict_micro(model, te, BATCH)
        test_auc = C.safe_auc(y_test, p_te)
        test_seconds = time.time() - t1
        # extra paper-friendly metric: mean within-game (per-game) AUC
        per_game = [C.safe_auc(y_test[test_game == g], p_te[test_game == g])
                    for g in np.unique(test_game)]
        per_game = [x for x in per_game if not np.isnan(x)]

        rec = C.base_record(condition, run, seed, epochs, data_path, extra={
            "origin_script": ORIGIN[condition],
            "n_runs": n_runs,
            "batch_size": BATCH,
            "patience": PATIENCE,
            "num_types": int(meta["num_types"]),
            "num_areas": int(meta["num_areas"]),
            "params": int(model.count_params()),
            "split_source": meta["split_source"],
            "split": meta["split"],
            "train_auc": fit["train_auc_last"],
            "val_auc_best": fit["val_auc_best"],
            "test_auc": test_auc,
            "test_auc_per_game_mean": float(np.mean(per_game)) if per_game else None,
            "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"],
            "wall_seconds": round(train_seconds + test_seconds, 2),
            "train_seconds": round(train_seconds, 2),
            "test_predict_seconds": round(test_seconds, 2),
            "seconds_per_epoch": round(train_seconds / max(1, fit["epochs_run"]), 3),
        })
        json_path = C.save_record(rec, tag)
        npz_path = C.save_curves(fit, condition, run, tag)
        print("RESULT %s run%d: train_auc=%.4f val_auc_best=%.4f test_auc=%.4f "
              "(best_epoch=%d, epochs_run=%d, %.1fs, %.2fs/epoch)"
              % (condition, run + 1, rec["train_auc"], rec["val_auc_best"],
                 rec["test_auc"], rec["best_epoch"], rec["epochs_run"],
                 rec["wall_seconds"], rec["seconds_per_epoch"]))
        print("saved: %s | %s" % (os.path.basename(json_path), os.path.basename(npz_path)))
        results.append(rec)

    te_aucs = [r["test_auc"] for r in results]
    print("\nSUMMARY %s: test AUC = %.4f +/- %.4f   (per run: %s)"
          % (condition, float(np.mean(te_aucs)), float(np.std(te_aucs)),
             [round(x, 4) for x in te_aucs]))
    return {"condition": condition, "test_aucs": te_aucs}

def main():
    ap = C.cli_parser(__doc__)
    args = ap.parse_args()
    conds = C.resolve_conditions(CONDITIONS, args.conditions)
    if args.list:
        print("\n".join(CONDITIONS))
        return
    cfg = C.config(args.n_runs, args.epochs)
    tag = args.tag
    C.ensure_dirs()
    tee = C.start_log("run_micro%s" % (("_" + tag) if tag else ""))
    try:
        print("run_micro.py | conditions=%s | N_RUNS=%d EPOCHS=%d | tag=%s"
              % (conds, cfg["n_runs"], cfg["epochs"], tag))
        print("cwd=%s (paths are absolute, CWD-independent)" % os.getcwd())
        summary = [run_condition(c, cfg["n_runs"], cfg["epochs"], tag) for c in conds]
        print("\n" + "=" * 74)
        for s in summary:
            vals = np.array(s["test_aucs"], dtype=float)
            print("%-16s test AUC %.4f +/- %.4f" % (s["condition"], vals.mean(), vals.std()))
        print("=" * 74)
    finally:
        C.stop_log(tee)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Note on conditions
# ---------------------------------------------------------------------------
# The `_lindan` entries in CONDITIONS / ORIGIN refer to an optional
# cross-dataset check that used data not included in this release
# (common.LINDAN_BLSR is None, so those branches cannot execute).
# They are retained for completeness only. The conditions analysed in the
# manuscript are: cnn, bilstm, transformer, cnn_no_location, bilstm_no_location.
