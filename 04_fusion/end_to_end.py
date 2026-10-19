# -*- coding: utf-8 -*-
"""
experiments_v2 / run_e2e.py
===========================
End-to-end (game-level) models: the whole game shot sequence goes into a
recurrent / attention encoder and the game label is predicted directly.

6 conditions:  {gru, lstm, transformer} x {pure, +macro}

Faithfulness notes
------------------
* The three spatial columns (hit_area / player_location_area /
  opponent_location_area) are zeroed in *both* variants, exactly as in the
  original scripts (the paper describes end-to-end as location-free).
* Architecture, layer sizes, dropout, L2, batch size 8 and the infinite
  generator + steps_per_epoch design are copied from
  D:\\data\\端到端实验\\train_{GRU,LSTM,Transformer}_model.py and
  train_{GRU,LSTM,Transformer}_macro_model.py.

v2 protocol changes: 3-way train/val/test split, val-based EarlyStopping
(patience 15), test evaluated exactly once, per-run seed 42+run, per-run JSON +
condition-suffixed npz curve file.

env / CLI
---------
  N_RUNS (default 5)  EPOCHS (default 100)
  V2_E2E_RECURRENT_DROPOUT (default 1 -> same as the original 0.5/0.5)
  python run_e2e.py [--conditions gru_pure,lstm_macro] [--n-runs 1] [--epochs 2]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os
import time

import numpy as np
import tensorflow as tf

import common as C

ARCHS = ["gru", "lstm", "transformer"]
CONDITIONS = ["%s_%s" % (a, s) for a in ARCHS for s in ("pure", "macro")]
ORIGIN = {
    "gru_pure": "GRU, stroke sequence only",
    "gru_macro": "GRU, sequence + whole-game style features",
    "lstm_pure": "LSTM, stroke sequence only",
    "lstm_macro": "LSTM, sequence + whole-game style features",
    "transformer_pure": "Transformer, stroke sequence only",
    "transformer_macro": "Transformer, sequence + whole-game style features",
}
BATCH = 8
PATIENCE = 15

def split_condition(condition: str):
    arch, variant = condition.rsplit("_", 1)
    if arch not in ARCHS or variant not in ("pure", "macro"):
        raise ValueError(condition)
    return arch, variant

def build(arch: str, macro_dim):
    if arch == "gru":
        return C.build_e2e_gru(macro_dim)
    if arch == "lstm":
        return C.build_e2e_lstm(macro_dim)
    if arch == "transformer":
        return C.build_e2e_transformer(macro_dim)
    raise ValueError(arch)

def run_condition(condition: str, n_runs: int, epochs: int, tag: str | None) -> dict:
    arch, variant = split_condition(condition)
    print("\n" + "=" * 74)
    print("CONDITION: %s   (%s)" % (condition, ORIGIN[condition]))
    print("=" * 74)

    ids = C.load_split_ids()
    macro_df, macro_cols = C.load_macro()
    macro_dim = len(macro_cols)
    macro_map = {int(g): np.asarray(row, dtype=np.float32)
                 for g, row in zip(macro_df["game_id"].values,
                                   macro_df[macro_cols].to_numpy(np.float32))}

    # spatial columns zeroed for *both* variants (paper + original scripts)
    df, _ = C.load_blsr(C.BLSR_V2, drop_location=True)
    tr = C.build_game_sequences(df, ids["train"])
    va = C.build_game_sequences(df, ids["val"])
    te = C.build_game_sequences(df, ids["test"])
    if variant == "macro":
        keys = set(macro_map)
        tr = [s for s in tr if s["game_id"] in keys]
        va = [s for s in va if s["game_id"] in keys]
        te = [s for s in te if s["game_id"] in keys]
    print("games: train %d / val %d / test %d | macro_dim=%d | variant=%s"
          % (len(tr), len(va), len(te), macro_dim, variant))
    print("location cols zeroed: %s | batch=%d" % (C.LOCATION_COLS, BATCH))

    y_te = np.array([s["label"] for s in te])
    results = []
    for run in range(n_runs):
        seed = 42 + run
        C.set_seed(seed)
        macro = macro_map if variant == "macro" else None
        print("\n--- %s | run %d/%d | seed %d ---" % (condition, run + 1, n_runs, seed))
        model = build(arch, macro_dim if variant == "macro" else None)
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc")])
        if run == 0:
            print("params=%d" % model.count_params(), flush=True)
        t0 = time.time()
        steps = max(1, len(tr) // BATCH)
        vsteps = max(1, len(va) // BATCH)
        fit = C.train_with_early_stopping(
            model,
            C.make_e2e_dataset(tr, BATCH, shuffle=True, seed=seed, macro=macro),
            C.make_e2e_dataset(va, BATCH, shuffle=False, macro=macro),
            epochs=epochs, patience=PATIENCE, verbose=2,
            steps_per_epoch=steps, validation_steps=vsteps)
        train_seconds = time.time() - t0

        t1 = time.time()
        p_te = C.predict_e2e(model, te, BATCH, macro)
        p_va = C.predict_e2e(model, va, BATCH, macro)
        p_tr = C.predict_e2e(model, tr, BATCH, macro)
        test_auc = C.safe_auc(y_te, p_te)
        val_auc = C.safe_auc([s["label"] for s in va], p_va)
        train_auc = C.safe_auc([s["label"] for s in tr], p_tr)
        infer_seconds = time.time() - t1

        rec = C.base_record(condition, run, seed, epochs, C.BLSR_V2, extra={
            "origin_script": ORIGIN[condition],
            "arch": arch,
            "variant": variant,
            "n_runs": n_runs,
            "batch_size": BATCH,
            "patience": PATIENCE,
            "macro_dim": macro_dim if variant == "macro" else 0,
            "location_cols_zeroed": True,
            "recurrent_dropout": 0.5 if C.env_flag("V2_E2E_RECURRENT_DROPOUT", True) else 0.0,
            "steps_per_epoch": steps,
            "validation_steps": vsteps,
            "params": int(model.count_params()),
            "split": {"train_games": len(tr), "val_games": len(va), "test_games": len(te)},
            "train_auc": train_auc,
            "val_auc": val_auc,
            "val_auc_best": fit["val_auc_best"],
            "test_auc": test_auc,
            "overfit_gap": train_auc - test_auc,
            "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"],
            "wall_seconds": round(train_seconds + infer_seconds, 2),
            "train_seconds": round(train_seconds, 2),
            "infer_seconds": round(infer_seconds, 2),
            "seconds_per_epoch": round(train_seconds / max(1, fit["epochs_run"]), 3),
        })
        json_path = C.save_record(rec, tag)
        npz_path = C.save_curves(fit, condition, run, tag)
        print("RESULT %s run%d: train_auc=%.4f val_auc_best=%.4f test_auc=%.4f "
              "(best_epoch=%d, epochs_run=%d, %.1fs, %.2fs/epoch)"
              % (condition, run + 1, train_auc, fit["val_auc_best"], test_auc,
                 fit["best_epoch"], fit["epochs_run"], rec["wall_seconds"],
                 rec["seconds_per_epoch"]))
        print("saved: %s | %s" % (os.path.basename(json_path), os.path.basename(npz_path)))
        results.append(rec)

    vals = [r["test_auc"] for r in results]
    print("\nSUMMARY %s: test AUC = %.4f +/- %.4f" % (condition, float(np.mean(vals)),
                                                      float(np.std(vals))))
    return {"condition": condition, "test_aucs": vals}

def main():
    ap = C.cli_parser(__doc__)
    args = ap.parse_args()
    conds = C.resolve_conditions(CONDITIONS, args.conditions)
    if args.list:
        print("\n".join(CONDITIONS))
        return
    cfg = C.config(args.n_runs, args.epochs)
    C.ensure_dirs()
    tee = C.start_log("run_e2e%s" % (("_" + args.tag) if args.tag else ""))
    try:
        print("run_e2e.py | conditions=%s | N_RUNS=%d EPOCHS=%d"
              % (conds, cfg["n_runs"], cfg["epochs"]))
        summary = [run_condition(c, cfg["n_runs"], cfg["epochs"], args.tag) for c in conds]
        print("\n" + "=" * 74)
        for s in summary:
            vals = np.array(s["test_aucs"], dtype=float)
            print("%-20s test AUC %.4f +/- %.4f" % (s["condition"], vals.mean(), vals.std()))
        print("=" * 74)
    finally:
        C.stop_log(tee)

if __name__ == "__main__":
    main()
