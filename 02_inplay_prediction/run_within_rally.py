# -*- coding: utf-8 -*-
"""
回合级"前 k 拍"曲线——**固定划分协议**版本（与表3 完全同口径）。

目的：报告只吃原始逐拍序列的模型（BiLSTM，同表3 架构），使 图7 的末端点
      与 表3 的 BiLSTM 行严格一致（同一模型、同一任务、同一划分）。

协议（逐字复刻 run_micro.py）：
  数据 v2（含落点）→ load_split_ids() 的固定 train/val/test 比赛划分
  每个 k：把每个回合的序列截断为前 k 拍 → BiLSTM → 早停(patience=15, 监控 val_auc)
  5 次独立运行，seed = 42..46，报测试集 AUC 的 均值±标准差

用法：--ks 1,2,3   或  --ks 5,8   或  --ks 12,all
输出：results_inplay/e5_fixed_k{ks}.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json, time, argparse
import numpy as np
import common as C

ROOT = r"D:\data\experiments_v2"
OUT = str(R2)
BATCH, PATIENCE, EPOCHS, N_RUNS = 32, 15, 100, 5

ap = argparse.ArgumentParser()
ap.add_argument("--ks", required=True, help="逗号分隔，如 1,2,3 或 12,all")
args = ap.parse_args()
KS = [("all" if x.strip() == "all" else int(x)) for x in args.ks.split(",")]
TAG = args.ks.replace(",", "-")

df, meta = C.load_blsr(C.BLSR_V2, drop_location=False)
ids = C.load_split_ids()
tr_ids, va_ids, te_ids = ids["train"], ids["val"], ids["test"]
FULL = (C.build_rally_sequences(df, tr_ids),
        C.build_rally_sequences(df, va_ids),
        C.build_rally_sequences(df, te_ids))
print("[%s] split: train %d / val %d / test %d rallies" % (TAG, len(FULL[0]), len(FULL[1]), len(FULL[2])),
      flush=True)

def trunc(seqs, k):
    if k == "all":
        return seqs
    out = []
    for s in seqs:
        n = min(int(k), len(s["shots"]))
        t = dict(s)
        t["shots"] = s["shots"][:n]
        t["types"] = s["types"][:n]
        out.append(t)
    return out

y_test = np.array([s["label"] for s in FULL[2]])
res = {"protocol": "fixed split (load_split_ids) + 5 runs seed 42..46; BiLSTM; patience=15",
       "ks": {}, "split": {"train": len(FULL[0]), "val": len(FULL[1]), "test": len(FULL[2])}}

t_all = time.time()
for k in KS:
    tr, va, te = (trunc(FULL[0], k), trunc(FULL[1], k), trunc(FULL[2], k))
    aucs, secs = [], []
    for run in range(N_RUNS):
        seed = 42 + run
        C.set_seed(seed)
        t0 = time.time()
        model = C.compile_model(C.build_bilstm(meta["num_types"]))
        C.train_with_early_stopping(
            model, C.make_micro_dataset(tr, BATCH, shuffle=True, seed=seed),
            C.make_micro_dataset(va, BATCH, shuffle=False),
            epochs=EPOCHS, patience=PATIENCE, verbose=0,
            steps_per_epoch=C.steps_for(tr, BATCH),
            validation_steps=C.steps_for(va, BATCH))
        p = C.predict_micro(model, te, BATCH)
        auc = C.safe_auc(y_test, p)
        aucs.append(float(auc)); secs.append(time.time() - t0)
        print("  k=%-4s run %d/5 seed %d  test_auc=%.4f  (%.0fs)" % (k, run + 1, seed, auc, secs[-1]),
              flush=True)
    res["ks"][str(k)] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=0)),
                         "runs": aucs, "seconds": secs}
    print("k=%-4s  mean %.4f ± %.4f" % (k, np.mean(aucs), np.std(aucs, ddof=0)), flush=True)
    json.dump(res, open(os.path.join(OUT, "e5_fixed_k%s.json" % TAG), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

print("TOTAL %.0fs" % (time.time() - t_all))
print("DONE", TAG)
