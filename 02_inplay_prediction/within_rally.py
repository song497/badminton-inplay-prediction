# -*- coding: utf-8 -*-
"""
Aggregate the per-k rally-level runs into the within-rally curve.

Input : results/02_inplay_prediction/e5_fixed_k*.json   (produced by run_within_rally.py)
Output: results/02_inplay_prediction/within_rally_lstm_curve.json
        test AUC (mean and standard deviation over runs) for k = 1, 2, 3, 5, 8, 12 and the whole rally
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import json
import glob

merged = {}
for path in sorted(glob.glob(os.path.join(str(R2), "e5_fixed_k*.json"))):
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    merged.update(payload.get("ks", {}))

KS = ["1", "2", "3", "5", "8", "12", "all"]
missing = [k for k in KS if k not in merged]
if missing:
    raise SystemExit("missing k in e5 files: %s (run run_within_rally.py first)" % missing)

for k in KS:
    print("  k=%-4s %.4f +- %.4f" % (k, merged[k]["auc_mean"], merged[k]["auc_std"]))

out = os.path.join(str(R2), "within_rally_lstm_curve.json")
json.dump({k: merged[k] for k in KS}, open(out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("written ->", out)
