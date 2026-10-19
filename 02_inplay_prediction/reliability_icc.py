# -*- coding: utf-8 -*-
"""
信噪比分解：一局比赛里观测到的"技术频差"，有多少是真实的风格差异，多少是抽样噪声？

方法（每个技术类别、每个时点 t）：
  对每局 g：n_g = 截至 t 的击球数，k_g = 该类别的计数，p_g = k_g / n_g
  Var_obs   = 各局 p_g 的方差（观测到的局间差异）
  Var_noise = 若所有局真实频率都等于合并频率 p̄，则抽样方差 = p̄(1-p̄)/n_g 的均值
  ICC       = (Var_obs − Var_noise) / Var_obs      （可靠度：真实差异占比）
    ICC ≈ 1 → 观测到的差异基本都是真实风格差异
    ICC ≈ 0 → 观测到的差异基本都是抽样噪声

同时输出"可靠度如何随观测时点增长"，以便与"风格类特征的预测力是否随时点增长"对照。
输出：results_inplay/snr_summary.json / snr_by_type.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, DATA, RESULTS, FIGURES, R1, R2, R3, R4  # noqa: E402
import os, json
from collections import Counter
import numpy as np
import pandas as pd

ROOT = str(ROOT)
DATA = str(DATA / "dataset_blsr_v2.csv") if not str(DATA).endswith(".csv") else DATA
OUT = str(R2)
WINDOWS = [5, 10, 15, 20, 25, 30]
WMAX = max(WINDOWS)

df = pd.read_csv(DATA)
TYPES = sorted(df["type"].dropna().unique())

# 逐局的回合序列（与 run_inplay / run_timealigned 一致）
games = {}
for gid, g in df.groupby("game_id", sort=True):
    rallies = []
    for rid, rg in g.groupby("rally_id", sort=True):
        rg = rg.sort_values("time_proportion")
        s = rg[rg["is_target_turn"] == 1]
        r = rg[rg["is_target_turn"] == 0]
        rallies.append({"server_types": Counter(s["type"]), "receiver_types": Counter(r["type"]),
                        "server_shots": int(len(s)), "receiver_shots": int(len(r))})
    games[int(gid)] = {"match_id": int(g["match_id"].iloc[0]), "rallies": rallies}

POP = sorted([g for g, v in games.items() if len(v["rallies"]) >= WMAX])

def per_game_rates(gids, upto_fn, side):
    """returns counts[type] = (k_g array, n_g array)"""
    K = {t: [] for t in TYPES}
    N = []
    for gid in gids:
        rl = games[gid]["rallies"]
        n = upto_fn(gid, rl)
        rl = rl[:n]
        c = Counter()
        tot = 0
        for r in rl:
            c.update(r["server_types"] if side == "server" else r["receiver_types"])
            tot += r["server_shots"] if side == "server" else r["receiver_shots"]
        N.append(tot)
        for t in TYPES:
            K[t].append(c.get(t, 0))
    return {t: (np.array(K[t], float), np.array(N, float)) for t in TYPES}

def icc_table(gids, upto_fn, side):
    """总体 + 逐类别的 ICC"""
    data = per_game_rates(gids, upto_fn, side)
    rows = []
    for t in TYPES:
        k, n = data[t]
        ok = n > 0
        if ok.sum() < 5:
            continue
        p_g = k[ok] / n[ok]
        p_bar = k[ok].sum() / n[ok].sum()
        v_obs = float(np.var(p_g, ddof=1))
        v_noise = float(np.mean(p_bar * (1 - p_bar) / n[ok]))
        icc = (v_obs - v_noise) / v_obs if v_obs > 0 else np.nan
        rows.append({"type": str(t), "side": side, "pooled_rate": float(p_bar),
                     "n_mean": float(n[ok].mean()), "var_obs": v_obs, "var_noise": v_noise,
                     "icc": float(icc), "noise_share": float(v_noise / v_obs) if v_obs > 0 else np.nan})
    return pd.DataFrame(rows)

print("=== 逐类别 ICC（整局，含落点数据集）===")
full_s = icc_table(sorted(games.keys()), lambda gid, rl: len(rl), "server")
full_r = icc_table(sorted(games.keys()), lambda gid, rl: len(rl), "receiver")
res = {"note": "ICC = 1 - noise_share; 真实风格差异占观测方差的比例",
       "full_game": {"server": full_s.to_dict("records"), "receiver": full_r.to_dict("records")},
       "by_cutoff": {}}

for name, tb in (("server", full_s), ("receiver", full_r)):
    w = tb["pooled_rate"].to_numpy() / tb["pooled_rate"].sum()
    print("侧=%s ｜ 按出现频率加权的平均 ICC = %.3f ｜ 噪声占比 = %.3f"
          % (name, float((tb["icc"] * w).sum()), float((tb["noise_share"] * w).sum())))
    print("   ICC 最低的 5 类:", tb.nsmallest(5, "icc")[["type", "icc", "n_mean"]].to_dict("records"))

print("\n=== ICC 随时点的变化（反映'再多看几回合，风格估计是否变可靠'）===")
hdr = "%-8s %-10s %-10s %-10s %-10s" % ("时点", "平均ICC", "噪声占比", "平均回合数", "平均击球数")
print(hdr)
for w in WINDOWS + ["full"]:
    if w == "full":
        gids, fn = sorted(games.keys()), (lambda gid, rl: len(rl))
    else:
        gids, fn = POP, (lambda gid, rl, w=w: w)
    tb = icc_table(gids, fn, "server")
    wts = tb["pooled_rate"].to_numpy() / tb["pooled_rate"].sum()
    icc_w = float((tb["icc"] * wts).sum())
    ns_w = float((tb["noise_share"] * wts).sum())
    n_shot = float(tb["n_mean"].mean() / max(1, (w if w != "full" else 30)) * (w if w != "full" else 30))
    res["by_cutoff"][str(w)] = {"weighted_icc": icc_w, "weighted_noise_share": ns_w,
                                "table": tb.to_dict("records")}
    print("%-8s %-10.3f %-10.3f %-10s %-10.0f" % (str(w), icc_w, ns_w, str(w), tb["n_mean"].mean()))

full_s.to_csv(os.path.join(OUT, "snr_by_type.csv"), index=False, encoding="utf-8-sig")
json.dump(res, open(os.path.join(OUT, "snr_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwritten -> snr_summary.json / snr_by_type.csv")
