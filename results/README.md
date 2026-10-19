# Results

Result files are grouped by the part of the analysis that produces them.

## 01_data_preprocessing

Nothing is written here. This step writes into `data/`:

| File | Contents |
|---|---|
| `dataset_blsr_v2.csv` | Stroke-level table used by every analysis |
| `macro_features_v2.csv` | 18-dimensional stroke-frequency features per game |
| `dataset_report.json` | Sample size (Table 1) |
| `train_/val_/test_match_ids_v2.txt` | Grouped match split |

## 02_inplay_prediction

| File | Produced by | Used in |
|---|---|---|
| `micro_oof_rally_probs.csv` | `run_timealigned.py` | input to four downstream scripts |
| `timealigned_summary.json` | `run_timealigned.py` | raw per-fold records |
| `timealigned_learner.json` | `arms_timealigned.py` | Table 2, Figure 2 |
| `micro_model_choice.json` | `micro_bilstm.py` | micro model selection |
| `t2c_summary.json` | `time_to_criterion.py` | time-to-criterion values |
| `snr_summary.json`, `snr_by_type.csv` | `reliability_icc.py` | Figure 3a |
| `within_rally_lstm_curve.json` | `within_rally.py` | Figure 5 |
| `macro_learner_check.json` | `run_timealigned.py` | learner sanity check |
| `e3_*`, `e4_*`, `e5_*`, `e1*` | intermediate runs | within-rally and representation checks |

## 03_posthoc_vs_inplay

| File | Produced by | Used in |
|---|---|---|
| `inplay_lr.json` | `inplay_vs_posthoc.py` | Table 3, Figure 3b, Figure 4 |
| `segment_information.json` | `segments.py` | segment analysis |
| `close_subset_foldmean.json` | `close_subset.py` | undecided-subset test |

## 04_fusion

| File | Produced by | Used in |
|---|---|---|
| `quality_lr.json` | `probability_quality.py` | Table 4 |
| `ece_bootstrap.json` | `bootstrap_ece.py` | calibration intervals |
| `fusion_learner_check.json` | `fusion_two_stage.py` | fusion numbers |
| `equivalence_inplay.json` | `fusion_two_stage.py` | equivalence test |
