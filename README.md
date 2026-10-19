# Code for "The ceiling is set by information availability"

Code and result files for the manuscript *The ceiling is set by information availability: in-play versus
post-hoc prediction of badminton match outcomes* (<authors>), submitted to *International Journal of
Performance Analysis in Sport*.

https://github.com/song497/badminton-inplay-prediction

## What the code does

We compare prediction of badminton game outcomes using information that is available while a game is in
progress with prediction using whole-match aggregate data. Every predictor carries an earliest-computable
time point, and all comparisons are run on the same split with the same evaluation setup.

The code is organised in four parts, in the order in which the analyses are reported.

## 01_data_preprocessing

`build_dataset.py` turns the public ShuttleSet releases into the stroke-level dataset and the grouped
match split used everywhere else. Duplicate matches across the two releases are dropped, which leaves
94 matches, 223 games and 7,585 rallies.

## 02_inplay_prediction

Predictability as a function of the prediction time point.

| Script | What it does |
|---|---|
| `arms_timealigned.py` | AUC of the seven information arms at each time point |
| `micro_bilstm.py` | Rally-level models; the BiLSTM is the micro feature extractor |
| `time_to_criterion.py` | Rallies needed to reach a target AUC |
| `within_rally.py` | Within-rally curve |
| `reliability_icc.py` | Intraclass correlation of stroke frequencies |

## 03_posthoc_vs_inplay

Comparison of the two evaluation stances.

| Script | What it does |
|---|---|
| `inplay_vs_posthoc.py` | In-play versus post-hoc AUC, gap size, tautology check |
| `segments.py` | First-m versus last-m rally windows |
| `close_subset.py` | Undecided games (score difference <= 2), fold-mean test |

## 04_fusion

Combining information sources, in the two-stage design and end to end.

| Script | What it does |
|---|---|
| `fusion_two_stage.py` | Seven two-stage fusion strategies |
| `end_to_end.py` | GRU / LSTM / Transformer on game-level stroke sequences |
| `probability_quality.py` | Accuracy, Brier score, ECE, net benefit |
| `bootstrap_ece.py` | Bootstrap intervals for calibration differences |

`common.py` and `config.py` at the top level are shared by all four parts.

## Requirements

Python 3.11 with TensorFlow 2.15. A CPU is enough.

```
pip install -r requirements.txt
```

## Data

The analyses use ShuttleSet and ShuttleSet22. Neither is redistributed here; download them from the
original releases (see `data/README.md`), put the raw `set*/match*.csv` files under `raw/`, and run the
first step.

## Running the analyses

Run everything from the repository root. Scripts write into `results/<part>/`. Deep models are trained
five times with seeds 42-46 and the reported values are means over runs.

The steps below must be run in this order: step 2 writes `micro_oof_rally_probs.csv`, which four later
scripts read, and step 5 writes `inplay_lr.json`, which step 6 reads.

```
# 1. build the dataset (requires data/dataset_blsr_all.csv, see data/README.md)
python 01_data_preprocessing/build_dataset.py

# 2. train the time-aligned arms; writes micro_oof_rally_probs.csv  [slowest step]
python 02_inplay_prediction/run_timealigned.py

# 3. micro model selection
python 02_inplay_prediction/micro_bilstm.py

# 4. arms summary -> Table 2 / Figure 2
python 02_inplay_prediction/arms_timealigned.py

# 5. in-play versus post-hoc -> Table 3 / Figure 3b / Figure 4
python 03_posthoc_vs_inplay/inplay_vs_posthoc.py

# 6. within-rally curve -> Figure 5  (needs step 5)
python 02_inplay_prediction/within_rally.py

# 7. time to criterion
python 02_inplay_prediction/time_to_criterion.py

# 8. reliability -> Figure 3a
python 02_inplay_prediction/reliability_icc.py

# 9. segment analysis
python 03_posthoc_vs_inplay/segments.py

# 10. undecided-subset test
python 03_posthoc_vs_inplay/close_subset.py

# 11. fusion
python 04_fusion/fusion_two_stage.py
python 04_fusion/end_to_end.py

# 12. probability quality -> Table 4
python 04_fusion/probability_quality.py

# 13. calibration intervals
python 04_fusion/bootstrap_ece.py
```

Computed results are kept in `results/`, so the numbers in the paper can be checked without re-training.

## Notes on the setup

- Folds are grouped by match, so no match appears in both training and test data.
- Probabilities used for fusion meta-features are out-of-fold.
- Performance is the mean over folds, not computed from pooled predictions.
- The post-hoc reference is evaluated on the same games as the in-play curves, so the comparison is not
  affected by a change of sample.
- The absolute time-point analyses use the 183 games with at least 30 rallies, which keeps the evaluation
  sample identical across time points.

## Licence

MIT for the code, see `LICENSE`. The ShuttleSet data remain under their original licences.
