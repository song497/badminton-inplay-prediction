# Code for "The ceiling is set by information availability"

Code and result files for the manuscript *The ceiling is set by information availability: in-play versus
post-hoc prediction of badminton match outcomes* (Song Jiayang, Zhou Tong), submitted to *International
Journal of Performance Analysis in Sport*.

https://github.com/song497/badminton-inplay-prediction

## What the code does

We compare prediction of badminton game outcomes using information that is available while a game is in
progress with prediction using whole-match aggregate data. Every predictor carries an earliest-computable
time point, and all comparisons are run on the same split with the same evaluation setup.

The code is organised in four parts, in the order in which the analyses are reported. A short Chinese
run-through of the same workflow is in `read.txt`.

## 01_data_preprocessing

`parse_raw.py` reads the raw ShuttleSet releases and writes one stroke-level table, translating the stroke
types from Chinese into English. `build_dataset.py` turns that table into the analysis dataset: it drops
duplicate matches that appear in both releases, builds the 18-dimensional stroke-frequency features, and
creates the grouped match-level train/validation/test split. What is left is 94 matches, 223 games and
7,585 rallies.

## 02_inplay_prediction

Predictability as a function of the prediction time point.

| Script | What it does |
|---|---|
| `run_timealigned.py` | Trains the seven information arms at each time point (XGBoost); writes the micro out-of-fold probabilities that four later scripts need |
| `arms_timealigned.py` | Same design with logistic regression, which is the learner used in the paper |
| `time_to_criterion.py` | Number of rallies needed to reach a target AUC |
| `run_within_rally.py` | Trains the BiLSTM on the first k shots of each rally |
| `within_rally.py` | Aggregates those runs into the within-rally curve |
| `reliability_icc.py` | Intraclass correlation of stroke frequencies |

## 03_posthoc_vs_inplay

Comparison of the two evaluation stances.

| Script | What it does |
|---|---|
| `inplay_vs_posthoc.py` | In-play versus post-hoc AUC, size of the gap, tautology check |
| `segments.py` | First-m versus last-m rally windows |
| `close_subset.py` | Undecided games (score difference <= 2), fold-mean test |

## 04_fusion

Model comparison and the combination of information sources.

| Script | What it does |
|---|---|
| `micro_bilstm.py` | Rally-level model comparison; the BiLSTM is used as the micro feature extractor |
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

The analyses use two public releases:

- **ShuttleSet** — https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
- **ShuttleSet22** — https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI-Challenge-IJCAI2023/ShuttleSet22

Neither is redistributed here. Download both, place the first under `data/set/` and rename the second to
`data/set22/`, then run the first step. `parse_raw.py` finds these folders automatically (it also accepts
`raw/` or `data/raw/`).

`data/match_list.txt` records the 102 matches used in the paper. The parser follows that list, so the same
dataset is rebuilt even if you have downloaded a larger release. Without the list you would get 153
matches, and none of the numbers in the paper would reproduce.

## Running the analyses

Run everything from the repository root. Scripts write into `results/<part>/`. Deep models are trained five
times with seeds 42-46 and the reported values are means over runs.

Steps 2, 3, 9 and 13 need `micro_oof_rally_probs.csv`, which step 1 writes, so step 1 must come first.
Everything else can be run in any order, or in parallel.

```
# 1. train the time-aligned arms; writes micro_oof_rally_probs.csv  [slowest step]
python 02_inplay_prediction/run_timealigned.py

# 2. arms summary -> Table 2 / Figure 2
python 02_inplay_prediction/arms_timealigned.py

# 3. time to criterion -> values quoted in the Abstract and Section 3.1
python 02_inplay_prediction/time_to_criterion.py

# 4. reliability -> Figure 3a
python 02_inplay_prediction/reliability_icc.py

# 5. within-rally: train, then aggregate -> Figure 5  [slow]
python 02_inplay_prediction/run_within_rally.py
python 02_inplay_prediction/within_rally.py

# 6. in-play versus post-hoc -> Table 3 / Figure 3b / Figure 4
python 03_posthoc_vs_inplay/inplay_vs_posthoc.py

# 7. segment analysis -> Section 3.2
python 03_posthoc_vs_inplay/segments.py

# 8. undecided-subset test -> Section 3.1
python 03_posthoc_vs_inplay/close_subset.py

# 9. micro model comparison
python 04_fusion/micro_bilstm.py

# 10. fusion and end-to-end models -> Section 3.3
python 04_fusion/fusion_two_stage.py
python 04_fusion/end_to_end.py

# 11. probability quality -> Table 4
python 04_fusion/probability_quality.py

# 12. calibration intervals -> Section 3.3
python 04_fusion/bootstrap_ece.py
```

The data steps must be run once before the first analysis:

```
python 01_data_preprocessing/parse_raw.py
python 01_data_preprocessing/build_dataset.py
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
