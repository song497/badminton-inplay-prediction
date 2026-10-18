# Code for "The ceiling is set by information availability"

Code and result files for the manuscript *The ceiling is set by information availability: in-play versus
post-hoc prediction of badminton match outcomes* (<authors>), submitted to *International Journal of
Performance Analysis in Sport*.

https://github.com/song497/badminton-inplay-prediction

## What the code does

We compare prediction of badminton game outcomes using information that is available while a game is in
progress with prediction using whole-match aggregate data. Every predictor carries an earliest-computable
time point, and all comparisons are run on the same split with the same evaluation setup.

## Requirements

Python 3.11 with TensorFlow 2.15. A CPU is enough.

```
pip install -r requirements.txt
```

## Data

The analyses use ShuttleSet and ShuttleSet22. Neither is redistributed here; download them from the
original releases (see `data/README.md`), put the raw `set*/match*.csv` files under `raw/`, and run

```
python build_dataset.py
```

This writes the processed dataset and the match-level split files into `data/`. Duplicate matches across
the two releases are dropped, which leaves 94 matches, 223 games and 7,585 rallies.

## Running the scripts

Run everything from the repository root. Scripts write into `results/`. Deep models are trained five times
with seeds 42-46 and the reported values are means over runs.

```
python micro_bilstm.py          # rally-level models (the BiLSTM is the micro feature extractor)
python fusion_two_stage.py      # two-stage fusion strategies
python end_to_end.py            # end-to-end sequence models
python arms_timealigned.py      # information arms across prediction time points
python inplay_vs_posthoc.py     # in-play versus post-hoc comparison, tautology check
python probability_quality.py   # accuracy, Brier score, ECE, net benefit
python reliability_icc.py       # intraclass correlation of stroke frequencies
python time_to_criterion.py     # rallies needed to reach a target AUC
python segments.py              # first-m versus last-m rally windows
python within_rally.py          # within-rally curve
python bootstrap_ece.py         # bootstrap intervals for calibration differences
python close_subset.py          # undecided games (score difference <= 2)
```

## Scripts and outputs

| Output | Script |
|---|---|
| Sample composition (`data/dataset_report.json`) | `build_dataset.py` |
| Information arms across time points (`results/timealigned_learner.json`) | `arms_timealigned.py` |
| In-play versus post-hoc, tautology (`results/inplay_lr.json`) | `inplay_vs_posthoc.py` |
| Probability quality and net benefit (`results/quality_lr.json`) | `probability_quality.py` |
| Intraclass correlation (`results/snr_summary.json`) | `reliability_icc.py` |
| Time to criterion (`results/t2c_summary.json`) | `time_to_criterion.py` |
| Segment analysis (`results/segment_information.json`) | `segments.py` |
| Within-rally curve (`results/within_rally_lstm_curve.json`) | `within_rally.py` |
| Fusion numbers (`results/fusion_learner_check.json`) | `fusion_two_stage.py` |
| Micro-model selection (`results/micro_model_choice.json`) | `micro_bilstm.py` |
| Bootstrap intervals for ECE (`results/ece_bootstrap.json`) | `bootstrap_ece.py` |
| Undecided-subset test (`results/close_subset_foldmean.json`) | `close_subset.py` |

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
