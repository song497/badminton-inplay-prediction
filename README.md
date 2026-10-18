# In-play versus post-hoc prediction of badminton match outcomes

Code and result files accompanying the manuscript

> **The ceiling is set by information availability: in-play versus post-hoc prediction of badminton match
> outcomes**
> *<AUTHORS>*, *International Journal of Performance Analysis in Sport* (submitted).

The study asks when, during a badminton game, the outcome actually becomes predictable, how far the
performance obtained from whole-match aggregate data overstates that in-play level, and whether combining
information sources can raise it. Every predictor is given an **earliest-computable timestamp**, and all
comparisons are made under one data split and one evaluation standard.

---

## 1. What is in this repository

| Path | Contents |
|---|---|
| `config.py` | Paths and reproducibility constants (folds, seeds) |
| `common.py` | Data loading, feature construction, models, cross-validation utilities |
| `build_dataset.py` | Builds `data/dataset_blsr_v2.csv` from the public ShuttleSet releases |
| `micro_bilstm.py` | Rally-level (micro) models; the BiLSTM is the micro feature extractor |
| `fusion_two_stage.py` | Seven two-stage fusion strategies |
| `end_to_end.py` | End-to-end GRU / LSTM / Transformer on game-level stroke sequences |
| `arms_timealigned.py` | Seven information arms × six prediction time points |
| `inplay_vs_posthoc.py` | In-play versus post-hoc comparison, tautology check |
| `probability_quality.py` | Accuracy, Brier score, ECE and net benefit by time point |
| `reliability_icc.py` | Weighted intraclass correlation of stroke frequencies |
| `segments.py` | First-*m* versus last-*m* rally window analysis |
| `time_to_criterion.py` | Rallies required to reach target AUC values |
| `within_rally.py` | Within-rally prediction curve |
| `bootstrap_ece.py` | Bootstrap confidence intervals for calibration differences |
| `close_subset.py` | Undecided-game subset (score difference ≤ 2), fold-mean test |
| `fig1_framework.py` | Figure 1 |
| `fig2_5_results.py` | Figures 2–5 |
| `data/` | Sample metadata and the grouped match split (see `data/README.md`) |
| `results/` | Pre-computed result files (JSON/CSV) for cross-checking |
| `figures/` | Pre-rendered figures (grayscale, 600 dpi) |

## 2. Manuscript item to script

| Manuscript item | Script | Output |
|---|---|---|
| Methods — data and sample | `build_dataset.py` | `data/dataset_blsr_v2.csv`, `data/dataset_report.json` |
| Table 1 — sample composition | `build_dataset.py` | `data/dataset_report.json` |
| Table 2, Figure 2 — arms × time points | `arms_timealigned.py` | `results/timealigned_learner.json` |
| Table 3, Figure 4 — in-play vs post-hoc | `inplay_vs_posthoc.py` | `results/inplay_lr.json` |
| Table 4 — probability quality, net benefit | `probability_quality.py` | `results/quality_lr.json` |
| Figure 3a — reliability (ICC) and style-arm AUC | `reliability_icc.py` | `results/snr_summary.json`, `snr_by_type.csv` |
| Figure 3b — proportion leading and eventually winning | `inplay_vs_posthoc.py` | `results/inplay_lr.json` (`taut`) |
| Figure 5 — within-rally prediction | `within_rally.py` | `results/within_rally_lstm_curve.json` |
| Time-to-criterion values (Abstract, Results) | `time_to_criterion.py` | `results/t2c_summary.json` |
| Segment analysis (Results) | `segments.py` | `results/segment_information.json` |
| Bootstrap CI for ECE (Table 4 text) | `bootstrap_ece.py` | `results/ece_bootstrap.json` |
| Undecided-subset significance (Results) | `close_subset.py` | `results/close_subset_foldmean.json` |
| Fusion figures quoted in Results | `fusion_two_stage.py` | `results/fusion_learner_check.json` |
| End-to-end figures quoted in Results | `end_to_end.py` | printed summary |
| Micro-model selection (0.8225 BiLSTM) | `micro_bilstm.py` | `results/micro_model_choice.json` |
| Figure 1 | `fig1_framework.py` | `figures/Fig1_framework.png` |
| Figures 2–5 | `fig2_5_results.py` | `figures/Fig2_arms.png` … |

## 3. Environment

```
Python 3.11
TensorFlow 2.15 (CPU is sufficient)
```

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts are run from the repository root:

```bash
python arms_timealigned.py
```

## 4. Data

The analysis uses two public stroke-level badminton datasets, **ShuttleSet** and **ShuttleSet22**
(Wang et al., 2023; 2024). They are not redistributed here; please obtain them from their original
releases (see `data/README.md`) and then run

```bash
python build_dataset.py
```

which writes `data/dataset_blsr_v2.csv`, the stroke-frequency features and the grouped
train/validation/test match lists. The derived dataset itself is **not** redistributed in this
repository (see `data/README.md`); the source releases must be obtained from their original authors.

## 5. Reproducing the results

```bash
python build_dataset.py        # only if the raw ShuttleSet data are available
python micro_bilstm.py         # micro feature extractor (run first)
python arms_timealigned.py     # Table 2, Figure 2
python inplay_vs_posthoc.py    # Table 3, Figure 4, Figure 3b, time-point curves
python probability_quality.py  # Table 4
python reliability_icc.py      # Figure 3a
python time_to_criterion.py    # time-to-criterion values
python segments.py             # segment analysis
python within_rally.py         # Figure 5
python fusion_two_stage.py     # fusion numbers
python end_to_end.py           # end-to-end numbers
python bootstrap_ece.py        # calibration confidence intervals
python close_subset.py         # undecided-subset test
python fig1_framework.py && python fig2_5_results.py
```

Every script writes into `results/` or `figures/`. The pre-computed result files and figures are
included so that the reported numbers can be verified without re-training. Note that the analysis
scripts require the derived dataset, which has to be rebuilt first (Section 4).

## 6. Design notes

- **Grouping.** All cross-validation folds are grouped by match, so no match contributes to both training
  and test data. The absolute time-point analyses use the 183 games with at least 30 rallies, which keeps the
  evaluation sample identical across time points.
- **Out-of-fold meta-features.** Probabilities used to build fusion meta-features are always out-of-fold;
  test-set probabilities come from a model trained on the full training set.
- **Fold mean.** Performance is reported as the mean over folds rather than from pooled predictions.
- **Seeds.** Deep models are trained five times with seeds 42–46; reported values are means.
- **Evaluation stance.** The post-hoc reference (whole-game stroke frequencies) is evaluated on the same
  population as the in-play curves so that differences in stance are not confounded with differences in
  sample.

## 7. Licence

Code: MIT (see `LICENSE`). The data derive from ShuttleSet and ShuttleSet22 and remain under their
original licences.
