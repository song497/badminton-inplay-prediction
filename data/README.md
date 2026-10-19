# Data

No dataset files are redistributed with this repository. Everything here is rebuilt from the two public
ShuttleSet releases.

## Step 0 — parse the raw releases

Download the two releases and unpack them so that this layout exists:

```
raw/
  set/<match name>/set1.csv, set2.csv, ...
```

Then run

```
python 01_data_preprocessing/parse_raw.py
```

This writes `data/dataset_blsr_all.csv` (88,520 strokes, 244 games, 102 matches) and
`data/game_to_match.csv` (the game to match mapping). Parsing the full set of raw files reproduces the
stroke-level table exactly.

## Step 1 — build the analysis dataset

```
python 01_data_preprocessing/build_dataset.py
```

This writes:

| File | Contents |
|---|---|
| `dataset_blsr_v2.csv` | the stroke-level table used by every analysis |
| `macro_features_v2.csv` | 18-dimensional stroke-frequency features per game |
| `dataset_report.json` | sample size (Table 1) |
| `train_/val_/test_match_ids_v2.txt` | grouped match split |

Duplicate matches between the two releases are dropped, which leaves 94 matches, 223 games and 7,585
rallies.

## Committed metadata

Only small metadata files are tracked:

| File | Contents |
|---|---|
| `dataset_report.json` | sample size |
| `train_/val_/test_match_ids_v2.txt` | the grouped split |

## Source data

- **ShuttleSet** — Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023). ShuttleSet: A
  human-annotated stroke-level singles dataset for badminton tactical analysis. KDD '23.
- **ShuttleSet22** — Wang, W.-Y., Du, W.-W., Peng, W.-C., & Ik, T.-U. (2024). Benchmarking stroke
  forecasting with stroke-level badminton dataset. arXiv:2306.15664.

Download both from their original releases and follow their licences. Neither is redistributed here, and
neither is the derived dataset.
