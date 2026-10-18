# Data

No dataset files are distributed with this repository. They are built from the public ShuttleSet releases
by `build_dataset.py`. What is kept here is only the small metadata that documents the sample.

| File | Content |
|---|---|
| `dataset_report.json` | Sample size: matches, games, rallies |
| `train_match_ids_v2.txt`, `val_match_ids_v2.txt`, `test_match_ids_v2.txt` | Grouped match-level split |

`build_dataset.py` also produces two files that are not tracked in git:

- `dataset_blsr_v2.csv` — one row per stroke (91,636 rows), with columns `game_id`, `rally_id`, `match_id`,
  `is_target_turn`, `aroundhead`, `backhand`, `time_proportion`, `roundscore_diff`, `continuous_score`,
  `hit_area`, `player_location_area`, `opponent_location_area`, `type`, `is_target_win`
- `macro_features_v2.csv` — 18-dimensional stroke-frequency features per game

## Source data

- **ShuttleSet** — Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023). ShuttleSet: A
  human-annotated stroke-level singles dataset for badminton tactical analysis. KDD '23.
- **ShuttleSet22** — Wang, W.-Y., Du, W.-W., Peng, W.-C., & Ik, T.-U. (2024). Benchmarking stroke
  forecasting with stroke-level badminton dataset. arXiv:2306.15664.

Download both from their original releases and follow their licences. They are not redistributed here, and
neither is the derived dataset.

## Rebuilding

Put the raw `set*/match*.csv` files under `raw/`, then run

```
python build_dataset.py
```

Duplicated matches between the two releases are dropped, leaving 94 matches, 223 games and 7,585 rallies.
