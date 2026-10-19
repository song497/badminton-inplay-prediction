"""Paths and global constants for the reproducibility repository.

All scripts import from here so that the code runs from any checkout location:
    from config import ROOT, DATA, RESULTS, FIGURES
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Per-part result folders (mirror the four code folders).
R1 = RESULTS / "01_data_preprocessing"
R2 = RESULTS / "02_inplay_prediction"
R3 = RESULTS / "03_posthoc_vs_inplay"
R4 = RESULTS / "04_fusion"

for _p in (DATA, RESULTS, FIGURES, R1, R2, R3, R4):
    _p.mkdir(exist_ok=True)

# Reproducibility defaults (the manuscript uses these exact values).
N_FOLDS = 5
SEED = 42
N_RUNS = 5          # independent runs for the deep models (seeds 42-46)
