"""Paths and global constants for the reproducibility repository.

All scripts import from here so that the code runs from any checkout location:
    from config import ROOT, DATA, RESULTS, FIGURES
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (DATA, RESULTS, FIGURES):
    _p.mkdir(exist_ok=True)

# Reproducibility defaults (the manuscript uses these exact values).
N_FOLDS = 5
SEED = 42
N_RUNS = 5          # independent runs for the deep models (seeds 42-46)
