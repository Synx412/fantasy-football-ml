from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BUNDLED_HISTORY = DATA_DIR / "fpl_multitask_training_2024_25.csv"
LEGACY_HISTORY = DATA_DIR / "fpl_training_2024_25.csv"


def load_bundled_history() -> pd.DataFrame:
    path = BUNDLED_HISTORY if BUNDLED_HISTORY.exists() else LEGACY_HISTORY
    if not path.exists():
        raise FileNotFoundError(f"Bundled training data is missing: {BUNDLED_HISTORY}")
    return pd.read_csv(path)
