from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
V8_HISTORY = DATA_DIR / "fpl_multitask_training_2025_26.csv"
V7_HISTORY = DATA_DIR / "fpl_multitask_training_2024_25.csv"
LEGACY_HISTORY = DATA_DIR / "fpl_training_2024_25.csv"
BUNDLED_HISTORY = V8_HISTORY


def load_bundled_history() -> pd.DataFrame:
    # v8 becomes active automatically once the 2025/26 leakage-safe training CSV
    # is present. Until then the deployed v7 history remains a safe fallback.
    for path in (V8_HISTORY, V7_HISTORY, LEGACY_HISTORY):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Bundled training data is missing. Expected one of: {V8_HISTORY}, {V7_HISTORY}, {LEGACY_HISTORY}"
    )
