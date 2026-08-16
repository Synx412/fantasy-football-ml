from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING_INPUT = ROOT / "data" / "fpl_training_2024_25.csv"
RAW_INPUT = ROOT / "data" / "fpl_2024_25_merged_gw.csv"
OUTPUT = ROOT / "data" / "fpl_multitask_training_2024_25.csv"


def build_multitask_history() -> pd.DataFrame:
    training = pd.read_csv(TRAINING_INPUT)
    raw = pd.read_csv(RAW_INPUT)

    required_training = {"name", "GW", "kickoff_time", "future_points"}
    required_raw = {
        "name",
        "GW",
        "kickoff_time",
        "fixture",
        "minutes",
        "starts",
        "total_points",
        "saves",
    }
    missing_training = required_training - set(training.columns)
    missing_raw = required_raw - set(raw.columns)
    if missing_training:
        raise ValueError(f"Training input is missing: {', '.join(sorted(missing_training))}")
    if missing_raw:
        raise ValueError(f"Raw input is missing: {', '.join(sorted(missing_raw))}")

    keys = ["name", "GW", "kickoff_time"]
    ordered = raw.copy()
    ordered["_kickoff"] = pd.to_datetime(ordered["kickoff_time"], errors="coerce", utc=True)
    ordered = ordered.sort_values(["name", "_kickoff", "fixture"]).drop(columns="_kickoff")
    ordered["historical_saves"] = ordered.groupby("name", sort=False)["saves"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").fillna(0.0).cumsum().shift(1)
    ).fillna(0.0)
    targets = ordered[keys + ["minutes", "starts", "total_points", "historical_saves"]].copy()
    targets = targets.rename(
        columns={
            "minutes": "future_minutes",
            "starts": "future_starts",
            "total_points": "raw_future_points",
            "historical_saves": "saves",
        }
    )
    targets["future_fixture_count"] = ordered.groupby(["name", "GW"])["fixture"].transform("count")
    targets["future_started"] = (targets["future_starts"] > 0).astype(int)
    targets["future_appearance"] = (targets["future_minutes"] > 0).astype(int)
    targets["future_start_rate"] = pd.to_numeric(
        targets["future_starts"], errors="coerce"
    ).clip(0.0, 1.0)

    result = training.merge(targets, on=keys, how="left", validate="one_to_one")
    target_columns = [
        "future_minutes",
        "future_starts",
        "future_fixture_count",
        "future_started",
        "future_appearance",
        "future_start_rate",
    ]
    if result[target_columns].isna().any().any():
        missing_rows = int(result[target_columns].isna().any(axis=1).sum())
        raise ValueError(f"Could not attach multitask targets to {missing_rows} training rows.")

    point_delta = (
        pd.to_numeric(result["future_points"], errors="coerce")
        - pd.to_numeric(result["raw_future_points"], errors="coerce")
    ).abs()
    if float(point_delta.mean()) > 0.10:
        raise ValueError("The raw gameweek rows do not align with the existing point targets.")

    result["future_minutes"] = pd.to_numeric(result["future_minutes"], errors="coerce").clip(0, 90)
    result["future_starts"] = pd.to_numeric(result["future_starts"], errors="coerce").clip(0, 1)
    result["future_fixture_count"] = pd.to_numeric(
        result["future_fixture_count"], errors="coerce"
    ).clip(1, 2)
    result["future_started"] = result["future_started"].astype(int)
    result["future_appearance"] = result["future_appearance"].astype(int)
    result["future_start_rate"] = pd.to_numeric(
        result["future_start_rate"], errors="coerce"
    ).clip(0.0, 1.0)
    result = result.drop(columns=["raw_future_points"])

    numeric = result.select_dtypes(include=[np.number]).columns
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("The generated training table contains non-finite numeric values.")
    return result


def main() -> None:
    result = build_multitask_history()
    result.to_csv(OUTPUT, index=False)
    print(
        f"Wrote {len(result):,} rows to {OUTPUT.name}; "
        f"start rate={result['future_started'].mean():.3f}, "
        f"appearance rate={result['future_appearance'].mean():.3f}."
    )


if __name__ == "__main__":
    main()
