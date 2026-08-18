from __future__ import annotations

"""Train the v8 FPL point + availability models on 2025/26 data.

Outputs are deliberately written to the existing production filenames so the
Streamlit app does not need a model-routing migration:

    models/fpl_points_v2.joblib
    models/fpl_multitask_bundle.joblib
    models/fpl_v8_report.json

The point artifact remains schema_version=1 for loader compatibility, but carries
model_version='v8.1' and training_season='2025-26'.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble_models import set_xgboost_device
from src.model import (
    APPEARANCE_TARGETS,
    MINUTES_TARGETS,
    POINT_TARGETS,
    PRETRAINED_MODEL_PATH,
    START_TARGETS,
    _compact_model_bundle,
    _fit,
    _history_key,
    _normalise_history,
    _regressor,
    _target,
    engineer_features,
)

DEFAULT_INPUT = ROOT / "data" / "fpl_multitask_training_2025_26.csv"
POINT_OUT = ROOT / "models" / "fpl_points_v2.joblib"
REPORT_OUT = ROOT / "models" / "fpl_v8_report.json"
POSITIONS = ["GK", "DEF", "MID", "FWD"]

POINT_FEATURES = [
    "price",
    "minutes_per_appearance",
    "start_probability",
    "goals_per90",
    "assists_per90",
    "clean_sheets_per90",
    "saves_per90",
    "form",
    "xg_per90",
    "xa_per90",
    "xgc_per90",
    "cards_per90",
    "bonus_per90",
    "bps_per90",
    "ict_index",
    "threat_per90",
    "creativity_per90",
    "influence_per90",
    "defensive_contribution_per90",
    "cbi_per90",
    "recoveries_per90",
    "tackles_per90",
    "chance_playing",
    "fixture_difficulty",
    "home",
    "fixture_count",
    "team_strength",
    "team_form_points",
    "team_attack_form",
    "team_defence_form",
    "opponent_strength",
    "rest_days",
]

MONOTONE = {
    "minutes_per_appearance": 1,
    "start_probability": 1,
    "goals_per90": 1,
    "assists_per90": 1,
    "clean_sheets_per90": 1,
    "saves_per90": 1,
    "xg_per90": 1,
    "xa_per90": 1,
    "bonus_per90": 1,
    "bps_per90": 1,
    "defensive_contribution_per90": 1,
    "chance_playing": 1,
    "fixture_difficulty": -1,
    "team_strength": 1,
    "opponent_strength": -1,
}
MONOTONE_CONSTRAINTS = tuple(MONOTONE.get(feature, 0) for feature in POINT_FEATURES)


def make_point_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=650,
        max_depth=3,
        learning_rate=0.022,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=15.0,
        reg_lambda=9.0,
        reg_alpha=0.15,
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        n_jobs=4,
        random_state=seed,
        monotone_constraints=MONOTONE_CONSTRAINTS,
    )


def mean_gw_spearman(frame: pd.DataFrame, prediction: np.ndarray | pd.Series) -> float:
    temp = frame.copy()
    temp["_prediction"] = np.asarray(prediction, dtype=float)
    values: list[float] = []
    for _, group in temp.groupby("GW"):
        if group["_prediction"].nunique() < 2 or group["future_points"].nunique() < 2:
            continue
        rho = spearmanr(group["_prediction"], group["future_points"]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    return float(np.mean(values)) if values else float("nan")


def topk_overlap(frame: pd.DataFrame, prediction: np.ndarray | pd.Series, k: int) -> float:
    temp = frame.copy()
    temp["_prediction"] = np.asarray(prediction, dtype=float)
    overlaps: list[float] = []
    for _, group in temp.groupby("GW"):
        k_here = min(int(k), len(group))
        if k_here <= 0:
            continue
        predicted_ids = set(group.nlargest(k_here, "_prediction").index)
        actual_ids = set(group.nlargest(k_here, "future_points").index)
        overlaps.append(len(predicted_ids & actual_ids) / k_here)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def metric_block(frame: pd.DataFrame, prediction: np.ndarray | pd.Series) -> dict[str, float | int]:
    pred = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    actual = frame["future_points"].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "mean_gw_spearman": mean_gw_spearman(frame, pred),
        "top10_overlap": topk_overlap(frame, pred, 10),
        "top25_overlap": topk_overlap(frame, pred, 25),
    }


def choose_official_blend(single: pd.DataFrame) -> tuple[float, dict[str, float | int]]:
    model = single["_model_xp"].to_numpy(dtype=float)
    official = np.maximum(single["official_xp"].to_numpy(dtype=float), 0.0)
    best_weight = 0.0
    best_metrics = metric_block(single, model)
    best_score = float(best_metrics["mae"])
    for weight in np.linspace(0.0, 0.65, 14):
        combined = (1.0 - weight) * model + weight * official
        metrics = metric_block(single, combined)
        score = float(metrics["mae"])
        # MAE is the deployment objective; rank correlation breaks near-ties.
        if score < best_score - 1e-6:
            best_weight, best_metrics, best_score = float(weight), metrics, score
        elif abs(score - best_score) <= 0.005:
            if float(metrics["mean_gw_spearman"]) > float(best_metrics["mean_gw_spearman"]):
                best_weight, best_metrics, best_score = float(weight), metrics, score
    return best_weight, best_metrics


def prepare_frame(history: pd.DataFrame) -> pd.DataFrame:
    history = _normalise_history(history)
    history = history[history["position"].astype(str).str.upper().isin(POSITIONS)].copy()
    frame = engineer_features(history).reset_index(drop=True)
    frame["future_points"] = pd.to_numeric(frame["future_points"], errors="coerce")
    frame["official_xp"] = pd.to_numeric(
        history.get("official_xp", pd.Series(np.nan, index=history.index)), errors="coerce"
    ).to_numpy()
    frame["GW"] = pd.to_numeric(history["GW"], errors="coerce").to_numpy()
    frame["future_fixture_count"] = pd.to_numeric(
        history.get("future_fixture_count", pd.Series(1.0, index=history.index)), errors="coerce"
    ).fillna(1.0).to_numpy()
    frame = frame.dropna(subset=["future_points", "GW"]).reset_index(drop=True)
    for feature in POINT_FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return frame


def train_point_artifact(frame: pd.DataFrame, heldout_start_gw: int) -> tuple[dict, dict]:
    train_mask = frame["GW"].lt(heldout_start_gw)
    test_mask = frame["GW"].ge(heldout_start_gw)
    if train_mask.sum() < 3000 or test_mask.sum() < 500:
        raise RuntimeError(
            f"Chronological split is too small: train={train_mask.sum()}, test={test_mask.sum()}"
        )

    heldout = frame[test_mask].copy()
    heldout["_model_xp"] = np.nan
    per_position: dict[str, dict] = {}

    for idx, position in enumerate(POSITIONS):
        train_rows = frame[train_mask & frame["position"].eq(position)]
        test_rows = frame[test_mask & frame["position"].eq(position)]
        if len(train_rows) < 150 or len(test_rows) < 30:
            raise RuntimeError(f"Not enough {position} rows for a reliable chronological validation.")
        model = make_point_model(8100 + idx)
        recency = 0.65 + 0.35 * (
            (train_rows["GW"] - train_rows["GW"].min())
            / max(float(train_rows["GW"].max() - train_rows["GW"].min()), 1.0)
        )
        model.fit(
            train_rows[POINT_FEATURES],
            train_rows["future_points"],
            sample_weight=recency.to_numpy(dtype=float),
        )
        predicted = np.maximum(model.predict(test_rows[POINT_FEATURES]), 0.0)
        heldout.loc[test_rows.index, "_model_xp"] = predicted
        per_position[position] = metric_block(test_rows, predicted)

    if heldout["_model_xp"].isna().any():
        raise RuntimeError("Point validation did not produce predictions for every held-out row.")

    model_metrics = metric_block(heldout, heldout["_model_xp"])
    # Do NOT tune the live Official-FPL prior from Vaastav's historical ``xP``
    # column. Those values are not guaranteed to be pre-deadline snapshots and
    # therefore are not a trustworthy out-of-sample prior for deployment.  v8.1
    # keeps Official FPL xP as an audit/display source only until we have a
    # timestamped archive captured before each deadline.
    blend_weight = 0.0
    blend_metrics = model_metrics.copy()
    official_metrics = {
        "status": "not_used_for_weight_selection",
        "reason": "historical xP timing is not guaranteed pre-deadline",
    }

    # Final production models use all 2025/26 rows after the untouched validation.
    final_models: dict[str, XGBRegressor] = {}
    for idx, position in enumerate(POSITIONS):
        rows = frame[frame["position"].eq(position)]
        model = make_point_model(9100 + idx)
        recency = 0.65 + 0.35 * (
            (rows["GW"] - rows["GW"].min())
            / max(float(rows["GW"].max() - rows["GW"].min()), 1.0)
        )
        model.fit(
            rows[POINT_FEATURES],
            rows["future_points"],
            sample_weight=recency.to_numpy(dtype=float),
        )
        model.set_params(device="cpu")
        final_models[position] = model

    validation = {
        "heldout_gameweeks": f"{heldout_start_gw}-{int(frame['GW'].max())}",
        "model_only": model_metrics,
        "official_fpl_single_fixture": official_metrics,
        "base_blend_single_fixture": blend_metrics,
        "selected_official_base_blend": blend_weight,
        "per_position": per_position,
        # Backwards-compatible fields used by the existing smoke test/UI.
        "rows": int(len(heldout)),
        "model_only_mae": float(model_metrics["mae"]),
        "model_only_mean_gw_spearman": float(model_metrics["mean_gw_spearman"]),
        "base_fpl_50_50_single_fixture_mae": float(blend_metrics["mae"]),
        "base_fpl_50_50_single_fixture_mean_gw_spearman": float(blend_metrics["mean_gw_spearman"]),
    }
    artifact = {
        "schema_version": 1,
        "kind": "fpl_points_v2",
        "model_version": "v8.1",
        "features": POINT_FEATURES,
        "models": final_models,
        "official_base_blend": float(blend_weight),
        "official_prior_predeadline_validated": False,
        "training_rows": int(len(frame)),
        "training_season": "2025-26",
        "validation": validation,
        "monotone_constraints": {key: int(value) for key, value in MONOTONE.items()},
        "scoring_era": "2025-26 defensive-contribution era",
        "notes": (
            "v8.1: 2025/26 leakage-safe chronological training; Assistant Manager excluded; "
            "team-match exposure denominator; defensive contributions/CBI/recoveries/tackles, "
            "ICT components and xGC included. Official FPL xP is excluded from model features "
            "and receives zero production weight until pre-deadline historical snapshots are available."
        ),
    }
    return artifact, validation


def train_multitask_artifact(history: pd.DataFrame) -> tuple[dict, dict]:
    normalised = _normalise_history(history)
    normalised = normalised[normalised["position"].astype(str).str.upper().isin(POSITIONS)].reset_index(drop=True)
    point_target = _target(normalised, POINT_TARGETS)
    if point_target is None:
        raise RuntimeError("Training data has no point target.")

    training = engineer_features(normalised).reset_index(drop=True)
    training[point_target] = pd.to_numeric(training[point_target], errors="coerce")
    training = training.dropna(subset=[point_target]).reset_index(drop=True)
    bundle = _compact_model_bundle(
        training,
        point_target,
        _target(training, START_TARGETS),
        _target(training, APPEARANCE_TARGETS),
        _target(training, MINUTES_TARGETS),
    )

    price_training = training.copy()
    price_training["price"] = pd.to_numeric(price_training["price"], errors="coerce")
    price_training = price_training.dropna(subset=["price"]).reset_index(drop=True)
    price_model = _fit(_regressor(73, max_iter=55), price_training, price_training["price"])

    if bundle.start_model is not None:
        set_xgboost_device(bundle.start_model, "cpu")
    if bundle.appearance_model is not None:
        set_xgboost_device(bundle.appearance_model, "cpu")
    if bundle.minutes_model is not None:
        set_xgboost_device(bundle.minutes_model, "cpu")
    for group in (bundle.point_models, bundle.starter_point_models, bundle.substitute_point_models):
        for model in group.values():
            set_xgboost_device(model, "cpu")

    artifact = {
        "schema_version": 3,
        "model_version": "v8.1",
        "training_season": "2025-26",
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "history_key": _history_key(normalised),
        "bundle": bundle,
        "price_model": price_model,
    }
    metrics = {
        "training_rows": int(bundle.training_rows),
        "point_mae": bundle.validation_mae,
        "start_brier": bundle.validation_start_brier,
        "appearance_brier": bundle.validation_appearance_brier,
        "minutes_mae": bundle.validation_minutes_mae,
        "trained_availability": list(bundle.trained_availability),
    }
    return artifact, metrics


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--heldout-start-gw", type=int, default=31)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run scripts/build_training_2025_26.py first."
        )
    history = pd.read_csv(args.input, low_memory=False)
    if history.empty:
        raise RuntimeError("Training CSV is empty.")

    frame = prepare_frame(history)
    point_artifact, point_validation = train_point_artifact(frame, args.heldout_start_gw)
    multitask_artifact, multitask_validation = train_multitask_artifact(history)

    POINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(point_artifact, POINT_OUT, compress=3)
    joblib.dump(multitask_artifact, PRETRAINED_MODEL_PATH, compress=3)

    report = {
        "model_version": "v8.1",
        "training_season": "2025-26",
        "point_model": point_validation,
        "availability_model": multitask_validation,
        "artifacts": {
            "point": str(POINT_OUT.relative_to(ROOT)),
            "multitask": str(PRETRAINED_MODEL_PATH.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Wrote {POINT_OUT} ({POINT_OUT.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {PRETRAINED_MODEL_PATH} ({PRETRAINED_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {REPORT_OUT}")
    print("Point held-out MAE:", round(float(point_validation["model_only"]["mae"]), 4))
    print("Point held-out Spearman:", round(float(point_validation["model_only"]["mean_gw_spearman"]), 4))
    print("Official-FPL Base weight (v8.1 safe default):", round(float(point_artifact["official_base_blend"]), 3))
    print("Start Brier:", multitask_validation["start_brier"])
    print("Appearance Brier:", multitask_validation["appearance_brier"])
    print("Minutes MAE:", multitask_validation["minutes_mae"])


if __name__ == "__main__":
    main()
