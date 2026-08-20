from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.ensemble_models import BlendedClassifier, BlendedRegressor
from src.preseason_prior import blend_preseason_availability_prior, blend_preseason_role_prior


NUMERIC_FEATURES = [
    "minutes_per_appearance",
    "start_probability",
    "goals_per90",
    "assists_per90",
    "clean_sheets_per90",
    "saves_per90",
    "rating",
    "form",
    "xg_per90",
    "xa_per90",
    "xgc_per90",
    "threat_per90",
    "creativity_per90",
    "influence_per90",
    "defensive_contribution_per90",
    "cbi_per90",
    "recoveries_per90",
    "tackles_per90",
    "cards_per90",
    "bonus_per90",
    "bps_per90",
    "ict_index",
    "selected_by_percent",
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
    "is_gk",
    "is_def",
    "is_mid",
    "is_fwd",
]
POINT_TARGETS = ["future_points", "next_points", "fantasy_points"]
START_TARGETS = ["future_started", "next_started", "started_next"]
APPEARANCE_TARGETS = ["future_appearance", "next_appearance", "appeared_next"]
MINUTES_TARGETS = ["future_minutes", "next_minutes", "minutes_next"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]
_MODEL_CACHE: dict[str, "_ModelBundle"] = {}
_PRICE_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
PRETRAINED_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "fpl_multitask_bundle.joblib"
)
XP_CALIBRATOR_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "fpl_xp_calibrator.joblib"
)
POINT_V2_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "fpl_points_v2.joblib"
)

XP_CALIBRATOR_FEATURES = [
    "raw_xp",
    "price",
    "minutes_per_appearance", "start_probability",
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "form", "bonus_per90", "bps_per90", "ict_index",
    "chance_playing", "fixture_difficulty", "home", "fixture_count",
    "team_strength", "team_form_points", "team_attack_form", "team_defence_form",
    "opponent_strength", "rest_days",
    "is_gk", "is_def", "is_mid", "is_fwd",
]


@dataclass
class PredictionResult:
    players: pd.DataFrame
    mode: str
    validation_mae: Optional[float]
    training_rows: int
    model_detail: str = ""
    validation_start_brier: Optional[float] = None
    validation_appearance_brier: Optional[float] = None
    validation_minutes_mae: Optional[float] = None


@dataclass
class _ModelBundle:
    training_rows: int
    start_model: Optional[Any]
    appearance_model: Optional[Any]
    minutes_model: Optional[Any]
    point_models: dict[str, Any]
    starter_point_models: dict[str, Any]
    substitute_point_models: dict[str, Any]
    role_point_priors: dict[str, tuple[float, float]]
    interval_offsets: dict[str, tuple[float, float]]
    validation_mae: Optional[float]
    validation_start_brier: Optional[float]
    validation_appearance_brier: Optional[float]
    validation_minutes_mae: Optional[float]
    trained_availability: list[str]


def _number(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    defaults = {
        "minutes": 0.0,
        "appearances": 0.0,
        "starts": 0.0,
        "goals": 0.0,
        "assists": 0.0,
        "clean_sheets": 0.0,
        "saves": 0.0,
        "rating": 6.0,
        "form": 0.0,
        "xg": 0.0,
        "xa": 0.0,
        "xgc": 0.0,
        "threat": 0.0,
        "creativity": 0.0,
        "influence": 0.0,
        "defensive_contribution": 0.0,
        "clearances_blocks_interceptions": 0.0,
        "recoveries": 0.0,
        "tackles": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
        "bonus": 0.0,
        "bps": 0.0,
        "ict_index": 0.0,
        "selected_by_percent": 0.0,
        "chance_playing": 1.0,
        "fixture_difficulty": 3.0,
        "home": 0.5,
        "fixture_count": 1.0,
        "team_strength": 0.5,
        "team_form_points": 1.5,
        "team_attack_form": 1.35,
        "team_defence_form": 1.35,
        "opponent_strength": 0.5,
        "rest_days": 7.0,
    }
    for column, default in defaults.items():
        result[column] = _number(result, column, default)

    if "position" not in result.columns:
        result["position"] = "MID"
    result["position"] = (
        result["position"]
        .fillna("MID")
        .astype(str)
        .str.upper()
        .replace(
            {
                "ATT": "FWD", "FW": "FWD", "ST": "FWD",
                "AM": "MID", "CM": "MID", "DM": "MID", "RM": "MID", "LM": "MID",
                "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
                "GOALKEEPER": "GK",
            }
        )
    )

    # Historical training rows use team-match exposure in the legacy
    # ``appearances`` column (normally a recent rolling window). Live providers,
    # however, may expose true player appearances. Prefer an explicit
    # team_matches_observed denominator when available so minutes/start features
    # have the same meaning at training and inference time: expected minutes and
    # starts per TEAM match, not per player appearance.
    appearances = result["appearances"].clip(lower=1.0)
    exposure_matches = appearances.copy()
    if "team_matches_observed" in df.columns:
        team_matches = pd.to_numeric(df["team_matches_observed"], errors="coerce")
        valid_team_matches = team_matches.notna() & team_matches.gt(0.0)
        exposure_matches.loc[valid_team_matches] = team_matches.loc[valid_team_matches]
    exposure_matches = exposure_matches.clip(lower=1.0)

    minutes = result["minutes"].clip(lower=1.0)
    result["minutes_per_appearance"] = (result["minutes"] / exposure_matches).clip(0.0, 90.0)
    inferred_start = (result["starts"] / exposure_matches).clip(0.0, 1.0)
    if "start_probability" in df.columns:
        supplied = pd.to_numeric(df["start_probability"], errors="coerce")
        result["start_probability"] = supplied.fillna(inferred_start).clip(0.0, 1.0)
    else:
        result["start_probability"] = inferred_start

    result["goals_per90"] = 90.0 * result["goals"] / minutes
    result["assists_per90"] = 90.0 * result["assists"] / minutes
    result["clean_sheets_per90"] = 90.0 * result["clean_sheets"] / minutes
    result["saves_per90"] = 90.0 * result["saves"] / minutes
    result["xg_per90"] = 90.0 * result["xg"] / minutes
    result["xa_per90"] = 90.0 * result["xa"] / minutes
    result["xgc_per90"] = 90.0 * result["xgc"] / minutes
    result["threat_per90"] = 90.0 * result["threat"] / minutes
    result["creativity_per90"] = 90.0 * result["creativity"] / minutes
    result["influence_per90"] = 90.0 * result["influence"] / minutes
    result["defensive_contribution_per90"] = 90.0 * result["defensive_contribution"] / minutes
    result["cbi_per90"] = 90.0 * result["clearances_blocks_interceptions"] / minutes
    result["recoveries_per90"] = 90.0 * result["recoveries"] / minutes
    result["tackles_per90"] = 90.0 * result["tackles"] / minutes
    result["cards_per90"] = 90.0 * (
        result["yellow_cards"] + 3.0 * result["red_cards"]
    ) / minutes
    result["bonus_per90"] = 90.0 * result["bonus"] / minutes
    result["bps_per90"] = 90.0 * result["bps"] / minutes
    for position, column in [("GK", "is_gk"), ("DEF", "is_def"), ("MID", "is_mid"), ("FWD", "is_fwd")]:
        result[column] = (result["position"] == position).astype(float)
    return result


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES)],
        remainder="drop",
    )


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.075,
                    max_iter=65,
                    max_leaf_nodes=16,
                    min_samples_leaf=55,
                    l2_regularization=0.8,
                    random_state=seed,
                ),
            ),
        ]
    )


def _regressor(
    seed: int,
    *,
    max_iter: int = 75,
    loss: str = "squared_error",
    quantile: Optional[float] = None,
) -> Pipeline:
    kwargs: dict[str, Any] = {
        "loss": loss,
        "learning_rate": 0.07,
        "max_iter": max_iter,
        "max_leaf_nodes": 18,
        "min_samples_leaf": 50,
        "l2_regularization": 0.65,
        "random_state": seed,
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return Pipeline(
        [("prep", _preprocessor()), ("model", HistGradientBoostingRegressor(**kwargs))]
    )


def _xgb_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=260,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=8.0,
                    reg_lambda=2.0,
                    reg_alpha=0.05,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device=os.getenv("FANTASY_XGB_DEVICE", "cpu"),
                    n_jobs=4,
                    random_state=seed,
                ),
            ),
        ]
    )


def _xgb_regressor(seed: int, *, n_estimators: int = 320) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=8.0,
                    reg_lambda=2.0,
                    reg_alpha=0.05,
                    objective="reg:squarederror",
                    tree_method="hist",
                    device=os.getenv("FANTASY_XGB_DEVICE", "cpu"),
                    n_jobs=4,
                    random_state=seed,
                ),
            ),
        ]
    )


def _availability_classifier(seed: int, *, xgb_weight: float) -> BlendedClassifier:
    return BlendedClassifier(
        models=[_classifier(seed), _xgb_classifier(seed + 1000)],
        weights=[1.0 - xgb_weight, xgb_weight],
    )


def _availability_regressor(seed: int, *, xgb_weight: float = 0.90) -> BlendedRegressor:
    return BlendedRegressor(
        models=[_regressor(seed, max_iter=65), _xgb_regressor(seed + 1000, n_estimators=340)],
        weights=[1.0 - xgb_weight, xgb_weight],
    )


def _point_regressor(seed: int, *, max_iter: int = 70, xgb_weight: float = 0.40) -> BlendedRegressor:
    return BlendedRegressor(
        models=[_regressor(seed, max_iter=max_iter), _xgb_regressor(seed + 1000, n_estimators=300)],
        weights=[1.0 - xgb_weight, xgb_weight],
    )


def _fit(model: Any, frame: pd.DataFrame, target: pd.Series) -> Any:
    weights = _recency_weights(frame)
    if isinstance(model, (BlendedClassifier, BlendedRegressor)):
        model.fit(frame[NUMERIC_FEATURES], target, sample_weight=weights)
    elif isinstance(model, Pipeline):
        model.fit(
            frame[NUMERIC_FEATURES],
            target,
            model__sample_weight=weights,
        )
    else:
        model.fit(frame[NUMERIC_FEATURES], target, sample_weight=weights)
    return model


def _probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    values = model.predict_proba(frame[NUMERIC_FEATURES])
    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline):
        classes = getattr(model.named_steps["model"], "classes_", None)
    classes = np.asarray(classes if classes is not None else [0, 1])
    positive = np.flatnonzero(classes == 1)
    return values[:, int(positive[0])] if len(positive) else np.zeros(len(frame))


def _target(frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    return next((candidate for candidate in candidates if candidate in frame.columns), None)


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.copy()
    if "GW" in ordered.columns:
        gameweeks = pd.to_numeric(ordered["GW"], errors="coerce")
        unique = np.sort(gameweeks.dropna().unique())
        if len(unique) >= 5:
            cutoff = unique[min(max(int(len(unique) * 0.80), 1), len(unique) - 1)]
            before = ordered[gameweeks < cutoff]
            after = ordered[gameweeks >= cutoff]
            if not before.empty and not after.empty:
                return before, after
    if "kickoff_time" in ordered.columns:
        order = pd.to_datetime(ordered["kickoff_time"], errors="coerce", utc=True)
        ordered = ordered.assign(_time=order).sort_values("_time").drop(columns="_time")
    split = min(max(int(len(ordered) * 0.80), 1), max(len(ordered) - 1, 1))
    return ordered.iloc[:split], ordered.iloc[split:]


def _recency_weights(frame: pd.DataFrame) -> np.ndarray:
    if "GW" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    gameweeks = pd.to_numeric(frame["GW"], errors="coerce")
    if gameweeks.notna().sum() == 0:
        return np.ones(len(frame), dtype=float)
    minimum = float(gameweeks.min())
    span = max(float(gameweeks.max()) - minimum, 1.0)
    return (0.55 + 0.45 * ((gameweeks.fillna(minimum) - minimum) / span)).to_numpy()


def _base_scenario(row: pd.Series) -> dict[str, Any]:
    return {
        "fixture_difficulty": row.get("fixture_difficulty", 3.0),
        "home": row.get("home", 0.5),
        "fixture_count": 1.0 if float(row.get("fixture_count", 1.0) or 0.0) > 0 else 0.0,
        "opponent_strength": row.get("opponent_strength", 0.5),
        "rest_days": row.get("rest_days", 7.0),
        "next_opponent": row.get("next_opponent", "TBD"),
        "next_kickoff": row.get("next_kickoff", ""),
        "period_index": 0,
    }


def _expand_scenarios(players: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, str]:
    horizon = max(1, int(horizon))
    expanded: list[dict[str, Any]] = []
    all_real = True
    for source_index, row in players.iterrows():
        value = row.get("fixture_scenarios")
        scenarios = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
        scenarios = [item for item in scenarios if int(item.get("period_index", 0)) < horizon]
        if not scenarios:
            all_real = False
            base = _base_scenario(row)
            scenarios = [{**base, "period_index": period} for period in range(horizon)]
        for scenario in scenarios:
            record = row.to_dict()
            record.update(scenario)

            # ClubElo was merged before fixture expansion. Resolve the actual
            # scenario opponent here so double gameweeks and horizons >1 do not
            # reuse the first opponent's Elo rating.
            strength_map = row.get("clubelo_strength_map")
            scenario_opponent = str(record.get("next_opponent") or "")
            if isinstance(strength_map, dict) and scenario_opponent in strength_map:
                record["clubelo_opponent_strength"] = float(strength_map.get(scenario_opponent))

            period = max(0, int(scenario.get("period_index", 0)))
            record["_source_index"] = source_index
            record["_period_index"] = period
            record["_scenario_weight"] = 0.92**period
            if period > 0:
                record["lineup_status"] = ""
                record["lineup_consensus_probability"] = np.nan
                record["lineup_source_count"] = 0.0
                record["lineup_expected_minutes"] = np.nan
            expanded.append(record)
    mode = "fixture-by-fixture horizon" if all_real else "repeated-fixture fallback horizon"
    return pd.DataFrame(expanded).reset_index(drop=True), mode


def _live_availability(
    frame: pd.DataFrame,
    start: np.ndarray,
    appearance: np.ndarray,
    minutes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fuse independent predicted-lineup evidence before availability/confirmed
    # lineup overrides. This applies to every player with matching intelligence.
    if "lineup_consensus_probability" in frame.columns:
        consensus = pd.to_numeric(
            frame["lineup_consensus_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        counts = pd.to_numeric(
            frame.get("lineup_source_count", pd.Series(1.0, index=frame.index)),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float)
        valid = np.isfinite(consensus)
        # One source is useful but not decisive. More independent sources gain
        # influence, capped so the trained model still matters until confirmed.
        weight = np.clip(0.15 + 0.10 * counts, 0.0, 0.75)
        start[valid] = (
            (1.0 - weight[valid]) * np.clip(start[valid], 0.0, 1.0)
            + weight[valid] * np.clip(consensus[valid], 0.0, 1.0)
        )
        appearance[valid] = np.maximum(appearance[valid], start[valid])

    if "lineup_expected_minutes" in frame.columns:
        consensus_minutes = pd.to_numeric(
            frame["lineup_expected_minutes"], errors="coerce"
        ).to_numpy(dtype=float)
        valid_minutes = np.isfinite(consensus_minutes)
        if valid_minutes.any():
            minutes[valid_minutes] = (
                0.55 * np.clip(minutes[valid_minutes], 0.0, 90.0)
                + 0.45 * np.clip(consensus_minutes[valid_minutes], 0.0, 90.0)
            )

    chance = frame["chance_playing"].clip(0.0, 1.0).to_numpy()
    has_fixture = (frame["fixture_count"].to_numpy() > 0).astype(float)
    start = np.clip(start, 0.0, 1.0) * chance * has_fixture
    appearance = np.clip(appearance, 0.0, 1.0) * chance * has_fixture
    minutes = np.clip(minutes, 0.0, 90.0) * chance * has_fixture
    if "lineup_status" in frame.columns:
        status = frame["lineup_status"].fillna("").astype(str).str.lower()
        starter = status.eq("starter").to_numpy()
        bench = status.eq("bench").to_numpy()
        out = status.isin(["out", "not_in_squad"]).to_numpy()
        start[starter], appearance[starter] = 1.0, 1.0
        minutes[starter] = np.maximum(minutes[starter], 65.0)
        start[bench], appearance[bench] = 0.0, 1.0
        minutes[bench] = np.clip(minutes[bench], 8.0, 35.0)
        start[out], appearance[out], minutes[out] = 0.0, 0.0, 0.0
    return start, appearance, minutes


def _statistical_projection(frame: pd.DataFrame) -> np.ndarray:
    p = engineer_features(frame)
    goal_points = p["position"].map({"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}).fillna(4.5)
    clean_points = p["position"].map({"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}).fillna(1.0)
    appearance = (
        p["chance_playing"].clip(0, 1)
        * (0.35 + 0.65 * p["start_probability"].clip(0, 1))
        * (p["minutes_per_appearance"] / 90.0).clip(0, 1)
    )
    attack = appearance * (
        goal_points * (0.65 * p["goals_per90"] + 0.35 * p["xg_per90"])
        + 3.0 * (0.65 * p["assists_per90"] + 0.35 * p["xa_per90"])
    )
    defence = appearance * clean_points * p["clean_sheets_per90"].clip(0, 1)
    base = appearance * (
        2.0
        + 0.30 * (p["rating"] - 6.0).clip(-1, 3)
        + 0.035 * p["bps_per90"].clip(0, 60)
        + 0.07 * p["form"].clip(0, 12)
    )
    fixture = (
        1.0
        + 0.055 * (3.0 - p["fixture_difficulty"].clip(1, 5))
        + 0.035 * (p["home"].clip(0, 1) - 0.5)
        + 0.10 * (p["team_strength"].clip(0, 1) - p["opponent_strength"].clip(0, 1))
    ).clip(0.70, 1.35)
    return np.maximum((attack + defence + base) * fixture, 0.0).to_numpy()


def _estimate_missing_prices_ml_legacy(
    players: pd.DataFrame,
    history: Optional[pd.DataFrame],
    budget: float,
    squad_size: int,
) -> pd.DataFrame:
    result = players.copy().reset_index(drop=True)
    if "price" not in result.columns:
        result["price"] = np.nan
    if "price_source" not in result.columns:
        result["price_source"] = "Missing"
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    missing = result["price"].isna() | (result["price"] <= 0)
    if not missing.any() or history is None or history.empty or "price" not in history.columns:
        return result

    training = engineer_features(history).reset_index(drop=True)
    training["price"] = pd.to_numeric(training["price"], errors="coerce")
    training = training.dropna(subset="price")
    current = engineer_features(result).reset_index(drop=True)
    predicted = np.full(len(result), np.nan)
    for position in POSITIONS:
        train_position = training[training["position"] == position]
        current_position = current[(current["position"] == position) & missing]
        if len(train_position) < 250 or current_position.empty:
            continue
        model = _fit(_regressor(73, max_iter=100), train_position, train_position["price"])
        predicted[current_position.index] = model.predict(current_position[NUMERIC_FEATURES])

    usable = missing.to_numpy() & np.isfinite(predicted)
    if usable.any():
        median = max(float(np.nanmedian(predicted[usable])), 0.1)
        desired = max(float(budget) / max(int(squad_size), 1) * 0.82, 3.5)
        scale = float(np.clip(desired / median, 0.80, 1.25))
        result.loc[usable, "price"] = np.round(np.clip(predicted[usable] * scale, 3.5, 15.0), 1)
        result.loc[usable, "price_source"] = "ML-estimated price — not official"
    return result


def _fallback_result(players: pd.DataFrame, expanded: pd.DataFrame, training_rows: int) -> PredictionResult:
    current = engineer_features(expanded)
    current["_points"] = _statistical_projection(current) * current["_scenario_weight"]
    grouped = current.groupby("_source_index", sort=False)
    output = players.copy()
    output["predicted_points"] = grouped["_points"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p10"] = 0.65 * output["predicted_points"]
    output["predicted_points_p90"] = 1.45 * output["predicted_points"]
    output["point_uncertainty"] = output["predicted_points_p90"] - output["predicted_points_p10"]
    engineered = engineer_features(players)
    output["start_probability"] = engineered["start_probability"]
    output["appearance_probability"] = engineered["chance_playing"].clip(0, 1)
    output["expected_minutes_next"] = engineered["minutes_per_appearance"]
    output["expected_minutes"] = engineered["minutes_per_appearance"]
    output["projection_confidence"] = 0.35 * output["appearance_probability"]
    output["prediction_mode"] = "Statistical fallback"
    return PredictionResult(
        output,
        "fallback",
        None,
        training_rows,
        "No suitable labelled history; using the explicit statistical fallback.",
    )


def _predict_players_legacy(
    players: pd.DataFrame,
    history: Optional[pd.DataFrame],
    horizon: int = 1,
) -> PredictionResult:
    base = players.copy().reset_index(drop=True)
    expanded, horizon_mode = _expand_scenarios(base, horizon)
    current = engineer_features(expanded).reset_index(drop=True)

    if history is None or history.empty:
        return _fallback_result(base, expanded, 0)
    history = history.copy()
    history.columns = [
        "GW" if str(column).strip() == "GW" else str(column).strip().lower()
        for column in history.columns
    ]
    point_target = _target(history, POINT_TARGETS)
    if point_target is None or len(history) < 500:
        return _fallback_result(base, expanded, len(history))

    train = engineer_features(history).reset_index(drop=True)
    train[point_target] = pd.to_numeric(train[point_target], errors="coerce")
    train = train.dropna(subset=point_target).reset_index(drop=True)
    if len(train) < 500:
        return _fallback_result(base, expanded, len(train))

    start_target = _target(train, START_TARGETS)
    appearance_target = _target(train, APPEARANCE_TARGETS)
    minutes_target = _target(train, MINUTES_TARGETS)
    availability_train, availability_test = _time_split(train)
    validation_availability = pd.DataFrame(index=train.index)
    trained_availability: list[str] = []

    historical_start_probability = current["start_probability"].clip(0, 1).to_numpy()
    start_probability = historical_start_probability.copy()
    appearance_probability = np.maximum(
        start_probability,
        (current["minutes_per_appearance"] / 45.0).clip(0, 1).to_numpy(),
    )
    expected_minutes = current["minutes_per_appearance"].to_numpy()
    start_brier = appearance_brier = minutes_mae = None

    if start_target and train[start_target].nunique() >= 2:
        validation_model = _fit(
            _classifier(41),
            availability_train,
            pd.to_numeric(availability_train[start_target], errors="coerce").fillna(0).astype(int),
        )
        values = _probability(validation_model, availability_test)
        validation_availability.loc[availability_test.index, "start"] = values
        start_brier = float(
            brier_score_loss(
                pd.to_numeric(availability_test[start_target], errors="coerce").fillna(0).astype(int),
                values,
            )
        )
        model = _fit(
            _classifier(41),
            train,
            pd.to_numeric(train[start_target], errors="coerce").fillna(0).astype(int),
        )
        start_probability = _probability(model, current)
        trained_availability.append("start")

    if appearance_target and train[appearance_target].nunique() >= 2:
        validation_model = _fit(
            _classifier(43),
            availability_train,
            pd.to_numeric(availability_train[appearance_target], errors="coerce").fillna(0).astype(int),
        )
        values = _probability(validation_model, availability_test)
        validation_availability.loc[availability_test.index, "appearance"] = values
        appearance_brier = float(
            brier_score_loss(
                pd.to_numeric(availability_test[appearance_target], errors="coerce").fillna(0).astype(int),
                values,
            )
        )
        model = _fit(
            _classifier(43),
            train,
            pd.to_numeric(train[appearance_target], errors="coerce").fillna(0).astype(int),
        )
        appearance_probability = _probability(model, current)
        trained_availability.append("appearance")

    if minutes_target and pd.to_numeric(train[minutes_target], errors="coerce").notna().sum() >= 500:
        validation_model = _fit(
            _regressor(47, max_iter=120),
            availability_train,
            pd.to_numeric(availability_train[minutes_target], errors="coerce").fillna(0.0),
        )
        values = np.clip(validation_model.predict(availability_test[NUMERIC_FEATURES]), 0, 90)
        validation_availability.loc[availability_test.index, "minutes"] = values
        minutes_mae = float(
            mean_absolute_error(
                pd.to_numeric(availability_test[minutes_target], errors="coerce").fillna(0.0),
                values,
            )
        )
        model = _fit(
            _regressor(47, max_iter=120),
            train,
            pd.to_numeric(train[minutes_target], errors="coerce").fillna(0.0),
        )
        expected_minutes = np.clip(model.predict(current[NUMERIC_FEATURES]), 0, 90)
        trained_availability.append("minutes")

    start_probability, appearance_probability, expected_minutes = _live_availability(
        current,
        start_probability,
        appearance_probability,
        expected_minutes,
    )
    current["start_probability"] = start_probability
    current["minutes_per_appearance"] = expected_minutes
    current["_appearance_probability"] = appearance_probability
    current["_expected_minutes"] = expected_minutes

    means = np.full(len(current), np.nan)
    lowers = np.full(len(current), np.nan)
    uppers = np.full(len(current), np.nan)
    errors: list[tuple[float, int]] = []
    trained_positions: list[str] = []

    for position in POSITIONS:
        position_train = train[train["position"] == position]
        position_current = current[current["position"] == position]
        if position_current.empty:
            continue
        if len(position_train) < 350:
            values = _statistical_projection(position_current)
            means[position_current.index] = values
            lowers[position_current.index] = 0.60 * values
            uppers[position_current.index] = 1.55 * values
            continue

        point_train, point_test = _time_split(position_train)
        validation_direct = _fit(
            _regressor(53),
            point_train,
            pd.to_numeric(point_train[point_target], errors="coerce").fillna(0.0),
        )
        adjusted_test = point_test.copy()
        availability = validation_availability.reindex(point_test.index)
        if "start" in availability.columns:
            adjusted_test["start_probability"] = availability["start"].fillna(
                adjusted_test["start_probability"]
            )
        if "minutes" in availability.columns:
            adjusted_test["minutes_per_appearance"] = availability["minutes"].fillna(
                adjusted_test["minutes_per_appearance"]
            )
        test_appearance = availability.get(
            "appearance", pd.Series(np.nan, index=point_test.index)
        ).fillna(adjusted_test["start_probability"]).clip(0, 1)
        direct_values = np.maximum(validation_direct.predict(adjusted_test[NUMERIC_FEATURES]), 0)

        conditional_train = point_train
        if appearance_target:
            appeared = pd.to_numeric(point_train[appearance_target], errors="coerce").fillna(0) > 0
            if int(appeared.sum()) >= 200:
                conditional_train = point_train[appeared]
        validation_conditional = _fit(
            _regressor(59, max_iter=130),
            conditional_train,
            pd.to_numeric(conditional_train[point_target], errors="coerce").fillna(0.0),
        )
        conditional_values = np.maximum(
            validation_conditional.predict(adjusted_test[NUMERIC_FEATURES]), 0
        )
        validation_values = 0.25 * direct_values + 0.75 * test_appearance.to_numpy() * conditional_values
        errors.append(
            (
                float(
                    mean_absolute_error(
                        pd.to_numeric(point_test[point_target], errors="coerce").fillna(0.0),
                        validation_values,
                    )
                ),
                len(point_test),
            )
        )

        direct_model = _fit(
            _regressor(53, max_iter=155),
            position_train,
            pd.to_numeric(position_train[point_target], errors="coerce").fillna(0.0),
        )
        direct = np.maximum(direct_model.predict(position_current[NUMERIC_FEATURES]), 0)
        conditional_full = position_train
        if appearance_target:
            appeared = pd.to_numeric(position_train[appearance_target], errors="coerce").fillna(0) > 0
            if int(appeared.sum()) >= 250:
                conditional_full = position_train[appeared]
        conditional_model = _fit(
            _regressor(59, max_iter=145),
            conditional_full,
            pd.to_numeric(conditional_full[point_target], errors="coerce").fillna(0.0),
        )
        conditional = np.maximum(conditional_model.predict(position_current[NUMERIC_FEATURES]), 0)
        mean = 0.25 * direct + 0.75 * position_current["_appearance_probability"].to_numpy() * conditional

        lower_model = _fit(
            _regressor(61, max_iter=120, loss="quantile", quantile=0.10),
            position_train,
            pd.to_numeric(position_train[point_target], errors="coerce").fillna(0.0),
        )
        upper_model = _fit(
            _regressor(67, max_iter=120, loss="quantile", quantile=0.90),
            position_train,
            pd.to_numeric(position_train[point_target], errors="coerce").fillna(0.0),
        )
        lower = np.minimum(
            np.maximum(lower_model.predict(position_current[NUMERIC_FEATURES]), 0), mean
        )
        upper = np.maximum(
            np.maximum(upper_model.predict(position_current[NUMERIC_FEATURES]), 0), mean
        )

        unavailable = position_current["fixture_count"].to_numpy() <= 0
        if "lineup_status" in position_current.columns:
            unavailable |= (
                position_current["lineup_status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(["out", "not_in_squad"])
                .to_numpy()
            )
        mean[unavailable], lower[unavailable], upper[unavailable] = 0.0, 0.0, 0.0
        means[position_current.index] = mean
        lowers[position_current.index] = lower
        uppers[position_current.index] = upper
        trained_positions.append(position)

    missing = ~np.isfinite(means)
    if missing.any():
        values = _statistical_projection(current.loc[missing])
        means[missing], lowers[missing], uppers[missing] = values, 0.60 * values, 1.55 * values

    current["_mean"] = means * current["_scenario_weight"]
    current["_lower"] = lowers * current["_scenario_weight"]
    current["_upper"] = uppers * current["_scenario_weight"]
    grouped = current.groupby("_source_index", sort=False)
    output = base.copy()
    output["predicted_points"] = grouped["_mean"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p10"] = grouped["_lower"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p90"] = grouped["_upper"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p10"] = np.minimum(output["predicted_points_p10"], output["predicted_points"])
    output["predicted_points_p90"] = np.maximum(output["predicted_points_p90"], output["predicted_points"])
    output["point_uncertainty"] = output["predicted_points_p90"] - output["predicted_points_p10"]

    first = (
        current.sort_values(["_source_index", "_period_index"])
        .drop_duplicates("_source_index")
        .set_index("_source_index")
    )
    output["historical_start_probability"] = engineer_features(base)["start_probability"].to_numpy()
    output["start_probability"] = first["start_probability"].reindex(output.index).fillna(0.0)
    output["appearance_probability"] = first["_appearance_probability"].reindex(output.index).fillna(0.0)
    output["expected_minutes_next"] = first["_expected_minutes"].reindex(output.index).fillna(0.0)
    output["expected_minutes"] = grouped["_expected_minutes"].sum().reindex(output.index).fillna(0.0)
    output["horizon_fixture_count"] = grouped["fixture_count"].sum().reindex(output.index).fillna(0.0)

    relative_interval = output["point_uncertainty"] / (output["predicted_points"] + 1.0)
    interval_confidence = 1.0 / (1.0 + relative_interval.clip(lower=0.0))
    periods = grouped["_period_index"].max().reindex(output.index).fillna(0.0) + 1.0
    horizon_penalty = np.power(0.97, np.maximum(periods - 1.0, 0.0))
    output["projection_confidence"] = np.clip(
        output["appearance_probability"]
        * (0.45 + 0.55 * interval_confidence)
        * horizon_penalty,
        0.0,
        1.0,
    )
    output["prediction_mode"] = "Multitask ML ensemble"

    validation_mae = None
    if errors:
        validation_mae = float(
            sum(error * count for error, count in errors) / sum(count for _, count in errors)
        )
    detail = (
        f"ML availability ({', '.join(trained_availability) or 'fallback'}) + "
        f"direct, conditional and quantile point models for {', '.join(trained_positions) or 'none'}; "
        f"{horizon_mode}."
    )
    return PredictionResult(
        output,
        "multitask_ml",
        validation_mae,
        len(train),
        detail,
        start_brier,
        appearance_brier,
        minutes_mae,
    )


def _normalise_history(history: pd.DataFrame) -> pd.DataFrame:
    normalised = history.copy()
    normalised.columns = [
        "GW" if str(column).strip().upper() == "GW" else str(column).strip().lower()
        for column in normalised.columns
    ]
    return normalised


def _history_key(history: pd.DataFrame) -> str:
    fingerprint_columns = [
        column
        for column in [
            "name",
            "GW",
            "kickoff_time",
            "position",
            "future_points",
            "future_started",
            "future_appearance",
            "future_minutes",
            "price",
        ]
        if column in history.columns
    ]
    digest = hashlib.sha256()
    digest.update(str(history.shape).encode("utf-8"))
    digest.update("|".join(map(str, history.columns)).encode("utf-8"))
    if fingerprint_columns:
        hashed = pd.util.hash_pandas_object(
            history[fingerprint_columns], index=True, categorize=True
        )
        digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_pretrained_artifact() -> Optional[dict[str, Any]]:
    if not PRETRAINED_MODEL_PATH.exists():
        return None
    artifact = joblib.load(PRETRAINED_MODEL_PATH)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 3:
        return None
    return artifact


def _pretrained_artifact_for(history: pd.DataFrame) -> Optional[dict[str, Any]]:
    artifact = _load_pretrained_artifact()
    if artifact is None or artifact.get("history_key") != _history_key(history):
        return None
    bundle = artifact.get("bundle")
    if not isinstance(bundle, _ModelBundle):
        return None
    return artifact


def _compact_model_bundle(
    train: pd.DataFrame,
    point_target: str,
    start_target: Optional[str],
    appearance_target: Optional[str],
    minutes_target: Optional[str],
) -> _ModelBundle:
    fit_frame, test_frame = _time_split(train)
    validation_predictions = pd.DataFrame(index=test_frame.index)
    trained_availability: list[str] = []
    start_model: Optional[Pipeline] = None
    appearance_model: Optional[Pipeline] = None
    minutes_model: Optional[Pipeline] = None
    start_brier = appearance_brier = minutes_mae = None

    if start_target:
        y_fit = pd.to_numeric(fit_frame[start_target], errors="coerce").fillna(0).astype(int)
        if y_fit.nunique() >= 2:
            start_model = _fit(_availability_classifier(41, xgb_weight=0.50), fit_frame, y_fit)
            values = _probability(start_model, test_frame)
            validation_predictions["start"] = values
            y_test = pd.to_numeric(test_frame[start_target], errors="coerce").fillna(0).astype(int)
            start_brier = float(brier_score_loss(y_test, values))
            trained_availability.append("start")

    if appearance_target:
        y_fit = (
            pd.to_numeric(fit_frame[appearance_target], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        if y_fit.nunique() >= 2:
            appearance_model = _fit(_availability_classifier(43, xgb_weight=0.90), fit_frame, y_fit)
            values = _probability(appearance_model, test_frame)
            validation_predictions["appearance"] = values
            y_test = (
                pd.to_numeric(test_frame[appearance_target], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            appearance_brier = float(brier_score_loss(y_test, values))
            trained_availability.append("appearance")

    if minutes_target:
        y_fit = pd.to_numeric(fit_frame[minutes_target], errors="coerce")
        if int(y_fit.notna().sum()) >= 400:
            minutes_model = _fit(
                _availability_regressor(47, xgb_weight=0.90),
                fit_frame,
                y_fit.fillna(0.0),
            )
            values = np.clip(minutes_model.predict(test_frame[NUMERIC_FEATURES]), 0, 90)
            validation_predictions["minutes"] = values
            y_test = pd.to_numeric(test_frame[minutes_target], errors="coerce").fillna(0.0)
            minutes_mae = float(mean_absolute_error(y_test, values))
            trained_availability.append("minutes")

    adjusted_test = test_frame.copy()
    if "start" in validation_predictions:
        adjusted_test["start_probability"] = validation_predictions["start"]
    if "minutes" in validation_predictions:
        adjusted_test["minutes_per_appearance"] = validation_predictions["minutes"]

    point_models: dict[str, Pipeline] = {}
    starter_point_models: dict[str, Pipeline] = {}
    substitute_point_models: dict[str, Pipeline] = {}
    role_point_priors: dict[str, tuple[float, float]] = {}
    interval_offsets: dict[str, tuple[float, float]] = {}
    errors: list[tuple[float, int]] = []
    for position_index, position in enumerate(POSITIONS):
        position_fit = fit_frame[fit_frame["position"] == position]
        position_test = adjusted_test[adjusted_test["position"] == position]
        if len(position_fit) < 250 or position_test.empty:
            continue
        model = _fit(
            _point_regressor(53 + position_index, max_iter=70, xgb_weight=0.40),
            position_fit,
            pd.to_numeric(position_fit[point_target], errors="coerce").fillna(0.0),
        )
        validation_values = np.maximum(
            model.predict(position_test[NUMERIC_FEATURES]), 0.0
        )
        point_models[position] = model

        starter_rows = position_fit.iloc[0:0]
        substitute_rows = position_fit.iloc[0:0]
        if start_target:
            started = (
                pd.to_numeric(position_fit[start_target], errors="coerce")
                .fillna(0)
                .gt(0)
            )
            starter_rows = position_fit[started]
        if appearance_target and start_target:
            appeared = (
                pd.to_numeric(position_fit[appearance_target], errors="coerce")
                .fillna(0)
                .gt(0)
            )
            substitute_rows = position_fit[appeared & ~started]

        position_points = pd.to_numeric(
            position_fit[point_target], errors="coerce"
        ).fillna(0.0)
        starter_prior = float(
            pd.to_numeric(starter_rows.get(point_target), errors="coerce").mean()
        ) if not starter_rows.empty else float(position_points.mean())
        substitute_prior = float(
            pd.to_numeric(substitute_rows.get(point_target), errors="coerce").mean()
        ) if not substitute_rows.empty else min(starter_prior, 1.0)
        starter_prior = max(starter_prior if np.isfinite(starter_prior) else 0.0, 0.0)
        substitute_prior = max(
            substitute_prior if np.isfinite(substitute_prior) else 0.0, 0.0
        )
        role_point_priors[position] = (starter_prior, substitute_prior)

        starter_model: Optional[Pipeline] = None
        substitute_model: Optional[Pipeline] = None
        if len(starter_rows) >= 200:
            starter_model = _fit(
                _point_regressor(79 + position_index, max_iter=65, xgb_weight=0.35),
                starter_rows,
                pd.to_numeric(starter_rows[point_target], errors="coerce").fillna(0.0),
            )
            starter_point_models[position] = starter_model
        if len(substitute_rows) >= 150:
            substitute_model = _fit(
                _point_regressor(89 + position_index, max_iter=55, xgb_weight=0.30),
                substitute_rows,
                pd.to_numeric(substitute_rows[point_target], errors="coerce").fillna(0.0),
            )
            substitute_point_models[position] = substitute_model

        if start_target and appearance_target:
            test_start = validation_predictions.get(
                "start", position_test["start_probability"]
            ).reindex(position_test.index).fillna(position_test["start_probability"]).clip(0, 1)
            test_appearance = validation_predictions.get(
                "appearance", test_start
            ).reindex(position_test.index).fillna(test_start).clip(0, 1)
            test_start = np.minimum(test_start.to_numpy(), test_appearance.to_numpy())
            test_appearance_values = test_appearance.to_numpy()
            starter_values = (
                np.maximum(starter_model.predict(position_test[NUMERIC_FEATURES]), 0.0)
                if starter_model is not None
                else np.full(len(position_test), starter_prior)
            )
            substitute_values = (
                np.maximum(substitute_model.predict(position_test[NUMERIC_FEATURES]), 0.0)
                if substitute_model is not None
                else np.full(len(position_test), substitute_prior)
            )
            role_values = (
                test_start * starter_values
                + np.maximum(test_appearance_values - test_start, 0.0) * substitute_values
            )
            validation_values = 0.80 * validation_values + 0.20 * role_values

        actual = pd.to_numeric(
            position_test[point_target], errors="coerce"
        ).fillna(0.0).to_numpy()
        residuals = actual - validation_values
        low_offset, high_offset = np.quantile(residuals, [0.10, 0.90])
        interval_offsets[position] = (float(low_offset), float(high_offset))
        errors.append((float(mean_absolute_error(actual, validation_values)), len(actual)))

    validation_mae = None
    if errors:
        validation_mae = float(
            sum(error * count for error, count in errors)
            / sum(count for _, count in errors)
        )
    return _ModelBundle(
        training_rows=len(train),
        start_model=start_model,
        appearance_model=appearance_model,
        minutes_model=minutes_model,
        point_models=point_models,
        starter_point_models=starter_point_models,
        substitute_point_models=substitute_point_models,
        role_point_priors=role_point_priors,
        interval_offsets=interval_offsets,
        validation_mae=validation_mae,
        validation_start_brier=start_brier,
        validation_appearance_brier=appearance_brier,
        validation_minutes_mae=minutes_mae,
        trained_availability=trained_availability,
    )


def _get_compact_bundle(
    history: pd.DataFrame,
    train: pd.DataFrame,
    point_target: str,
    start_target: Optional[str],
    appearance_target: Optional[str],
    minutes_target: Optional[str],
) -> _ModelBundle:
    key = _history_key(history)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        bundle = _compact_model_bundle(
            train,
            point_target,
            start_target,
            appearance_target,
            minutes_target,
        )
        while len(_MODEL_CACHE) >= 2:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = bundle
        return bundle


def _get_price_model(history: pd.DataFrame, training: pd.DataFrame) -> Pipeline:
    key = f"{_history_key(history)}:price"
    cached = _PRICE_CACHE.get(key)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _PRICE_CACHE.get(key)
        if cached is not None:
            return cached
        model = _fit(
            _regressor(73, max_iter=55),
            training,
            training["price"],
        )
        while len(_PRICE_CACHE) >= 2:
            _PRICE_CACHE.pop(next(iter(_PRICE_CACHE)))
        _PRICE_CACHE[key] = model
        return model


def estimate_missing_prices_ml(
    players: pd.DataFrame,
    history: Optional[pd.DataFrame],
    budget: float,
    squad_size: int,
) -> pd.DataFrame:
    result = players.copy().reset_index(drop=True)
    if "price" not in result.columns:
        result["price"] = np.nan
    if "price_source" not in result.columns:
        result["price_source"] = "Missing"
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    missing = result["price"].isna() | (result["price"] <= 0)
    if not missing.any() or history is None or history.empty:
        return result

    normalised = _normalise_history(history)
    if "price" not in normalised.columns:
        return result
    artifact = _pretrained_artifact_for(normalised)
    model = artifact.get("price_model") if artifact is not None else None
    if not isinstance(model, Pipeline):
        training = engineer_features(normalised).reset_index(drop=True)
        training["price"] = pd.to_numeric(training["price"], errors="coerce")
        training = training.dropna(subset=["price"]).reset_index(drop=True)
        if len(training) < 500:
            return result
        model = _get_price_model(normalised, training)
    current = engineer_features(result).reset_index(drop=True)
    predicted = np.full(len(result), np.nan)
    predicted[missing.to_numpy()] = model.predict(
        current.loc[missing, NUMERIC_FEATURES]
    )
    usable = missing.to_numpy() & np.isfinite(predicted)
    if usable.any():
        median = max(float(np.nanmedian(predicted[usable])), 0.1)
        desired = max(float(budget) / max(int(squad_size), 1) * 0.82, 3.5)
        scale = float(np.clip(desired / median, 0.80, 1.25))
        result.loc[usable, "price"] = np.round(
            np.clip(predicted[usable] * scale, 3.5, 15.0), 1
        )
        result.loc[usable, "price_source"] = "ML-estimated price — not official"
    return result




@lru_cache(maxsize=1)
def _load_xp_calibrator() -> Optional[dict[str, Any]]:
    """Load the lightweight FPL xP residual calibrator if installed."""
    if not XP_CALIBRATOR_PATH.exists():
        return None
    try:
        artifact = joblib.load(XP_CALIBRATOR_PATH)
    except Exception:
        return None
    if not isinstance(artifact, dict):
        return None
    if artifact.get("kind") != "fpl_xp_residual_calibrator":
        return None
    if artifact.get("schema_version") != 1:
        return None
    if artifact.get("model") is None:
        return None
    return artifact


def _calibrator_frame(frame: pd.DataFrame, raw_xp: np.ndarray) -> pd.DataFrame:
    """Build the exact leakage-safe feature view expected by the calibrator."""
    view = frame.copy()
    view["raw_xp"] = np.asarray(raw_xp, dtype=float)
    defaults = {
        "price": 5.5,
        "minutes_per_appearance": 0.0,
        "start_probability": 0.0,
        "goals_per90": 0.0,
        "assists_per90": 0.0,
        "xg_per90": 0.0,
        "xa_per90": 0.0,
        "form": 0.0,
        "bonus_per90": 0.0,
        "bps_per90": 0.0,
        "ict_index": 0.0,
        "selected_by_percent": 0.0,
        "chance_playing": 1.0,
        "fixture_difficulty": 3.0,
        "home": 0.5,
        "fixture_count": 1.0,
        "team_strength": 0.5,
        "team_form_points": 1.5,
        "team_attack_form": 1.35,
        "team_defence_form": 1.35,
        "opponent_strength": 0.5,
        "rest_days": 7.0,
        "is_gk": 0.0,
        "is_def": 0.0,
        "is_mid": 0.0,
        "is_fwd": 0.0,
    }
    for column in XP_CALIBRATOR_FEATURES:
        if column == "raw_xp":
            continue
        if column not in view.columns:
            view[column] = defaults.get(column, 0.0)
        view[column] = pd.to_numeric(view[column], errors="coerce").fillna(
            defaults.get(column, 0.0)
        )
    view["raw_xp"] = pd.to_numeric(view["raw_xp"], errors="coerce").fillna(0.0)
    return view[XP_CALIBRATOR_FEATURES]


def _calibrate_xp(
    raw_xp: np.ndarray,
    frame: pd.DataFrame,
    active: np.ndarray,
) -> np.ndarray:
    """Calibrate PL xP without affecting leagues that lack Official FPL xP.

    The artifact predicts a residual around historical Official FPL expected
    points. Corrections are deliberately capped so one calibration layer cannot
    create a new extreme ranking by itself.
    """
    values = np.maximum(np.asarray(raw_xp, dtype=float), 0.0)
    artifact = _load_xp_calibrator()
    active = np.asarray(active, dtype=bool)
    if artifact is None or not active.any():
        return values
    try:
        features = _calibrator_frame(frame, values)
        correction = np.asarray(artifact["model"].predict(features), dtype=float)
        limit = float(artifact.get("correction_clip", 1.5))
        correction = np.clip(correction, -limit, limit)
        values[active] = np.maximum(values[active] + correction[active], 0.0)
    except Exception:
        # Prediction must remain fail-soft. The four-source model still works
        # even if the optional calibrator artifact is missing/incompatible.
        return np.maximum(np.asarray(raw_xp, dtype=float), 0.0)
    return values


def _calibrate_xp_gameweek(
    raw_xp: np.ndarray,
    frame: pd.DataFrame,
    active: np.ndarray,
) -> np.ndarray:
    """Apply the FPL residual calibrator once per player/gameweek, not once per fixture.

    Historical Official FPL ``xP`` is a gameweek total and is repeated on each
    raw fixture row in double gameweeks. Calibrating each scenario separately
    therefore double-counts the correction. This helper aggregates period-0
    scenarios, applies one correction to the gameweek total, then distributes
    that correction back across the scenarios in proportion to their raw xP.
    Future horizon periods are intentionally left model-only because ``ep_next``
    does not forecast them.
    """
    values = np.maximum(np.asarray(raw_xp, dtype=float), 0.0)
    active = np.asarray(active, dtype=bool)
    artifact = _load_xp_calibrator()
    if artifact is None or not active.any():
        return values

    if "_source_index" not in frame.columns or "_period_index" not in frame.columns:
        return _calibrate_xp(values, frame, active)

    result = values.copy()
    source_ids = pd.to_numeric(frame["_source_index"], errors="coerce")
    periods = pd.to_numeric(frame["_period_index"], errors="coerce").fillna(0).astype(int)

    matchup_columns = ["fixture_difficulty", "home", "opponent_strength", "rest_days"]

    try:
        for source_id in source_ids.dropna().unique():
            mask = source_ids.eq(source_id) & periods.eq(0)
            idxs = np.flatnonzero(mask.to_numpy())
            if len(idxs) == 0 or not active[idxs].any():
                continue

            raw_total = float(np.sum(values[idxs]))
            representative = frame.iloc[[idxs[0]]].copy()
            representative["fixture_count"] = float(len(idxs))
            for column in matchup_columns:
                if column in frame.columns:
                    representative[column] = pd.to_numeric(
                        frame.iloc[idxs][column], errors="coerce"
                    ).mean()

            features = _calibrator_frame(representative, np.asarray([raw_total], dtype=float))
            correction = float(artifact["model"].predict(features)[0])
            limit = float(artifact.get("correction_clip", 1.5))
            correction = float(np.clip(correction, -limit, limit))
            calibrated_total = max(raw_total + correction, 0.0)

            if raw_total > 1e-9:
                shares = np.maximum(values[idxs], 0.0)
                shares = shares / max(float(shares.sum()), 1e-9)
            else:
                shares = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
            result[idxs] = np.maximum(values[idxs] + (calibrated_total - raw_total) * shares, 0.0)
    except Exception:
        return values

    return result

def _restore_base_provider_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the primary-provider view (Official FPL for Premier League).

    SoccerData enrichment preserves base_* columns before modifying features.
    This restores those values so the base source gets its own point estimate.
    """
    view = frame.copy()

    for feature, base_col in [
        ("start_probability", "base_start_probability"),
        ("team_strength", "base_team_strength"),
        ("opponent_strength", "base_opponent_strength"),
    ]:
        if base_col in view.columns:
            base_values = pd.to_numeric(view[base_col], errors="coerce")
            view[feature] = base_values.fillna(pd.to_numeric(view[feature], errors="coerce"))

    minutes = pd.to_numeric(view.get("minutes", 0.0), errors="coerce").fillna(0.0).clip(lower=1.0)
    for total_col, per90_col, base_col in [
        ("xg", "xg_per90", "base_xg"),
        ("xa", "xa_per90", "base_xa"),
    ]:
        if base_col in view.columns:
            values = pd.to_numeric(view[base_col], errors="coerce")
            view[total_col] = values.fillna(pd.to_numeric(view[total_col], errors="coerce").fillna(0.0))
            view[per90_col] = 90.0 * pd.to_numeric(view[total_col], errors="coerce").fillna(0.0) / minutes

    return view


def _blend_ml_start_probability(
    frame: pd.DataFrame,
    ml_start: np.ndarray,
    provider_start: np.ndarray,
) -> np.ndarray:
    """Blend availability without erasing competition-specific role evidence.

    The bundled classifier is trained on Premier League/FPL rows.  Its score is
    useful as a cross-league correction, but it is not calibrated strongly
    enough to replace a live league provider's starts-per-team-match prior.  A
    55% model weight made every opening-round La Liga player fall below 60%,
    including players who had just started.  Official FPL keeps the validated
    55/45 blend; other providers keep 75% of their league-specific role prior.
    """
    model_values = np.clip(np.asarray(ml_start, dtype=float), 0.0, 1.0)
    provider_values = np.clip(np.asarray(provider_start, dtype=float), 0.0, 1.0)
    if len(model_values) != len(provider_values):
        raise ValueError("ML and provider start-probability arrays must have equal length.")

    provider_weight = 0.45 if "official_fpl_xp" in frame.columns else 0.75
    return np.clip(
        (1.0 - provider_weight) * model_values
        + provider_weight * provider_values,
        0.0,
        1.0,
    )


def _availability_without_cross_source(
    bundle: _ModelBundle,
    frame: pd.DataFrame,
    *,
    use_espn: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict availability for one source view without accidental cross-source fusion."""
    historical_start = pd.to_numeric(
        frame["start_probability"], errors="coerce"
    ).fillna(0.0).clip(0, 1).to_numpy(dtype=float)

    start = historical_start.copy()
    appearance = np.maximum(
        start,
        (pd.to_numeric(frame["minutes_per_appearance"], errors="coerce")
         .fillna(0.0).clip(0, 90) / 45.0).clip(0, 1).to_numpy(dtype=float),
    )
    minutes = pd.to_numeric(
        frame["minutes_per_appearance"], errors="coerce"
    ).fillna(0.0).clip(0, 90).to_numpy(dtype=float)

    if bundle.start_model is not None:
        ml_start = _probability(bundle.start_model, frame)
        start = _blend_ml_start_probability(frame, ml_start, historical_start)

        matches = pd.to_numeric(
            frame.get("team_matches_observed", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        cap = np.clip(np.maximum(0.55, historical_start + 0.30), 0.0, 0.90)
        guarded = matches >= 5.0
        start[guarded] = np.minimum(start[guarded], cap[guarded])

    if bundle.appearance_model is not None:
        appearance = _probability(bundle.appearance_model, frame)

    if bundle.minutes_model is not None:
        minutes = np.clip(
            bundle.minutes_model.predict(frame[NUMERIC_FEATURES]), 0.0, 90.0
        )

    start, appearance = blend_preseason_availability_prior(
        frame, start, appearance
    )
    appearance = np.maximum(appearance, start)

    # Predicted-lineup consensus is independent of FPL/ESPN/Understat/ClubElo,
    # so every source xP should respect it. This keeps Total xP consistent with
    # the displayed start probability. Do NOT treat ESPN recent_minutes as a
    # separate lineup vote: merge_recent_lineup_history leaves source_count at 0.
    if "lineup_consensus_probability" in frame.columns:
        consensus = pd.to_numeric(
            frame["lineup_consensus_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        source_counts = pd.to_numeric(
            frame.get("lineup_source_count", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        valid_consensus = np.isfinite(consensus) & (source_counts > 0.0)
        consensus_weight = np.clip(0.15 + 0.10 * source_counts, 0.0, 0.75)
        start[valid_consensus] = (
            (1.0 - consensus_weight[valid_consensus]) * start[valid_consensus]
            + consensus_weight[valid_consensus] * np.clip(consensus[valid_consensus], 0.0, 1.0)
        )
        appearance[valid_consensus] = np.maximum(
            appearance[valid_consensus], start[valid_consensus]
        )

        if "lineup_expected_minutes" in frame.columns:
            consensus_minutes = pd.to_numeric(
                frame["lineup_expected_minutes"], errors="coerce"
            ).to_numpy(dtype=float)
            valid_minutes = valid_consensus & np.isfinite(consensus_minutes)
            minutes[valid_minutes] = (
                0.55 * np.clip(minutes[valid_minutes], 0.0, 90.0)
                + 0.45 * np.clip(consensus_minutes[valid_minutes], 0.0, 90.0)
            )

    # ESPN gets a genuinely separate availability/minutes signal.
    if use_espn and "recent_start_rate" in frame.columns:
        recent_start = pd.to_numeric(
            frame["recent_start_rate"], errors="coerce"
        ).to_numpy(dtype=float)
        samples = pd.to_numeric(
            frame.get("recent_lineup_matches", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)

        valid = np.isfinite(recent_start) & (samples > 0)
        weight = np.clip(0.15 + 0.08 * samples, 0.20, 0.65)
        start[valid] = (
            (1.0 - weight[valid]) * start[valid]
            + weight[valid] * np.clip(recent_start[valid], 0.0, 1.0)
        )

        if "recent_minutes" in frame.columns:
            recent_minutes = pd.to_numeric(
                frame["recent_minutes"], errors="coerce"
            ).to_numpy(dtype=float)
            valid_minutes = np.isfinite(recent_minutes) & valid
            minutes[valid_minutes] = (
                0.55 * minutes[valid_minutes]
                + 0.45 * np.clip(recent_minutes[valid_minutes], 0.0, 90.0)
            )
        appearance = np.maximum(appearance, start)

    chance = pd.to_numeric(
        frame.get("chance_playing", pd.Series(1.0, index=frame.index)),
        errors="coerce",
    ).fillna(1.0).clip(0, 1).to_numpy(dtype=float)
    fixture = (
        pd.to_numeric(
            frame.get("fixture_count", pd.Series(1.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float) > 0
    ).astype(float)

    start = np.clip(start, 0, 1) * chance * fixture
    appearance = np.clip(appearance, 0, 1) * chance * fixture
    minutes = np.clip(minutes, 0, 90) * chance * fixture

    if "lineup_status" in frame.columns:
        status = frame["lineup_status"].fillna("").astype(str).str.lower()
        starter = status.eq("starter").to_numpy()
        bench = status.eq("bench").to_numpy()
        out = status.isin(["out", "not_in_squad"]).to_numpy()

        start[starter], appearance[starter] = 1.0, 1.0
        minutes[starter] = np.maximum(minutes[starter], 65.0)
        start[bench], appearance[bench] = 0.0, 1.0
        minutes[bench] = np.clip(minutes[bench], 8.0, 35.0)
        start[out], appearance[out], minutes[out] = 0.0, 0.0, 0.0

    return start, appearance, minutes



@lru_cache(maxsize=1)
def _load_point_v2_artifact() -> Optional[dict[str, Any]]:
    """Load the independent v7 fantasy-point model if installed.

    This artifact is intentionally separate from ``fpl_multitask_bundle`` so the
    user's StatsBomb-enhanced start/appearance/minutes models remain untouched.
    """
    if not POINT_V2_PATH.exists():
        return None
    try:
        artifact = joblib.load(POINT_V2_PATH)
    except Exception:
        return None
    if not isinstance(artifact, dict):
        return None
    if artifact.get("kind") != "fpl_points_v2" or artifact.get("schema_version") != 1:
        return None
    if not isinstance(artifact.get("models"), dict) or not artifact.get("features"):
        return None
    return artifact


def _point_v2_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    defaults = {
        "price": 5.5,
        "minutes_per_appearance": 0.0,
        "start_probability": 0.0,
        "goals_per90": 0.0,
        "assists_per90": 0.0,
        "clean_sheets_per90": 0.0,
        "saves_per90": 0.0,
        "form": 0.0,
        "xg_per90": 0.0,
        "xa_per90": 0.0,
        "xgc_per90": 0.0,
        "threat_per90": 0.0,
        "creativity_per90": 0.0,
        "influence_per90": 0.0,
        "defensive_contribution_per90": 0.0,
        "cbi_per90": 0.0,
        "recoveries_per90": 0.0,
        "tackles_per90": 0.0,
        "cards_per90": 0.0,
        "bonus_per90": 0.0,
        "bps_per90": 0.0,
        "ict_index": 0.0,
        "chance_playing": 1.0,
        "fixture_difficulty": 3.0,
        "home": 0.5,
        "fixture_count": 1.0,
        "team_strength": 0.5,
        "team_form_points": 1.5,
        "team_attack_form": 1.35,
        "team_defence_form": 1.35,
        "opponent_strength": 0.5,
        "rest_days": 7.0,
    }
    view = frame.copy()
    for column in features:
        if column not in view.columns:
            view[column] = defaults.get(column, 0.0)
        view[column] = pd.to_numeric(view[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(defaults.get(column, 0.0))
    return view[features]


def _point_v2_predict(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """Return the installed FPL point-model predictions for Premier League source views.

    v8 artifacts are trained without Official FPL ``ep_next`` as an input; the
    official number remains a separate Base-branch prior. Older schema-compatible
    artifacts continue to load during migration.
    """
    artifact = _load_point_v2_artifact()
    if artifact is None:
        return None
    # Keep non-PL modes on their existing model path. The FPL artifact is trained
    # specifically on FPL history and should not silently override other leagues.
    if "official_fpl_xp" not in frame.columns:
        return None
    features = list(artifact["features"])
    matrix = _point_v2_frame(frame, features)
    result = np.full(len(frame), np.nan, dtype=float)
    for position in POSITIONS:
        rows = frame[frame["position"].eq(position)]
        if rows.empty:
            continue
        model = artifact["models"].get(position)
        if model is None:
            continue
        values = np.asarray(model.predict(matrix.loc[rows.index]), dtype=float)
        result[rows.index.to_numpy()] = np.maximum(values, 0.0)
    return result

def _point_projection_for_source(
    bundle: _ModelBundle,
    frame: pd.DataFrame,
    start: np.ndarray,
    appearance: np.ndarray,
    minutes: np.ndarray,
) -> np.ndarray:
    """Translate one source view into fantasy expected points.

    Premier League rows use the installed point model, which is independent of FPL
    ``ep_next`` and was trained with price + leakage-safe opponent context.
    Other leagues retain the existing bundle point models.
    """
    source_frame = frame.copy()
    source_frame["start_probability"] = np.clip(start, 0, 1)
    source_frame["minutes_per_appearance"] = np.clip(minutes, 0, 90)

    v2 = _point_v2_predict(source_frame)
    if v2 is not None and np.isfinite(v2).all():
        unavailable = (
            pd.to_numeric(source_frame["fixture_count"], errors="coerce").fillna(0.0).to_numpy() <= 0
        ) | (np.asarray(appearance, dtype=float) <= 0.001)
        v2 = np.maximum(np.asarray(v2, dtype=float), 0.0)
        v2[unavailable] = 0.0
        return v2

    values = np.full(len(source_frame), np.nan)

    for position in POSITIONS:
        rows = source_frame[source_frame["position"] == position]
        if rows.empty:
            continue

        model = bundle.point_models.get(position)
        if model is None:
            predicted = _statistical_projection(rows)
        else:
            direct = np.maximum(model.predict(rows[NUMERIC_FEATURES]), 0.0)

            starter_prior, substitute_prior = bundle.role_point_priors.get(
                position, (float(np.mean(direct)), 1.0)
            )
            starter_model = bundle.starter_point_models.get(position)
            substitute_model = bundle.substitute_point_models.get(position)

            starter_points = (
                np.maximum(starter_model.predict(rows[NUMERIC_FEATURES]), 0.0)
                if starter_model is not None
                else np.full(len(rows), starter_prior)
            )
            substitute_points = (
                np.maximum(substitute_model.predict(rows[NUMERIC_FEATURES]), 0.0)
                if substitute_model is not None
                else np.full(len(rows), substitute_prior)
            )

            row_idx = rows.index.to_numpy()
            start_values = np.clip(start[row_idx], 0, 1)
            appearance_values = np.clip(appearance[row_idx], 0, 1)
            role_expected = (
                start_values * starter_points
                + np.maximum(appearance_values - start_values, 0.0) * substitute_points
            )
            predicted = 0.80 * direct + 0.20 * role_expected

        unavailable = (
            (pd.to_numeric(rows["fixture_count"], errors="coerce").fillna(0.0).to_numpy() <= 0)
            | (appearance[rows.index.to_numpy()] <= 0.001)
        )
        predicted = np.asarray(predicted, dtype=float)
        predicted[unavailable] = 0.0
        values[rows.index.to_numpy()] = predicted

    missing = ~np.isfinite(values)
    if missing.any():
        fallback = source_frame.loc[missing].copy()
        values[missing] = _statistical_projection(fallback)
    return np.maximum(values, 0.0)


def _four_source_point_ensemble(
    bundle: _ModelBundle,
    engineered: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Produce four source-conditioned xPs and combine them without a shared anchor.

    v7 architecture:
      * FPL/Base: independent v7 point model + Official FPL ep_next as a 50% prior
        for the NEXT gameweek only.
      * ESPN: independent v7 point model with ESPN start/minutes evidence; no ep_next.
      * Understat: independent v7 point model with Understat's own-season-window
        goals/assists/xG/xA rates; no ep_next. Goalkeepers get zero Understat weight.
      * ClubElo: independent v7 point model with Elo team/opponent context; no ep_next.

    This removes the old failure mode where all source xPs were merely small
    deltas around the same Official FPL number.
    """
    base = _restore_base_provider_features(engineered)

    # ---------- Base / Official FPL branch ----------
    base_start, base_app, base_minutes = _availability_without_cross_source(
        bundle, base, use_espn=False
    )
    base_view = base.copy()
    base_view["start_probability"] = np.clip(base_start, 0.0, 1.0)
    base_view["minutes_per_appearance"] = np.clip(base_minutes, 0.0, 90.0)
    xp_base_model = _point_projection_for_source(
        bundle, base_view, base_start, base_app, base_minutes
    )

    xp_base_model = blend_preseason_role_prior(
        base_view,
        xp_base_model,
        base_start,
        base_app,
    )

    official_total = pd.to_numeric(
        engineered.get("official_fpl_xp", pd.Series(np.nan, index=engineered.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    # ep_next is a gameweek total. Allocate only to period 0 and preserve DGW total.
    official = np.full(len(engineered), np.nan, dtype=float)
    if "_source_index" in engineered.columns and "_period_index" in engineered.columns:
        source_ids = pd.to_numeric(engineered["_source_index"], errors="coerce")
        periods = pd.to_numeric(engineered["_period_index"], errors="coerce").fillna(0).astype(int)
        for source_id in source_ids.dropna().unique():
            mask = source_ids.eq(source_id) & periods.eq(0)
            row_idx = np.flatnonzero(mask.to_numpy())
            if len(row_idx) == 0:
                continue
            totals = official_total[row_idx]
            finite = totals[np.isfinite(totals) & (totals >= 0.0)]
            if len(finite) == 0:
                continue
            total_xp = float(finite[0])
            shares = np.maximum(xp_base_model[row_idx], 0.05)
            shares = shares / max(float(shares.sum()), 1e-9)
            official[row_idx] = total_xp * shares
    else:
        official = official_total.copy()

    official_valid = np.isfinite(official) & (official >= 0.0)
    artifact = _load_point_v2_artifact()
    # Historical Vaastav ``xP`` is not guaranteed to be a pre-deadline snapshot.
    # Only permit a non-zero Official-FPL prior when the artifact explicitly says
    # it was calibrated on timestamped pre-deadline forecasts. v8.1 therefore
    # defaults to model-only Base xP instead of importing a possibly mis-timed prior.
    prior_is_validated = bool(artifact.get("official_prior_predeadline_validated", False)) if artifact else False
    official_blend = float(artifact.get("official_base_blend", 0.0)) if prior_is_validated else 0.0
    official_blend = float(np.clip(official_blend, 0.0, 0.65))
    xp_base = np.where(
        official_valid,
        (1.0 - official_blend) * xp_base_model + official_blend * official,
        xp_base_model,
    )

    # ---------- ESPN branch ----------
    espn = base.copy()
    espn_start, espn_app, espn_minutes = _availability_without_cross_source(
        bundle, espn, use_espn=True
    )
    espn["start_probability"] = np.clip(espn_start, 0.0, 1.0)
    espn["minutes_per_appearance"] = np.clip(espn_minutes, 0.0, 90.0)
    xp_espn = _point_projection_for_source(
        bundle, espn, espn_start, espn_app, espn_minutes
    )
    espn_sample = pd.to_numeric(
        engineered.get("recent_lineup_matches", pd.Series(0.0, index=engineered.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    espn_present = pd.to_numeric(
        engineered.get("soccerdata_espn", pd.Series(False, index=engineered.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float) > 0
    espn_stale = pd.Series(
        engineered.get("espn_is_previous_season", False), index=engineered.index
    ).fillna(False).astype(bool).to_numpy()
    espn_conf = (
        np.clip(espn_sample / 5.0, 0.0, 1.0)
        * espn_present.astype(float)
        * np.where(espn_stale, 0.50, 1.00)
    )

    # ---------- Understat branch ----------
    understat = base.copy()
    understat_present = pd.to_numeric(
        engineered.get("soccerdata_understat", pd.Series(False, index=engineered.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float) > 0
    understat_minutes = pd.to_numeric(
        engineered.get("understat_minutes", pd.Series(np.nan, index=engineered.index)),
        errors="coerce",
    )
    understat_minutes_safe = understat_minutes.clip(lower=1.0)

    for total_col, per90_col, source_col in [
        ("goals", "goals_per90", "understat_goals"),
        ("assists", "assists_per90", "understat_assists"),
        ("xg", "xg_per90", "understat_xg"),
        ("xa", "xa_per90", "understat_xa"),
    ]:
        if source_col not in engineered.columns:
            continue
        external = pd.to_numeric(engineered[source_col], errors="coerce")
        valid = (
            external.notna()
            & understat_minutes.notna()
            & understat_minutes.gt(0.0)
            & pd.Series(understat_present, index=understat.index)
        )
        understat.loc[valid, total_col] = external.loc[valid]
        understat.loc[valid, per90_col] = (
            90.0 * external.loc[valid] / understat_minutes_safe.loc[valid]
        )

    understat["start_probability"] = np.clip(base_start, 0.0, 1.0)
    understat["minutes_per_appearance"] = np.clip(base_minutes, 0.0, 90.0)
    xp_understat_model = _point_projection_for_source(
        bundle, understat, base_start, base_app, base_minutes
    )

    understat_sample = pd.to_numeric(
        engineered.get("understat_matches", pd.Series(0.0, index=engineered.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    understat_stale = pd.Series(
        engineered.get("understat_is_previous_season", False), index=engineered.index
    ).fillna(False).astype(bool).to_numpy()
    position = engineered["position"].astype(str).str.upper().to_numpy()
    # Understat's attacking player data is not a goalkeeper fantasy signal.
    # Defenders retain a reduced attacking-return signal; MID/FWD retain full signal.
    position_relevance = np.where(
        position == "GK", 0.0,
        np.where(position == "DEF", 0.65, 1.0),
    )
    understat_conf = (
        np.clip(understat_sample / 10.0, 0.0, 1.0)
        * understat_present.astype(float)
        * np.isfinite(understat_minutes.to_numpy(dtype=float)).astype(float)
        * (understat_minutes.to_numpy(dtype=float) > 0.0).astype(float)
        * np.where(understat_stale, 0.50, 1.00)
        * position_relevance
    )
    xp_understat = np.where(position_relevance > 0.0, xp_understat_model, np.nan)

    # ---------- ClubElo branch ----------
    elo = base.copy()
    elo_team = pd.to_numeric(
        engineered.get("clubelo_strength", pd.Series(np.nan, index=engineered.index)),
        errors="coerce",
    )
    elo_opp = pd.to_numeric(
        engineered.get("clubelo_opponent_strength", pd.Series(np.nan, index=engineered.index)),
        errors="coerce",
    )
    team_valid = elo_team.notna()
    opp_valid = elo_opp.notna()
    elo.loc[team_valid, "team_strength"] = elo_team.loc[team_valid].clip(0.0, 1.0)
    elo.loc[opp_valid, "opponent_strength"] = elo_opp.loc[opp_valid].clip(0.0, 1.0)
    elo.loc[opp_valid, "fixture_difficulty"] = (
        1.0 + 4.0 * elo_opp.loc[opp_valid].clip(0.0, 1.0)
    ).clip(1.0, 5.0)
    elo["start_probability"] = np.clip(base_start, 0.0, 1.0)
    elo["minutes_per_appearance"] = np.clip(base_minutes, 0.0, 90.0)
    xp_elo = _point_projection_for_source(
        bundle, elo, base_start, base_app, base_minutes
    )
    elo_present = pd.to_numeric(
        engineered.get("soccerdata_clubelo", pd.Series(False, index=engineered.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float) > 0
    elo_conf = elo_present.astype(float) * np.where(
        team_valid.to_numpy() & opp_valid.to_numpy(),
        1.0,
        np.where(team_valid.to_numpy(), 0.60, 0.0),
    )

    # ---------- Reliability-weighted late fusion ----------
    # Full-confidence priors: Base 50%, ESPN 20%, Understat 20%, ClubElo 10%.
    # Missing/irrelevant sources are zero-weighted and remaining weights renormalize.
    w_base = np.ones(len(engineered), dtype=float) * 1.00
    w_espn = 0.40 * espn_conf
    w_understat = 0.40 * understat_conf
    w_elo = 0.20 * elo_conf

    denominator = np.maximum(w_base + w_espn + w_understat + w_elo, 1e-9)
    understat_safe = np.where(np.isfinite(xp_understat), xp_understat, 0.0)
    ensemble = (
        w_base * xp_base
        + w_espn * xp_espn
        + w_understat * understat_safe
        + w_elo * xp_elo
    ) / denominator

    variance = (
        w_base * (xp_base - ensemble) ** 2
        + w_espn * (xp_espn - ensemble) ** 2
        + w_understat * (understat_safe - ensemble) ** 2
        + w_elo * (xp_elo - ensemble) ** 2
    ) / denominator
    disagreement = np.sqrt(np.maximum(variance, 0.0))

    return ensemble, {
        "base": xp_base,
        "espn": xp_espn,
        "understat": xp_understat,
        "clubelo": xp_elo,
        "w_base": w_base / denominator,
        "w_espn": w_espn / denominator,
        "w_understat": w_understat / denominator,
        "w_clubelo": w_elo / denominator,
    }, disagreement


def predict_players(
    players: pd.DataFrame,
    history: Optional[pd.DataFrame],
    horizon: int = 1,
) -> PredictionResult:
    base = players.copy().reset_index(drop=True)
    expanded, horizon_mode = _expand_scenarios(base, horizon)
    current = engineer_features(expanded).reset_index(drop=True)
    if history is None or history.empty:
        return _fallback_result(base, expanded, 0)

    normalised = _normalise_history(history)
    point_target = _target(normalised, POINT_TARGETS)
    if point_target is None or len(normalised) < 500:
        return _fallback_result(base, expanded, len(normalised))
    artifact = _pretrained_artifact_for(normalised)
    if artifact is not None:
        bundle = artifact["bundle"]
        deployment_mode = "pre-trained"
    else:
        train = engineer_features(normalised).reset_index(drop=True)
        train[point_target] = pd.to_numeric(train[point_target], errors="coerce")
        train = train.dropna(subset=[point_target]).reset_index(drop=True)
        if len(train) < 500:
            return _fallback_result(base, expanded, len(train))
        start_target = _target(train, START_TARGETS)
        appearance_target = _target(train, APPEARANCE_TARGETS)
        minutes_target = _target(train, MINUTES_TARGETS)
        bundle = _get_compact_bundle(
            normalised,
            train,
            point_target,
            start_target,
            appearance_target,
            minutes_target,
        )
        deployment_mode = "session-trained"

    # NEW: each live source gets its own fantasy-point projection first.
    source_ensemble, source_xp, source_disagreement = _four_source_point_ensemble(
        bundle, current
    )

    historical_start_probability = current["start_probability"].clip(0, 1).to_numpy()
    start_probability = historical_start_probability.copy()
    appearance_probability = np.maximum(
        start_probability,
        (current["minutes_per_appearance"] / 45.0).clip(0, 1).to_numpy(),
    )
    expected_minutes = current["minutes_per_appearance"].to_numpy()
    if bundle.start_model is not None:
        ml_start_probability = _probability(bundle.start_model, current)
        start_probability = _blend_ml_start_probability(
            current,
            ml_start_probability,
            historical_start_probability,
        )
        # Guard against the exact failure exposed by rotation players such as
        # Marmoush/Gabriel Jesus: a model score must not jump a low real-world
        # start prior to ~95% solely because of sparse player appearances.
        matches = pd.to_numeric(
            current.get("team_matches_observed", pd.Series(0.0, index=current.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        has_external_lineup = pd.to_numeric(
            current.get("lineup_source_count", pd.Series(0.0, index=current.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float) > 0
        enough_history = matches >= 5.0
        cap = np.clip(np.maximum(0.55, historical_start_probability + 0.30), 0.0, 0.90)
        guarded = enough_history & (~has_external_lineup)
        start_probability[guarded] = np.minimum(start_probability[guarded], cap[guarded])
    if bundle.appearance_model is not None:
        appearance_probability = _probability(bundle.appearance_model, current)
    if bundle.minutes_model is not None:
        expected_minutes = np.clip(
            bundle.minutes_model.predict(current[NUMERIC_FEATURES]), 0, 90
        )
    start_probability, appearance_probability = blend_preseason_availability_prior(
        current, start_probability, appearance_probability
    )
    appearance_probability = np.maximum(appearance_probability, start_probability)
    start_probability, appearance_probability, expected_minutes = _live_availability(
        current,
        start_probability,
        appearance_probability,
        expected_minutes,
    )
    current["start_probability"] = start_probability
    current["minutes_per_appearance"] = expected_minutes
    current["_appearance_probability"] = appearance_probability
    current["_expected_minutes"] = expected_minutes

    means = np.full(len(current), np.nan)
    lowers = np.full(len(current), np.nan)
    uppers = np.full(len(current), np.nan)
    trained_positions: list[str] = []
    for position in POSITIONS:
        position_current = current[current["position"] == position]
        if position_current.empty:
            continue
        model = bundle.point_models.get(position)
        if model is None:
            mean = _statistical_projection(position_current)
            lower, upper = 0.60 * mean, 1.55 * mean
        else:
            direct = np.maximum(
                model.predict(position_current[NUMERIC_FEATURES]), 0.0
            )
            starter_prior, substitute_prior = bundle.role_point_priors.get(
                position, (float(np.mean(direct)), 1.0)
            )
            starter_model = bundle.starter_point_models.get(position)
            substitute_model = bundle.substitute_point_models.get(position)
            starter_points = (
                np.maximum(starter_model.predict(position_current[NUMERIC_FEATURES]), 0.0)
                if starter_model is not None
                else np.full(len(position_current), starter_prior)
            )
            substitute_points = (
                np.maximum(substitute_model.predict(position_current[NUMERIC_FEATURES]), 0.0)
                if substitute_model is not None
                else np.full(len(position_current), substitute_prior)
            )
            start_values = position_current["start_probability"].clip(0, 1).to_numpy()
            appearance_values = position_current["_appearance_probability"].clip(0, 1).to_numpy()
            role_expected = (
                start_values * starter_points
                + np.maximum(appearance_values - start_values, 0.0) * substitute_points
            )
            mean = 0.80 * direct + 0.20 * role_expected
            low_offset, high_offset = bundle.interval_offsets[position]
            lower = np.minimum(np.maximum(mean + low_offset, 0.0), mean)
            upper = np.maximum(np.maximum(mean + high_offset, 0.0), mean)
            trained_positions.append(position)

        unavailable = (
            (position_current["fixture_count"].to_numpy() <= 0)
            | (position_current["_appearance_probability"].to_numpy() <= 0.001)
        )
        if "lineup_status" in position_current.columns:
            unavailable |= (
                position_current["lineup_status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(["out", "not_in_squad"])
                .to_numpy()
            )
        mean[unavailable], lower[unavailable], upper[unavailable] = 0.0, 0.0, 0.0
        means[position_current.index] = mean
        lowers[position_current.index] = lower
        uppers[position_current.index] = upper

    missing = ~np.isfinite(means)
    if missing.any():
        values = _statistical_projection(current.loc[missing])
        means[missing] = values
        lowers[missing] = 0.60 * values
        uppers[missing] = 1.55 * values

    # Final fantasy xP is now late-fused from four independent source views.
    # The original ML interval is retained as a baseline and widened when the
    # source-specific xP estimates disagree.
    base_low_gap = np.maximum(means - lowers, 0.0)
    base_high_gap = np.maximum(uppers - means, 0.0)
    ensemble_lower = np.maximum(
        source_ensemble - base_low_gap - 0.75 * source_disagreement, 0.0
    )
    ensemble_upper = np.maximum(
        source_ensemble + base_high_gap + 0.75 * source_disagreement,
        source_ensemble,
    )

    scenario_weight = current["_scenario_weight"].to_numpy(dtype=float)
    current["_mean"] = source_ensemble * scenario_weight
    current["_lower"] = ensemble_lower * scenario_weight
    current["_upper"] = ensemble_upper * scenario_weight

    # Persist each source's xP so the final result can be audited.
    current["_xp_base"] = source_xp["base"] * scenario_weight
    current["_xp_espn"] = np.where(
        source_xp["w_espn"] > 0, source_xp["espn"] * scenario_weight, np.nan
    )
    current["_xp_understat"] = np.where(
        source_xp["w_understat"] > 0,
        source_xp["understat"] * scenario_weight,
        np.nan,
    )
    current["_xp_clubelo"] = np.where(
        source_xp["w_clubelo"] > 0,
        source_xp["clubelo"] * scenario_weight,
        np.nan,
    )
    current["_w_base"] = source_xp["w_base"]
    current["_w_espn"] = source_xp["w_espn"]
    current["_w_understat"] = source_xp["w_understat"]
    current["_w_clubelo"] = source_xp["w_clubelo"]

    grouped = current.groupby("_source_index", sort=False)
    output = base.copy()
    output["predicted_points"] = grouped["_mean"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p10"] = grouped["_lower"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p90"] = grouped["_upper"].sum().reindex(output.index).fillna(0.0)
    output["predicted_points_p10"] = np.minimum(
        output["predicted_points_p10"], output["predicted_points"]
    )
    output["predicted_points_p90"] = np.maximum(
        output["predicted_points_p90"], output["predicted_points"]
    )
    output["point_uncertainty"] = (
        output["predicted_points_p90"] - output["predicted_points_p10"]
    )

    # Per-source expected points and effective weights used in final xP.
    output["xp_base_provider"] = grouped["_xp_base"].sum().reindex(output.index).fillna(0.0)
    output["xp_espn"] = grouped["_xp_espn"].sum(min_count=1).reindex(output.index)
    output["xp_understat"] = grouped["_xp_understat"].sum(min_count=1).reindex(output.index)
    output["xp_clubelo"] = grouped["_xp_clubelo"].sum(min_count=1).reindex(output.index)

    first = (
        current.sort_values(["_source_index", "_period_index"])
        .drop_duplicates("_source_index")
        .set_index("_source_index")
    )
    output["historical_start_probability"] = engineer_features(base)[
        "start_probability"
    ].to_numpy()
    output["start_probability"] = first["start_probability"].reindex(output.index).fillna(0.0)
    output["appearance_probability"] = first["_appearance_probability"].reindex(output.index).fillna(0.0)
    output["expected_minutes_next"] = first["_expected_minutes"].reindex(output.index).fillna(0.0)
    output["expected_minutes"] = grouped["_expected_minutes"].sum().reindex(output.index).fillna(0.0)
    output["horizon_fixture_count"] = grouped["fixture_count"].sum().reindex(output.index).fillna(0.0)
    output["xp_weight_base"] = first["_w_base"].reindex(output.index).fillna(1.0)
    output["xp_weight_espn"] = first["_w_espn"].reindex(output.index).fillna(0.0)
    output["xp_weight_understat"] = first["_w_understat"].reindex(output.index).fillna(0.0)
    output["xp_weight_clubelo"] = first["_w_clubelo"].reindex(output.index).fillna(0.0)

    relative_interval = output["point_uncertainty"] / (output["predicted_points"] + 1.0)
    interval_confidence = 1.0 / (1.0 + relative_interval.clip(lower=0.0))
    periods = grouped["_period_index"].max().reindex(output.index).fillna(0.0) + 1.0
    horizon_penalty = np.power(0.97, np.maximum(periods - 1.0, 0.0))
    output["projection_confidence"] = np.clip(
        output["appearance_probability"]
        * (0.45 + 0.55 * interval_confidence)
        * horizon_penalty,
        0.0,
        1.0,
    )
    point_artifact = _load_point_v2_artifact() or {}
    point_version = str(point_artifact.get("model_version") or "v7")
    training_season = str(point_artifact.get("training_season") or "legacy")
    prior_is_validated = bool(point_artifact.get("official_prior_predeadline_validated", False))
    official_prior = float(np.clip(point_artifact.get("official_base_blend", 0.0), 0.0, 0.65)) if prior_is_validated else 0.0
    output["prediction_mode"] = f"{point_version} independent-source late-fusion xP"

    detail = (
        f"{deployment_mode} availability ({', '.join(bundle.trained_availability) or 'fallback'}) + "
        f"{point_version} PL point model ({training_season}; leakage-safe opponent context) + "
        f"late-fusion xP: FPL/Base uses a {official_prior:.0%} official-next-GW prior only; "
        f"ESPN, Understat and ClubElo are independently source-conditioned; "
        f"Understat is disabled for GK and freshness-weighted elsewhere; "
        f"source-disagreement uncertainty; {horizon_mode}"
    )
    point_validation = point_artifact.get("validation", {}) if isinstance(point_artifact, dict) else {}
    displayed_point_mae = point_validation.get("model_only_mae", bundle.validation_mae)
    try:
        displayed_point_mae = float(displayed_point_mae) if displayed_point_mae is not None else None
    except (TypeError, ValueError):
        displayed_point_mae = bundle.validation_mae

    return PredictionResult(
        output,
        "multitask_ml",
        displayed_point_mae,
        bundle.training_rows,
        detail,
        bundle.validation_start_brier,
        bundle.validation_appearance_brier,
        bundle.validation_minutes_mae,
    )
