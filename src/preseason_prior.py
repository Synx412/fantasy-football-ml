from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

PRIOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fpl_preseason_role_prior_2025_26.csv"
)


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _aliases(full_name: str) -> set[str]:
    raw = unicodedata.normalize("NFKD", str(full_name or ""))
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    tokens = [re.sub(r"[^A-Za-z0-9]+", "", t).lower() for t in raw.split()]
    tokens = [t for t in tokens if len(t) >= 3]
    out = {_norm(full_name)}
    out.update(tokens)
    if tokens:
        surname = tokens[-1]
        out.add(surname)
        out.add(tokens[0][0] + surname)
        for token in tokens[1:-1]:
            if len(token) >= 4:
                out.add(token)
                out.add(tokens[0][0] + token)
    return {x for x in out if len(x) >= 3}


@lru_cache(maxsize=1)
def _prior_lookup() -> dict[tuple[str, str], dict[str, float]]:
    if not PRIOR_PATH.exists():
        return {}
    df = pd.read_csv(PRIOR_PATH)
    required = {
        "name", "position", "starter_ppg_est", "sub_ppg_est",
        "starts", "appearances", "season_points",
    }
    if not required.issubset(df.columns):
        return {}

    candidates: dict[tuple[str, str], list[dict[str, float]]] = {}
    for _, row in df.iterrows():
        position = str(row["position"]).upper()
        record = {
            "starter_ppg_est": float(row["starter_ppg_est"]),
            "sub_ppg_est": float(row["sub_ppg_est"]),
            "starts": float(row["starts"]),
            "appearances": float(row["appearances"]),
            "season_points": float(row["season_points"]),
        }
        for alias in _aliases(str(row["name"])):
            candidates.setdefault((alias, position), []).append(record)

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in candidates.items():
        unique = {
            (
                round(v["starter_ppg_est"], 6),
                round(v["sub_ppg_est"], 6),
                round(v["starts"], 4),
                round(v["appearances"], 4),
                round(v["season_points"], 4),
            )
            for v in values
        }
        if len(unique) == 1:
            lookup[key] = values[0]
    return lookup


def _team_match_evidence(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    observed = pd.to_numeric(
        frame.get("team_matches_observed", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    if "club" in frame.columns and "appearances" in frame.columns:
        club = frame["club"].fillna("").astype(str)
        apps = pd.to_numeric(frame["appearances"], errors="coerce").fillna(0.0)
        club_max = apps.groupby(club).transform("max").to_numpy(dtype=float)
    else:
        club_max = np.zeros(n, dtype=float)

    observed = np.where(np.isfinite(observed) & (observed >= 0.0), observed, club_max)
    if "appearances" in frame.columns:
        own_apps = pd.to_numeric(
            frame["appearances"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        stale_rollover = (own_apps <= 1.0) & (club_max <= 1.0) & (observed >= 20.0)
        observed[stale_rollover] = club_max[stale_rollover]

    return np.clip(observed, 0.0, 38.0)


def preseason_role_prior(
    frame: pd.DataFrame,
    start: np.ndarray,
    appearance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    prior_xp = np.full(n, np.nan, dtype=float)
    weight = np.zeros(n, dtype=float)

    if "official_fpl_xp" not in frame.columns or "name" not in frame.columns:
        return prior_xp, weight

    lookup = _prior_lookup()
    if not lookup:
        return prior_xp, weight

    starts_now = np.clip(np.asarray(start, dtype=float), 0.0, 1.0)
    apps_now = np.clip(np.asarray(appearance, dtype=float), starts_now, 1.0)

    difficulty = pd.to_numeric(
        frame.get("fixture_difficulty", pd.Series(3.0, index=frame.index)),
        errors="coerce",
    ).fillna(3.0).clip(1.0, 5.0).to_numpy(dtype=float)
    home = pd.to_numeric(
        frame.get("home", pd.Series(0.5, index=frame.index)),
        errors="coerce",
    ).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
    team_strength = pd.to_numeric(
        frame.get("team_strength", pd.Series(0.5, index=frame.index)),
        errors="coerce",
    ).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
    opp_strength = pd.to_numeric(
        frame.get("opponent_strength", pd.Series(0.5, index=frame.index)),
        errors="coerce",
    ).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
    matches = _team_match_evidence(frame)

    for j, (_, row) in enumerate(frame.iterrows()):
        position = str(row.get("position", "")).upper()
        record = lookup.get((_norm(row.get("name")), position))
        if record is None:
            continue

        established = float(np.clip(record["starts"] / 15.0, 0.20, 1.0))
        role_xp = (
            starts_now[j] * max(record["starter_ppg_est"], 0.0)
            + max(apps_now[j] - starts_now[j], 0.0)
            * max(record["sub_ppg_est"], 0.0)
        )

        fixture_factor = (
            1.0
            + 0.07 * (3.0 - difficulty[j])
            + 0.04 * (2.0 * home[j] - 1.0)
            + 0.05 * (team_strength[j] - opp_strength[j])
        )
        fixture_factor = float(np.clip(fixture_factor, 0.82, 1.18))

        prior_xp[j] = max(role_xp * fixture_factor, 0.0)
        weight[j] = 0.50 * np.exp(-matches[j] / 3.0) * established

    return prior_xp, np.clip(weight, 0.0, 0.50)


def blend_preseason_role_prior(
    frame: pd.DataFrame,
    model_xp: np.ndarray,
    start: np.ndarray,
    appearance: np.ndarray,
) -> np.ndarray:
    model = np.maximum(np.asarray(model_xp, dtype=float), 0.0)
    prior, weight = preseason_role_prior(frame, start, appearance)
    valid = np.isfinite(prior) & (weight > 0.0)
    if not valid.any():
        return model

    out = model.copy()
    out[valid] = (
        (1.0 - weight[valid]) * model[valid]
        + weight[valid] * np.maximum(prior[valid], 0.0)
    )
    return np.maximum(out, 0.0)
