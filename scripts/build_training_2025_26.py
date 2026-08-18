from __future__ import annotations

"""Build leakage-safe 2025/26 FPL training rows for the v8 model.

The script can run in Colab or locally. By default it downloads the public
2025/26 merged gameweek file from the vaastav/Fantasy-Premier-League dataset,
then writes:

    data/fpl_2025_26_merged_gw.csv
    data/fpl_multitask_training_2025_26.csv

Every feature is constructed from information available before the target GW
(except schedule/price/ownership fields that are known before kickoff). Current
match outcomes are used only as labels.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/2025-26/gws/merged_gw.csv"
)
DEFAULT_RAW = ROOT / "data" / "fpl_2025_26_merged_gw.csv"
DEFAULT_OUTPUT = ROOT / "data" / "fpl_multitask_training_2025_26.csv"
POSITIONS = {"GK", "DEF", "MID", "FWD"}

RAW_REQUIRED = {
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "element",
    "expected_assists",
    "expected_goals",
    "fixture",
    "goals_scored",
    "ict_index",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "red_cards",
    "saves",
    "selected",
    "starts",
    "team_a_score",
    "team_h_score",
    "total_points",
    "value",
    "was_home",
    "yellow_cards",
    "GW",
}

CUMULATIVE_SOURCE = {
    "minutes": "minutes",
    "appearances": "appeared",
    "starts": "starts",
    "goals": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "saves": "saves",
    "xg": "expected_goals",
    "xa": "expected_assists",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "bonus": "bonus",
    "bps": "bps",
    "ict_index": "ict_index",
    "threat": "threat",
    "creativity": "creativity",
    "influence": "influence",
    "xgc": "expected_goals_conceded",
    "defensive_contribution": "defensive_contribution",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "recoveries": "recoveries",
    "tackles": "tackles",
}


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"})


def download_raw(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        headers={"User-Agent": "FantasyXI-v8-training/1.0"},
        timeout=120,
    )
    response.raise_for_status()
    if len(response.content) < 100_000:
        raise RuntimeError("Downloaded training source is unexpectedly small.")
    destination.write_bytes(response.content)
    return destination


def load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    missing = RAW_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Raw 2025/26 file is missing columns: {', '.join(sorted(missing))}")

    raw = raw.copy()
    raw["position"] = raw["position"].astype(str).str.upper().str.strip()
    # Assistant Manager is a fantasy-game row, not a footballer. Never let it
    # leak into the MID model.
    raw = raw[raw["position"].isin(POSITIONS)].copy()
    raw["was_home"] = _boolean(raw["was_home"])
    raw["kickoff_time"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True)
    raw["GW"] = pd.to_numeric(raw["GW"], errors="coerce")
    raw["fixture"] = pd.to_numeric(raw["fixture"], errors="coerce")
    raw = raw.dropna(subset=["name", "team", "kickoff_time", "GW", "fixture"])
    raw["GW"] = raw["GW"].astype(int)
    raw["fixture"] = raw["fixture"].astype(int)

    numeric = set(RAW_REQUIRED) - {
        "name", "position", "team", "kickoff_time", "opponent_team", "was_home"
    }
    numeric |= set(CUMULATIVE_SOURCE.values())
    _numeric(raw, numeric)
    raw["appeared"] = (raw["minutes"] > 0).astype(float)
    raw["starts"] = raw["starts"].clip(0.0, 1.0)
    return raw.sort_values(["GW", "kickoff_time", "fixture", "name"]).reset_index(drop=True)


def fixture_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (gw, fixture_id), group in raw.groupby(["GW", "fixture"], sort=True):
        home = group.loc[group["was_home"], "team"].dropna().astype(str).unique()
        away = group.loc[~group["was_home"], "team"].dropna().astype(str).unique()
        if len(home) != 1 or len(away) != 1:
            continue
        hs = pd.to_numeric(group["team_h_score"], errors="coerce").dropna()
        aas = pd.to_numeric(group["team_a_score"], errors="coerce").dropna()
        if hs.empty or aas.empty:
            continue
        rows.append(
            {
                "GW": int(gw),
                "fixture": int(fixture_id),
                "kickoff_time": group["kickoff_time"].min(),
                "home_team": str(home[0]),
                "away_team": str(away[0]),
                "home_score": float(hs.iloc[0]),
                "away_score": float(aas.iloc[0]),
            }
        )
    fixtures = pd.DataFrame(rows)
    if fixtures.empty:
        raise RuntimeError("Could not reconstruct fixtures from merged gameweek data.")
    return fixtures.sort_values(["GW", "kickoff_time", "fixture"]).reset_index(drop=True)


def pre_match_team_context(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Freeze form/strength at each GW boundary to avoid target leakage.

    A DGW's second match therefore does not use the first match's result. Rest
    days are schedule information, so the known gap between DGW fixtures is kept.
    """
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    history: dict[str, list[dict[str, float]]] = {team: [] for team in teams}
    previous_scheduled: dict[str, pd.Timestamp | None] = {team: None for team in teams}
    rows: list[dict] = []

    for gw in sorted(fixtures["GW"].unique()):
        metrics: dict[str, dict[str, float]] = {}
        for team in teams:
            all_games = history[team]
            last5 = all_games[-5:]

            def avg(seq: list[dict[str, float]], key: str, default: float) -> float:
                return float(np.mean([game[key] for game in seq])) if seq else float(default)

            metrics[team] = {
                "ppm": avg(last5, "pts", 1.5),
                "gf": avg(last5, "gf", 1.35),
                "ga": avg(last5, "ga", 1.35),
                "ppm_all": avg(all_games, "pts", 1.5),
                "gf_all": avg(all_games, "gf", 1.35),
                "ga_all": avg(all_games, "ga", 1.35),
                "n": float(len(all_games)),
            }

        strength_score = {
            team: (
                0.58 * values["ppm_all"]
                + 0.27 * values["gf_all"]
                - 0.23 * values["ga_all"]
            )
            for team, values in metrics.items()
        }
        score_series = pd.Series(strength_score, dtype=float)
        if score_series.nunique() <= 1:
            strength = {team: 0.5 for team in teams}
        else:
            ranks = score_series.rank(method="average", pct=True)
            n = max(len(ranks), 2)
            strength = {
                team: float(np.clip((ranks[team] - 1.0 / n) / (1.0 - 1.0 / n), 0.0, 1.0))
                for team in teams
            }

        gw_fixtures = fixtures[fixtures["GW"].eq(gw)].sort_values("kickoff_time")
        # Schedule-aware rest days can advance within a DGW without revealing a result.
        schedule_cursor = previous_scheduled.copy()
        for _, fixture in gw_fixtures.iterrows():
            kickoff = pd.Timestamp(fixture["kickoff_time"])
            for is_home in (True, False):
                team = fixture["home_team"] if is_home else fixture["away_team"]
                opponent = fixture["away_team"] if is_home else fixture["home_team"]
                prior = schedule_cursor.get(team)
                rest_days = 7.0 if prior is None else (kickoff - prior).total_seconds() / 86400.0
                rest_days = float(np.clip(rest_days, 2.0, 21.0))
                opponent_strength = float(strength.get(opponent, 0.5))
                difficulty = float(
                    np.clip(
                        3.0 + 2.0 * (opponent_strength - 0.5) - (0.12 if is_home else 0.0),
                        1.0,
                        5.0,
                    )
                )
                tm = metrics[team]
                rows.append(
                    {
                        "GW": int(gw),
                        "fixture": int(fixture["fixture"]),
                        "team": str(team),
                        "home": 1.0 if is_home else 0.0,
                        "team_matches_observed": tm["n"],
                        "team_form_points": tm["ppm"],
                        "team_attack_form": tm["gf"],
                        "team_defence_form": tm["ga"],
                        "team_strength": float(strength.get(team, 0.5)),
                        "opponent_strength": opponent_strength,
                        "fixture_difficulty": difficulty,
                        "rest_days": rest_days,
                    }
                )
                schedule_cursor[team] = kickoff

        # Only after every GW feature has been frozen are results added to history.
        for _, fixture in gw_fixtures.iterrows():
            hs = float(fixture["home_score"])
            aas = float(fixture["away_score"])
            hp = 3.0 if hs > aas else 1.0 if hs == aas else 0.0
            ap = 3.0 if aas > hs else 1.0 if hs == aas else 0.0
            history[str(fixture["home_team"])].append({"pts": hp, "gf": hs, "ga": aas})
            history[str(fixture["away_team"])].append({"pts": ap, "gf": aas, "ga": hs})
        previous_scheduled = schedule_cursor

    return pd.DataFrame(rows)


def player_pre_gw_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative player state using only earlier GWs."""
    work = raw.copy()
    # Current-GW ownership is available before matches. Estimate the denominator
    # from the FPL squad identity: each manager owns 15 players.
    selected_sum = work.groupby("GW")["selected"].transform("sum").clip(lower=1.0)
    estimated_managers = selected_sum / 15.0
    work["selected_by_percent_pre"] = (100.0 * work["selected"] / estimated_managers).clip(0.0, 100.0)

    aggregate_map: dict[str, tuple[str, str]] = {}
    for out, source in CUMULATIVE_SOURCE.items():
        aggregate_map[out] = (source, "sum")
    aggregate_map["points_this_gw"] = ("total_points", "sum")

    aggregate_map["selected_by_percent_pre"] = ("selected_by_percent_pre", "first")
    gw_player = (
        work.groupby(["name", "GW"], as_index=False)
        .agg(**aggregate_map)
        .sort_values(["name", "GW"])
        .reset_index(drop=True)
    )

    for column in CUMULATIVE_SOURCE:
        gw_player[f"pre_{column}"] = (
            gw_player.groupby("name", sort=False)[column].cumsum()
            - gw_player[column]
        )

    gw_player["form"] = (
        gw_player.groupby("name", sort=False)["points_this_gw"]
        .transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    keep = ["name", "GW", "form", "selected_by_percent_pre"] + [
        f"pre_{column}" for column in CUMULATIVE_SOURCE
    ]
    return gw_player[keep]


def build_training(raw: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixture_table(raw)
    team_context = pre_match_team_context(fixtures)
    player_context = player_pre_gw_features(raw)

    frame = raw.merge(player_context, on=["name", "GW"], how="left", validate="many_to_one")
    frame = frame.merge(
        team_context,
        on=["GW", "fixture", "team"],
        how="left",
        validate="many_to_one",
    )
    if frame["team_strength"].isna().any():
        bad = int(frame["team_strength"].isna().sum())
        raise RuntimeError(f"Could not attach leakage-safe team context to {bad} rows.")

    output = pd.DataFrame(index=frame.index)
    output["name"] = frame["name"].astype(str)
    output["position"] = frame["position"].astype(str)
    output["team"] = frame["team"].astype(str)
    output["GW"] = frame["GW"].astype(int)
    output["kickoff_time"] = frame["kickoff_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output["fixture"] = frame["fixture"].astype(int)
    output["price"] = pd.to_numeric(frame["value"], errors="coerce").fillna(55.0) / 10.0

    for column in CUMULATIVE_SOURCE:
        output[column] = pd.to_numeric(frame[f"pre_{column}"], errors="coerce").fillna(0.0)

    output["rating"] = 6.0
    output["form"] = pd.to_numeric(frame["form"], errors="coerce").fillna(0.0)
    output["selected_by_percent"] = pd.to_numeric(
        frame["selected_by_percent_pre"], errors="coerce"
    ).fillna(0.0)
    output["chance_playing"] = 1.0
    for column in [
        "fixture_difficulty", "home", "team_matches_observed", "team_strength",
        "team_form_points", "team_attack_form", "team_defence_form",
        "opponent_strength", "rest_days",
    ]:
        output[column] = pd.to_numeric(frame[column], errors="coerce")

    # Current game/GW labels. These are never fed back into the pre-match features.
    output["future_points"] = pd.to_numeric(frame["total_points"], errors="coerce").fillna(0.0)
    output["future_minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0).clip(0.0, 90.0)
    output["future_starts"] = pd.to_numeric(frame["starts"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    output["future_started"] = (output["future_starts"] > 0.0).astype(int)
    output["future_appearance"] = (output["future_minutes"] > 0.0).astype(int)
    output["future_start_rate"] = output["future_starts"]
    output["future_fixture_count"] = (
        frame.groupby(["name", "GW"])["fixture"].transform("count").clip(1, 2).astype(float)
    )
    output["official_xp"] = pd.to_numeric(frame["xP"], errors="coerce")

    # Preserve the target-era defensive component values for audit/backtesting.
    for target, source in [
        ("target_defensive_contribution", "defensive_contribution"),
        ("target_cbi", "clearances_blocks_interceptions"),
        ("target_recoveries", "recoveries"),
        ("target_tackles", "tackles"),
    ]:
        values = frame[source] if source in frame.columns else pd.Series(0.0, index=frame.index)
        output[target] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    numeric_columns = output.select_dtypes(include=[np.number]).columns
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not np.isfinite(output[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Generated training table contains non-finite numeric values.")
    if not set(output["position"].unique()).issubset(POSITIONS):
        raise RuntimeError("Assistant Manager or unknown positions leaked into the training table.")
    return output.sort_values(["GW", "kickoff_time", "fixture", "name"]).reset_index(drop=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw", type=Path, default=None, help="Existing merged_gw.csv; skips download")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_path = args.raw
    if raw_path is None:
        print(f"Downloading 2025/26 FPL history to {args.raw_output} ...")
        raw_path = download_raw(args.url, args.raw_output)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw = load_raw(raw_path)
    training = build_training(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output, index=False)
    print(
        f"Wrote {len(training):,} leakage-safe rows to {args.output}. "
        f"GWs={training['GW'].min()}-{training['GW'].max()}, "
        f"players={training['name'].nunique():,}."
    )


if __name__ == "__main__":
    main()
