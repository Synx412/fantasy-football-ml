from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LEAGUE_ALIASES = {
    "epl": "ENG-Premier League",
    "premier-league": "ENG-Premier League",
    "laliga": "ESP-La Liga",
    "la-liga": "ESP-La Liga",
    "bundesliga": "GER-Bundesliga",
    "serie-a": "ITA-Serie A",
    "seriea": "ITA-Serie A",
    "ligue-1": "FRA-Ligue 1",
    "ligue1": "FRA-Ligue 1",
}


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index()
    if isinstance(result.columns, pd.MultiIndex):
        names: list[str] = []
        for values in result.columns.to_flat_index():
            parts = [str(v).strip() for v in values if str(v).strip() and str(v).lower() != "nan"]
            names.append("__".join(parts))
        result.columns = names
    else:
        result.columns = [str(c).strip() for c in result.columns]
    return result


def _canon(value: str) -> str:
    value = value.lower().replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "", value)


def _find_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {_canon(c): c for c in frame.columns}
    for candidate in candidates:
        key = _canon(candidate)
        if key in lookup:
            return lookup[key]
    # Fall back to suffix matching for MultiIndex-flattened FBref fields.
    for candidate in candidates:
        key = _canon(candidate)
        for norm, original in lookup.items():
            if norm.endswith(key):
                return original
    return None


def _number(frame: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    column = _find_column(frame, *list(candidates))
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    column = _find_column(frame, *list(candidates))
    if column is None:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def build_fbref_enrichment(leagues: list[str], season: str) -> pd.DataFrame:
    import soccerdata as sd

    fbref = sd.FBref(leagues=leagues, seasons=season)
    standard = _flatten_columns(fbref.read_player_season_stats(stat_type="standard"))
    shooting = _flatten_columns(fbref.read_player_season_stats(stat_type="shooting"))
    playing = _flatten_columns(fbref.read_player_season_stats(stat_type="playing_time"))

    # Build the identity table first. FBref's reset index normally exposes league/season/team/player.
    identity = pd.DataFrame(
        {
            "league": _text(standard, ["league"]),
            "season": _text(standard, ["season"]),
            "club": _text(standard, ["team", "squad"]),
            "name": _text(standard, ["player"]),
            "position_raw": _text(standard, ["pos", "position"]),
            "minutes": _number(standard, ["Playing Time__Min", "Min", "minutes"]),
            "appearances": _number(standard, ["Playing Time__MP", "MP", "matches"]),
            "starts": _number(standard, ["Playing Time__Starts", "Starts"]),
            "goals": _number(standard, ["Performance__Gls", "Gls", "goals"]),
            "assists": _number(standard, ["Performance__Ast", "Ast", "assists"]),
            "xg": _number(standard, ["Expected__xG", "xG"]),
            "xa": _number(standard, ["Expected__xAG", "xAG", "xA"]),
            "yellow_cards": _number(standard, ["Performance__CrdY", "CrdY"]),
            "red_cards": _number(standard, ["Performance__CrdR", "CrdR"]),
        }
    )

    # Some FBref tables expose useful extras under shooting/playing-time. Merge by normalized identity.
    def attach(source: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
        aux = pd.DataFrame(
            {
                "league": _text(source, ["league"]),
                "season": _text(source, ["season"]),
                "club": _text(source, ["team", "squad"]),
                "name": _text(source, ["player"]),
                **{new: _number(source, candidates) for new, candidates in mapping.items()},
            }
        )
        return aux

    shoot = attach(
        shooting,
        {
            "shots": ["Standard__Sh", "Sh", "shots"],
            "shots_on_target": ["Standard__SoT", "SoT", "shots on target"],
        },
    )
    play = attach(
        playing,
        {
            "starts_playing_time": ["Starts__Starts", "Starts"],
            "minutes_playing_time": ["Playing Time__Min", "Min"],
        },
    )

    keys = ["league", "season", "club", "name"]
    result = identity.merge(shoot, on=keys, how="left").merge(play, on=keys, how="left")
    result["starts"] = result["starts"].fillna(result["starts_playing_time"])
    result["minutes"] = result["minutes"].fillna(result["minutes_playing_time"])
    result = result.drop(columns=["starts_playing_time", "minutes_playing_time"])

    raw_position = result["position_raw"].fillna("").str.upper()
    result["position"] = np.select(
        [
            raw_position.str.contains("GK", regex=False),
            raw_position.str.contains("DF", regex=False),
            raw_position.str.contains("MF", regex=False),
            raw_position.str.contains("FW", regex=False),
        ],
        ["GK", "DEF", "MID", "FWD"],
        default="MID",
    )
    result["data_source"] = "FBref via soccerdata"
    result = result[result["name"].str.len().gt(0) & result["club"].str.len().gt(0)].copy()
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download free public soccer player statistics with soccerdata/FBref for offline model enrichment."
    )
    parser.add_argument(
        "--leagues",
        default="epl,laliga,bundesliga,serie-a,ligue-1",
        help="Comma-separated aliases or soccerdata league IDs.",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Season accepted by soccerdata, e.g. 2025 or 2025-26.",
    )
    parser.add_argument(
        "--output",
        default="data/public_player_enrichment.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    requested = [item.strip() for item in args.leagues.split(",") if item.strip()]
    leagues = [LEAGUE_ALIASES.get(item.lower(), item) for item in requested]
    result = build_fbref_enrichment(leagues, str(args.season))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} public player rows to {output}")


if __name__ == "__main__":
    main()
