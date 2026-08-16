from __future__ import annotations

import argparse
import re
from pathlib import Path

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


def flatten(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "__".join(str(v).strip() for v in values if str(v).strip() and str(v).lower() != "nan")
            for values in result.columns.to_flat_index()
        ]
    else:
        result.columns = [str(c).strip() for c in result.columns]
    return result


def canon(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def find_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {canon(c): c for c in frame.columns}
    for candidate in candidates:
        if canon(candidate) in lookup:
            return lookup[canon(candidate)]
    for candidate in candidates:
        key = canon(candidate)
        for normalized, original in lookup.items():
            if normalized.endswith(key):
                return original
    return None


def build_recent_lineups(leagues: list[str], season: str, matches_per_team: int) -> pd.DataFrame:
    import soccerdata as sd

    fbref = sd.FBref(leagues=leagues, seasons=season)
    schedule = flatten(fbref.read_schedule(force_cache=False))
    game_col = find_column(schedule, "game_id")
    date_col = find_column(schedule, "date")
    home_col = find_column(schedule, "home_team")
    away_col = find_column(schedule, "away_team")
    score_col = find_column(schedule, "score")
    if not all([game_col, date_col, home_col, away_col]):
        raise ValueError("Could not identify game_id/date/home_team/away_team in the FBref schedule.")

    schedule["_date"] = pd.to_datetime(schedule[date_col], errors="coerce", utc=True)
    completed = schedule[schedule["_date"].notna() & (schedule["_date"] <= pd.Timestamp.now(tz="UTC"))].copy()
    if score_col:
        completed = completed[completed[score_col].notna()]
    completed = completed.sort_values("_date")

    selected_ids: set[str] = set()
    for team in pd.unique(pd.concat([completed[home_col], completed[away_col]], ignore_index=True).dropna()):
        team_matches = completed[(completed[home_col] == team) | (completed[away_col] == team)].tail(matches_per_team)
        selected_ids.update(team_matches[game_col].dropna().astype(str))

    date_lookup = completed.set_index(completed[game_col].astype(str))["_date"].to_dict()
    lineup_frames: list[pd.DataFrame] = []
    for game_id in sorted(selected_ids, key=lambda gid: date_lookup.get(gid, pd.Timestamp(0, tz="UTC"))):
        try:
            lineup = flatten(fbref.read_lineup(match_id=game_id))
        except Exception as exc:
            print(f"Skipping {game_id}: {exc}")
            continue
        if lineup.empty:
            continue
        player_col = find_column(lineup, "player")
        team_col = find_column(lineup, "team")
        starter_col = find_column(lineup, "is_starter")
        minutes_col = find_column(lineup, "minutes_played")
        if not all([player_col, team_col, starter_col]):
            continue
        part = pd.DataFrame(
            {
                "name": lineup[player_col].astype(str),
                "club": lineup[team_col].astype(str),
                "is_starter": lineup[starter_col].astype(str).str.lower().isin(["true", "1", "yes"]),
                "minutes": pd.to_numeric(lineup[minutes_col], errors="coerce") if minutes_col else np.nan,
                "game_id": game_id,
                "date": date_lookup.get(game_id),
            }
        )
        lineup_frames.append(part)

    if not lineup_frames:
        return pd.DataFrame(columns=["name", "club", "recent_lineup_matches", "recent_starts", "recent_start_rate", "recent_minutes", "latest_lineup_date"])

    all_rows = pd.concat(lineup_frames, ignore_index=True).sort_values("date")
    output_rows: list[dict] = []
    for (name, club), group in all_rows.groupby(["name", "club"], sort=False):
        group = group.tail(matches_per_team)
        output_rows.append(
            {
                "name": name,
                "club": club,
                "recent_lineup_matches": int(group["game_id"].nunique()),
                "recent_starts": int(group["is_starter"].sum()),
                "recent_start_rate": float(group["is_starter"].mean()),
                "recent_minutes": float(pd.to_numeric(group["minutes"], errors="coerce").fillna(0.0).mean()),
                "latest_lineup_date": group["date"].max().isoformat() if group["date"].notna().any() else "",
            }
        )
    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch recent actual lineups from FBref via soccerdata.")
    parser.add_argument("--leagues", default="epl,laliga,bundesliga,serie-a,ligue-1")
    parser.add_argument("--season", required=True, help="Example: 2026 or 2026-27")
    parser.add_argument("--matches-per-team", type=int, default=6)
    parser.add_argument("--output", default="data/recent_lineup_history.csv")
    args = parser.parse_args()
    requested = [item.strip() for item in args.leagues.split(",") if item.strip()]
    leagues = [LEAGUE_ALIASES.get(item.lower(), item) for item in requested]
    result = build_recent_lineups(leagues, str(args.season), max(1, min(args.matches_per_team, 10)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} recent-lineup player rows to {output}")


if __name__ == "__main__":
    main()
