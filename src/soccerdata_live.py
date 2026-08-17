from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .providers import estimate_start_probability, normalize_club, normalize_name, safe_float


SOCCERDATA_LEAGUES = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
    "Champions League": "INT-Champions League",
}


_CLUB_ALIASES = {
    "mancity": "manchestercity",
    "manchestercity": "manchestercity",
    "manutd": "manchesterunited",
    "manchesterunited": "manchesterunited",
    "tottenhamhotspur": "tottenham",
    "spurs": "tottenham",
    "parissaintgermain": "psg",
    "parissg": "psg",
    "psg": "psg",
    "internazionalemilano": "inter",
    "internazionale": "inter",
    "intermilan": "inter",
    "acmilan": "milan",
    "bayernmunchen": "bayernmunich",
    "bayernmunich": "bayernmunich",
    "borussiadortmund": "dortmund",
    "rbLeipzig".lower(): "rbleipzig",
    "atleticomadrid": "atleticomadrid",
    "atleticodemadrid": "atleticomadrid",
}


def canonical_club_key(value: object) -> str:
    key = normalize_club(value)
    return _CLUB_ALIASES.get(key, key)


def _align_external_clubs(external: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Rewrite external club labels to the live pool's label when keys clearly match."""
    if external is None or external.empty or "club" not in external.columns:
        return external.copy() if isinstance(external, pd.DataFrame) else pd.DataFrame()
    result = external.copy()
    live_by_key = {canonical_club_key(club): str(club) for club in players.get("club", pd.Series(dtype=str)).dropna().unique()}
    if not live_by_key:
        return result
    keys = list(live_by_key)
    for idx, value in result["club"].items():
        key = canonical_club_key(value)
        if key in live_by_key:
            result.at[idx, "club"] = live_by_key[key]
            continue
        # Conservative contained-name fallback for e.g. Brighton vs Brighton & Hove Albion.
        matches = [candidate for candidate in keys if min(len(key), len(candidate)) >= 5 and (key in candidate or candidate in key)]
        if len(matches) == 1:
            result.at[idx, "club"] = live_by_key[matches[0]]
    return result


@dataclass
class SoccerDataBundle:
    understat_players: pd.DataFrame
    espn_recent_lineups: pd.DataFrame
    sofascore_table: pd.DataFrame
    sofascore_schedule: pd.DataFrame
    clubelo: pd.DataFrame
    status: dict[str, dict[str, Any]]


class SoccerDataError(RuntimeError):
    pass


def _load_soccerdata():
    try:
        import soccerdata as sd  # type: ignore
    except Exception as exc:
        raise SoccerDataError(
            "SoccerData could not be imported. Check that soccerdata==1.9.1 is in requirements.txt "
            f"and that Streamlit redeployed successfully. Original error: {exc}"
        ) from exc
    return sd


def _flatten(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
        out = out.reset_index()
    out.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        for col in out.columns
    ]
    return out


def _pick_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _position_group(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"gk", "goalkeeper"} or "goalkeeper" in text:
        return "GK"
    if text in {"sub", "substitute"}:
        return "MID"
    if any(token in text for token in ["back", "defender", "defence", "dc", "dl", "dr", "cb", "lb", "rb"]):
        return "DEF"
    if any(token in text for token in ["forward", "striker", "fw", "fwd", "winger"]):
        return "FWD"
    if any(token in text for token in ["midfield", "midfielder", "dm", "cm", "am", "ml", "mr", "mc"]):
        return "MID"
    # Understat position codes often look like FW, AMR, AML, MC, DC, GK.
    upper = str(value or "").upper()
    if "GK" in upper:
        return "GK"
    if upper.startswith("D"):
        return "DEF"
    if upper.startswith("F"):
        return "FWD"
    return "MID"


def _parse_minute(value: object, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"start", "kickoff"}:
        return 0.0
    if text in {"end", "full", "full time", "full_time"}:
        return 90.0
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        return float(np.clip(float(match.group()), 0.0, 120.0))
    except ValueError:
        return default


def _espn_minutes(sub_in: object, sub_out: object, is_starter: bool) -> float:
    start = _parse_minute(sub_in, 0.0 if is_starter else None)
    end = _parse_minute(sub_out, 90.0)
    if start is None:
        # Unused bench player when ESPN lists the matchday squad.
        return 0.0
    if end is None:
        end = 90.0
    return float(np.clip(end - start, 0.0, 90.0))


def _fetch_understat(sd, league: str, season: int) -> pd.DataFrame:
    reader = sd.Understat(leagues=league, seasons=season)
    raw = _flatten(reader.read_player_match_stats(force_cache=False))
    if raw.empty:
        raise SoccerDataError("Understat returned no player-match rows for this league/season.")

    name_col = _pick_column(raw, "player", "player_name", "name")
    club_col = _pick_column(raw, "team", "club", "team_name")
    if name_col is None or club_col is None:
        raise SoccerDataError("Understat response did not contain player/team identity columns.")

    minutes_col = _pick_column(raw, "minutes", "mins")
    position_col = _pick_column(raw, "position", "position_name")
    game_col = _pick_column(raw, "game", "game_id", "match", "match_id")

    work = pd.DataFrame(
        {
            "name": raw[name_col].astype(str),
            "club": raw[club_col].astype(str),
            "minutes": pd.to_numeric(raw[minutes_col], errors="coerce").fillna(0.0)
            if minutes_col else 0.0,
            "position_raw": raw[position_col].astype(str) if position_col else "MID",
            "game_key": raw[game_col].astype(str) if game_col else raw.index.astype(str),
        }
    )
    numeric_map = {
        "goals": ("goals", "total_goals"),
        "assists": ("assists", "goal_assists"),
        "shots": ("shots", "total_shots"),
        "xg": ("xg", "expected_goals"),
        "xa": ("xa", "expected_assists"),
        "yellow_cards": ("yellow_cards",),
        "red_cards": ("red_cards",),
        "xg_chain": ("xg_chain",),
        "xg_buildup": ("xg_buildup",),
    }
    for target, candidates in numeric_map.items():
        source = _pick_column(raw, *candidates)
        work[target] = pd.to_numeric(raw[source], errors="coerce").fillna(0.0) if source else 0.0

    work["position"] = work["position_raw"].map(_position_group)
    is_sub = work["position_raw"].astype(str).str.lower().str.contains("sub")
    work["started"] = ((work["minutes"] > 0) & ~is_sub).astype(float)
    work["appeared"] = (work["minutes"] > 0).astype(float)

    rows: list[dict[str, Any]] = []
    for (name, club), group in work.groupby(["name", "club"], sort=False):
        group = group.copy()
        appearances = float(group["appeared"].sum())
        starts = float(group["started"].sum())
        minutes = float(group["minutes"].sum())
        position = group.loc[group["minutes"].idxmax(), "position"] if not group.empty else "MID"
        rows.append(
            {
                "name": name,
                "club": club,
                "position": position,
                "minutes": minutes,
                "appearances": appearances,
                "starts": starts,
                "goals": float(group["goals"].sum()),
                "assists": float(group["assists"].sum()),
                "xg": float(group["xg"].sum()),
                "xa": float(group["xa"].sum()),
                "shots": float(group["shots"].sum()),
                "yellow_cards": float(group["yellow_cards"].sum()),
                "red_cards": float(group["red_cards"].sum()),
                "xg_chain": float(group["xg_chain"].sum()),
                "xg_buildup": float(group["xg_buildup"].sum()),
                "understat_matches": float(group["game_key"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _fetch_espn_recent(sd, league: str, season: int, recent_matches: int) -> pd.DataFrame:
    reader = sd.ESPN(leagues=league, seasons=season)
    schedule = _flatten(reader.read_schedule())
    if schedule.empty:
        raise SoccerDataError("ESPN returned no schedule rows.")

    game_id_col = _pick_column(schedule, "game_id", "match_id", "id")
    date_col = _pick_column(schedule, "date", "kickoff", "kickoff_time")
    if game_id_col is None:
        raise SoccerDataError("ESPN schedule did not expose game_id.")

    sched = schedule.copy()
    if date_col:
        sched["_date"] = pd.to_datetime(sched[date_col], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        completed = sched[sched["_date"].notna() & (sched["_date"] <= now)].copy()
        if completed.empty:
            completed = sched.copy()
        completed = completed.sort_values("_date")
    else:
        completed = sched.copy()

    # Roughly recent_matches for every club without fetching the whole season.
    limit = max(20, min(160, int(recent_matches) * 12))
    selected = completed.tail(limit)
    match_ids = [
        int(value)
        for value in pd.to_numeric(selected[game_id_col], errors="coerce").dropna().unique().tolist()
    ]
    if not match_ids:
        raise SoccerDataError("ESPN had no usable completed match IDs yet.")

    raw = _flatten(reader.read_lineup(match_id=match_ids))
    if raw.empty:
        raise SoccerDataError("ESPN returned no lineup rows for the recent matches.")

    name_col = _pick_column(raw, "player", "player_name", "name")
    club_col = _pick_column(raw, "team", "club", "team_name")
    if name_col is None or club_col is None:
        raise SoccerDataError("ESPN lineup response did not contain player/team columns.")

    game_col = _pick_column(raw, "game", "game_id", "match", "match_id")
    pos_col = _pick_column(raw, "position", "position_name")
    sub_in_col = _pick_column(raw, "sub_in", "subin")
    sub_out_col = _pick_column(raw, "sub_out", "subout")
    formation_col = _pick_column(raw, "formation_place", "formation_position")

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        position_text = str(row.get(pos_col, "") if pos_col else "")
        sub_in = row.get(sub_in_col) if sub_in_col else None
        formation = safe_float(row.get(formation_col), 0.0) if formation_col else 0.0
        starter = str(sub_in).strip().lower() == "start" or (
            formation > 0 and "substitute" not in position_text.lower()
        )
        minutes = _espn_minutes(sub_in, row.get(sub_out_col) if sub_out_col else None, starter)
        rows.append(
            {
                "name": str(row[name_col]),
                "club": str(row[club_col]),
                "game_key": str(row.get(game_col, "")) if game_col else "",
                "started": float(starter),
                "minutes": minutes,
            }
        )
    lineup = pd.DataFrame(rows)
    if lineup.empty:
        raise SoccerDataError("ESPN lineups could not be normalized.")

    # Keep only each player's most recent N matchday-squad records.
    result_rows: list[dict[str, Any]] = []
    for (name, club), group in lineup.groupby(["name", "club"], sort=False):
        recent = group.tail(max(1, int(recent_matches)))
        result_rows.append(
            {
                "name": name,
                "club": club,
                "recent_start_rate": float(recent["started"].mean()),
                "recent_lineup_matches": float(len(recent)),
                "recent_minutes": float(recent["minutes"].mean()),
                "espn_recent_starts": float(recent["started"].sum()),
            }
        )
    return pd.DataFrame(result_rows)


def _fetch_sofascore(sd, league: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    reader = sd.Sofascore(leagues=league, seasons=season)
    table = _flatten(reader.read_league_table(force_cache=False))
    schedule = _flatten(reader.read_schedule(force_cache=False))
    if table.empty and schedule.empty:
        raise SoccerDataError("SofaScore returned neither a league table nor a schedule.")
    return table, schedule


def _fetch_clubelo(sd, league: str) -> pd.DataFrame:
    reader = sd.ClubElo()
    frame = _flatten(reader.read_by_date())
    if frame.empty:
        raise SoccerDataError("ClubElo returned no ratings.")
    team_col = _pick_column(frame, "team", "club")
    elo_col = _pick_column(frame, "elo")
    league_col = _pick_column(frame, "league")
    if team_col is None or elo_col is None:
        raise SoccerDataError("ClubElo response did not contain team and elo columns.")
    out = frame.copy()
    if league_col and league != "INT-Champions League":
        filtered = out[out[league_col].astype(str).eq(league)].copy()
        if not filtered.empty:
            out = filtered
    out = out[[team_col, elo_col]].rename(columns={team_col: "club", elo_col: "elo"})
    out["elo"] = pd.to_numeric(out["elo"], errors="coerce")
    return out.dropna(subset=["elo"]).drop_duplicates("club", keep="last").reset_index(drop=True)


def fetch_soccerdata_bundle(
    competition: str,
    season: int,
    recent_matches: int = 5,
) -> SoccerDataBundle:
    league = SOCCERDATA_LEAGUES.get(competition)
    status: dict[str, dict[str, Any]] = {}
    empty = pd.DataFrame()
    if not league:
        message = f"No SoccerData league mapping is configured for {competition}."
        return SoccerDataBundle(empty, empty, empty, empty, empty, {
            source: {"ok": False, "message": message, "rows": 0}
            for source in ["ESPN", "Understat", "SofaScore", "ClubElo"]
        })

    try:
        sd = _load_soccerdata()
    except SoccerDataError as exc:
        return SoccerDataBundle(empty, empty, empty, empty, empty, {
            source: {"ok": False, "message": str(exc), "rows": 0}
            for source in ["ESPN", "Understat", "SofaScore", "ClubElo"]
        })

    understat = empty
    espn = empty
    sofa_table = empty
    sofa_schedule = empty
    elo = empty

    try:
        understat = _fetch_understat(sd, league, season)
        status["Understat"] = {"ok": True, "message": "player-match xG/xA loaded", "rows": len(understat)}
    except Exception as exc:
        status["Understat"] = {"ok": False, "message": str(exc), "rows": 0}

    try:
        espn = _fetch_espn_recent(sd, league, season, recent_matches)
        status["ESPN"] = {"ok": True, "message": "recent actual lineups loaded", "rows": len(espn)}
    except Exception as exc:
        status["ESPN"] = {"ok": False, "message": str(exc), "rows": 0}

    try:
        sofa_table, sofa_schedule = _fetch_sofascore(sd, league, season)
        status["SofaScore"] = {
            "ok": True,
            "message": f"table + schedule loaded ({len(sofa_schedule)} schedule rows)",
            "rows": len(sofa_table),
        }
    except Exception as exc:
        status["SofaScore"] = {"ok": False, "message": str(exc), "rows": 0}

    try:
        elo = _fetch_clubelo(sd, league)
        status["ClubElo"] = {"ok": True, "message": "current club Elo ratings loaded", "rows": len(elo)}
    except Exception as exc:
        status["ClubElo"] = {"ok": False, "message": str(exc), "rows": 0}

    return SoccerDataBundle(understat, espn, sofa_table, sofa_schedule, elo, status)


def _lookup_by_normalized_name(frame: pd.DataFrame, name_col: str, value_col: str) -> dict[str, float]:
    if frame.empty or name_col not in frame.columns or value_col not in frame.columns:
        return {}
    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        key = canonical_club_key(row.get(name_col))
        value = safe_float(row.get(value_col), np.nan)
        if key and np.isfinite(value):
            result[key] = float(value)
    return result


def build_soccerdata_player_pool(bundle: SoccerDataBundle) -> pd.DataFrame:
    """Create a usable free player pool from Understat when API-Football is unavailable."""
    src = bundle.understat_players.copy()
    if src.empty:
        raise SoccerDataError(
            "SoccerData cannot build a live player pool because Understat did not return player-match data. "
            "Use official FPL for Premier League, or Upload/Demo until Understat works for this competition."
        )
    appearances = pd.to_numeric(src.get("appearances"), errors="coerce").fillna(0.0)
    starts = pd.to_numeric(src.get("starts"), errors="coerce").fillna(0.0)
    start_probability = (starts + 0.5) / (appearances.clip(lower=1.0) + 1.0)
    out = pd.DataFrame(
        {
            "player_id": np.nan,
            "name": src["name"].astype(str),
            "club": src["club"].astype(str),
            "position": src.get("position", "MID"),
            "price": np.nan,
            "minutes": pd.to_numeric(src.get("minutes"), errors="coerce").fillna(0.0),
            "appearances": appearances,
            "starts": starts,
            "start_probability": start_probability.clip(0.0, 1.0),
            "team_matches_observed": appearances,
            "rating": 6.0,
            "goals": pd.to_numeric(src.get("goals"), errors="coerce").fillna(0.0),
            "assists": pd.to_numeric(src.get("assists"), errors="coerce").fillna(0.0),
            "clean_sheets": 0.0,
            "saves": 0.0,
            "yellow_cards": pd.to_numeric(src.get("yellow_cards"), errors="coerce").fillna(0.0),
            "red_cards": pd.to_numeric(src.get("red_cards"), errors="coerce").fillna(0.0),
            "xg": pd.to_numeric(src.get("xg"), errors="coerce").fillna(0.0),
            "xa": pd.to_numeric(src.get("xa"), errors="coerce").fillna(0.0),
            "form": 0.0,
            "total_points": 0.0,
            "bonus": 0.0,
            "bps": 0.0,
            "ict_index": 0.0,
            "selected_by_percent": 0.0,
            "chance_playing": 1.0,
            "injury_reason": "",
            "fixture_difficulty": 3.0,
            "home": 0.5,
            "next_opponent": "TBD",
            "next_kickoff": "",
            "fixture_count": 1.0,
            "next_fixture_ids": [[] for _ in range(len(src))],
            "fixture_scenarios": [[] for _ in range(len(src))],
            "team_strength": 0.5,
            "team_form_points": 1.5,
            "team_attack_form": 1.35,
            "team_defence_form": 1.35,
            "opponent_strength": 0.5,
            "rest_days": 7.0,
            "lineup_status": "",
            "price_source": "Missing",
            "data_source": "SoccerData Understat live",
        }
    )
    return out.reset_index(drop=True)


def _merge_understat(players: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if stats.empty:
        return players.copy(), 0
    result = players.copy()
    right = _align_external_clubs(stats, players)
    result["_name_key"] = result["name"].map(normalize_name)
    result["_club_key"] = result["club"].map(canonical_club_key)
    right["_name_key"] = right["name"].map(normalize_name)
    right["_club_key"] = right["club"].map(canonical_club_key)
    right = right.drop_duplicates(["_name_key", "_club_key"], keep="last").set_index(["_name_key", "_club_key"])
    matched = 0
    for idx, row in result.iterrows():
        key = (row["_name_key"], row["_club_key"])
        if key not in right.index:
            continue
        external = right.loc[key]
        matched += 1
        source = str(row.get("data_source") or "")
        proxy_like = "API-Football" in source or "SoccerData" in source
        for column in ["xg", "xa"]:
            external_value = safe_float(external.get(column), np.nan)
            if not np.isfinite(external_value):
                continue
            current = safe_float(row.get(column), 0.0)
            if proxy_like or current <= 0:
                result.at[idx, column] = external_value
            else:
                # Official FPL xG/xA remain primary; Understat is an independent cross-source check.
                result.at[idx, column] = 0.80 * current + 0.20 * external_value
            result.at[idx, f"understat_{column}"] = external_value
        result.at[idx, "understat_matches"] = safe_float(external.get("understat_matches"), 0.0)
        result.at[idx, "soccerdata_understat"] = True
    return result.drop(columns=["_name_key", "_club_key"]), matched


def _merge_recent_espn(players: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if recent.empty:
        return players.copy(), 0
    from .lineup_intelligence import merge_recent_lineup_history

    recent = _align_external_clubs(recent, players)
    result = merge_recent_lineup_history(players, recent)
    matched = int(pd.to_numeric(
        result.get("recent_lineup_matches", pd.Series(0.0, index=result.index)), errors="coerce"
    ).fillna(0.0).gt(0).sum())
    if matched:
        signal = pd.to_numeric(
            result.get("recent_lineup_matches", pd.Series(0.0, index=result.index)), errors="coerce"
        ).fillna(0.0).gt(0)
        result.loc[signal, "soccerdata_espn"] = True
    return result, matched


def _merge_club_strength(players: pd.DataFrame, bundle: SoccerDataBundle) -> tuple[pd.DataFrame, int, int]:
    result = players.copy()
    matched_sofa = 0
    matched_elo = 0

    # SofaScore league-table points per game: small, conservative strength blend.
    table = _flatten(bundle.sofascore_table)
    if not table.empty:
        team_col = _pick_column(table, "team", "club")
        mp_col = _pick_column(table, "mp", "played", "matches_played")
        pts_col = _pick_column(table, "pts", "points")
        if team_col and mp_col and pts_col:
            lookup: dict[str, tuple[float, float]] = {}
            for _, row in table.iterrows():
                mp = max(safe_float(row.get(mp_col), 0.0), 1.0)
                ppg = safe_float(row.get(pts_col), 0.0) / mp
                lookup[canonical_club_key(row.get(team_col))] = (float(np.clip(ppg / 3.0, 0.0, 1.0)), mp)
            for idx, row in result.iterrows():
                key = canonical_club_key(row.get("club"))
                if key in lookup:
                    sofa_strength, mp = lookup[key]
                    current = safe_float(row.get("team_strength"), 0.5)
                    result.at[idx, "team_strength"] = float(np.clip(0.80 * current + 0.20 * sofa_strength, 0.0, 1.0))
                    result.at[idx, "sofascore_strength"] = sofa_strength
                    result.at[idx, "soccerdata_sofascore"] = True
                    result.at[idx, "team_matches_observed"] = mp
                    if "starts" in result.columns:
                        result.at[idx, "start_probability"] = estimate_start_probability(row.get("starts"), mp)
                    matched_sofa += 1

    elo = bundle.clubelo.copy()
    if not elo.empty:
        elo["_club_key"] = elo["club"].map(canonical_club_key)
        elo["elo"] = pd.to_numeric(elo["elo"], errors="coerce")
        elo = elo.dropna(subset=["elo"])
        if not elo.empty:
            # Convert raw Elo to within-source percentile. This keeps the feature in the model's 0..1 range.
            elo["elo_strength"] = elo["elo"].rank(pct=True).clip(0.0, 1.0)
            lookup = elo.drop_duplicates("_club_key", keep="last").set_index("_club_key")["elo_strength"].to_dict()
            for idx, row in result.iterrows():
                key = canonical_club_key(row.get("club"))
                if key in lookup:
                    current = safe_float(row.get("team_strength"), 0.5)
                    result.at[idx, "team_strength"] = float(np.clip(0.85 * current + 0.15 * lookup[key], 0.0, 1.0))
                    result.at[idx, "clubelo_strength"] = lookup[key]
                    result.at[idx, "soccerdata_clubelo"] = True
                    matched_elo += 1
                opponent = canonical_club_key(row.get("next_opponent"))
                if opponent in lookup:
                    current_opp = safe_float(row.get("opponent_strength"), 0.5)
                    result.at[idx, "opponent_strength"] = float(
                        np.clip(0.85 * current_opp + 0.15 * lookup[opponent], 0.0, 1.0)
                    )
    return result, matched_sofa, matched_elo


def apply_soccerdata_bundle(players: pd.DataFrame, bundle: SoccerDataBundle) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply only successful sources; a failed scraper never blocks squad generation."""
    result, understat_matches = _merge_understat(players, bundle.understat_players)
    # Team-match counts/strength first, then recent ESPN lineups. This preserves the
    # rotation correction instead of overwriting it with the season-level prior.
    result, sofa_matches, elo_matches = _merge_club_strength(result, bundle)
    result, espn_matches = _merge_recent_espn(result, bundle.espn_recent_lineups)

    any_signal = pd.Series(False, index=result.index)
    for column in ["soccerdata_understat", "soccerdata_espn", "soccerdata_sofascore", "soccerdata_clubelo"]:
        if column in result.columns:
            any_signal |= result[column].eq(True)
    if "data_source" in result.columns and any_signal.any():
        result.loc[any_signal, "data_source"] = (
            result.loc[any_signal, "data_source"].fillna("").astype(str) + " + SoccerData"
        ).str.strip(" +")

    return result, {
        "Understat": understat_matches,
        "ESPN": espn_matches,
        "SofaScore": sofa_matches,
        "ClubElo": elo_matches,
    }
