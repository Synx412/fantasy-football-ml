from __future__ import annotations

import math
import re
from io import StringIO
from datetime import timedelta

import requests
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


ESPN_DIRECT_LEAGUES = {
    "Premier League": "eng.1",
    "La Liga": "esp.1",
    "Bundesliga": "ger.1",
    "Serie A": "ita.1",
    "Ligue 1": "fra.1",
    "Champions League": "uefa.champions",
}

UNDERSTAT_DIRECT_LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

SOFASCORE_DIRECT_NAMES = {
    "Premier League": {"Premier League"},
    "La Liga": {"LaLiga", "La Liga"},
    "Bundesliga": {"Bundesliga"},
    "Serie A": {"Serie A"},
    "Ligue 1": {"Ligue 1"},
    "Champions League": {"UEFA Champions League", "Champions League"},
}

CLUBELO_COUNTRIES = {
    "Premier League": "ENG",
    "La Liga": "ESP",
    "Bundesliga": "GER",
    "Serie A": "ITA",
    "Ligue 1": "FRA",
}

_DIRECT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 16) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# Used only to improve cross-provider player-name matching for the Premier League.
# Failure is harmless: the merger falls back to the existing names.
_FPL_NAME_CACHE: dict[int, set[str]] | None = None


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



def _load_fpl_name_aliases() -> dict[int, set[str]]:
    """Return FPL player-id -> conservative aliases from official bootstrap data.

    The live player table intentionally displays FPL's short ``web_name``.
    ESPN/Understat usually expose full names, so exact web_name matching misses
    many legitimate rows. This helper keeps the UI name unchanged while adding
    hidden aliases for matching.
    """
    global _FPL_NAME_CACHE
    if _FPL_NAME_CACHE is not None:
        return _FPL_NAME_CACHE

    aliases: dict[int, set[str]] = {}
    try:
        response = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            headers=_DIRECT_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        for p in payload.get("elements", []) or []:
            try:
                pid = int(p.get("id"))
            except Exception:
                continue
            first = str(p.get("first_name") or "").strip()
            second = str(p.get("second_name") or "").strip()
            web = str(p.get("web_name") or "").strip()

            values = {
                normalize_name(web),
                normalize_name(f"{first} {second}".strip()),
                normalize_name(second),
            }
            # Common cross-provider form: first initial + surname.
            if first and second:
                values.add(normalize_name(f"{first[0]} {second}"))
            aliases[pid] = {v for v in values if len(v) >= 2}
    except Exception:
        aliases = {}

    _FPL_NAME_CACHE = aliases
    return aliases


def _candidate_aliases_for_players(players: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Group live-player aliases by club for conservative provider matching."""
    fpl_aliases = _load_fpl_name_aliases() if "player_id" in players.columns else {}
    grouped: dict[str, list[dict[str, Any]]] = {}

    for idx, row in players.iterrows():
        club_key = canonical_club_key(row.get("club"))
        if not club_key:
            continue

        aliases = {normalize_name(row.get("name"))}
        try:
            pid = int(row.get("player_id"))
            aliases |= fpl_aliases.get(pid, set())
        except Exception:
            pass

        aliases = {a for a in aliases if len(a) >= 2}
        grouped.setdefault(club_key, []).append(
            {
                "idx": idx,
                "live_name": str(row.get("name") or ""),
                "position": str(row.get("position") or "").upper(),
                "aliases": aliases,
            }
        )
    return grouped


def _align_external_player_names(external: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Rewrite external names to the live pool's display name when uniquely matched.

    Match order:
      1. exact alias match within the same club;
      2. unique contained alias (e.g. Saka <-> Bukayo Saka);
      3. optional position filter when the external source exposes a position.

    Ambiguous matches are deliberately left untouched rather than guessing.
    """
    if external is None or external.empty or "name" not in external.columns:
        return external.copy() if isinstance(external, pd.DataFrame) else pd.DataFrame()

    result = _align_external_clubs(external, players)
    grouped = _candidate_aliases_for_players(players)

    for idx, row in result.iterrows():
        ext_key = normalize_name(row.get("name"))
        club_key = canonical_club_key(row.get("club"))
        if not ext_key or not club_key:
            continue

        candidates = grouped.get(club_key, [])
        if not candidates:
            continue

        ext_pos = str(row.get("position") or "").upper()
        if ext_pos in {"GK", "DEF", "MID", "FWD"}:
            positional = [c for c in candidates if c["position"] == ext_pos]
            if positional:
                candidates = positional

        # Exact match against any official/display alias.
        exact = [c for c in candidates if ext_key in c["aliases"]]
        if len(exact) == 1:
            result.at[idx, "name"] = exact[0]["live_name"]
            continue

        # Full providers commonly return "Bukayo Saka" while FPL displays "Saka".
        # Require a reasonably informative alias and a UNIQUE player in the club.
        contained = []
        for candidate in candidates:
            hit = False
            for alias in candidate["aliases"]:
                if len(alias) < 4:
                    continue
                if alias in ext_key or (len(ext_key) >= 4 and ext_key in alias):
                    hit = True
                    break
            if hit:
                contained.append(candidate)

        if len(contained) == 1:
            result.at[idx, "name"] = contained[0]["live_name"]

    return result

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
                "understat_season_used": int(season),
                "understat_is_previous_season": False,
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
                "espn_season_used": int(season),
                "espn_is_previous_season": False,
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



def _direct_json(url: str, timeout: int = 20) -> Any:
    response = requests.get(url, headers=_DIRECT_HEADERS, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except Exception as exc:
        raise SoccerDataError(f"Direct source returned non-JSON data from {url}: {exc}") from exc


def _direct_text(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=_DIRECT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _sub_clock(value: Any) -> float | None:
    if isinstance(value, dict):
        clock = value.get("clock") or {}
        value = clock.get("displayValue")
    return _parse_minute(value, None)


def _direct_espn_recent(
    competition: str,
    season: int,
    recent_matches: int,
) -> pd.DataFrame:
    league_id = ESPN_DIRECT_LEAGUES.get(competition)
    if not league_id:
        raise SoccerDataError(f"No direct ESPN mapping for {competition}.")

    base = f"http://site.api.espn.com/apis/site/v2/sports/soccer/{league_id}"

    def completed_events_for_season(start_year: int) -> list[dict[str, Any]]:
        # ESPN exposes a season calendar from the July-1 scoreboard response.
        seed = _direct_json(f"{base}/scoreboard?dates={start_year}0701")
        leagues = seed.get("leagues") or []
        calendar = leagues[0].get("calendar", []) if leagues else []

        now = datetime.now(tz=timezone.utc)
        dates: list[datetime] = []
        for raw in calendar:
            try:
                parsed = pd.to_datetime(raw, utc=True).to_pydatetime()
            except Exception:
                continue
            if parsed <= now:
                dates.append(parsed)

        # Recent match dates only. 10 dates is normally several matchweeks and
        # avoids dozens of unnecessary requests.
        dates = sorted(dates)[-10:]
        events: list[dict[str, Any]] = []
        for dt in dates:
            payload = _direct_json(f"{base}/scoreboard?dates={dt.strftime('%Y%m%d')}")
            for event in payload.get("events", []) or []:
                status = ((event.get("status") or {}).get("type") or {})
                if bool(status.get("completed")) or str(status.get("state", "")).lower() == "post":
                    events.append(event)
        # newest first, unique by id
        unique: dict[str, dict[str, Any]] = {}
        for event in sorted(events, key=lambda e: str(e.get("date", "")), reverse=True):
            eid = str(event.get("id", ""))
            if eid and eid not in unique:
                unique[eid] = event
        return list(unique.values())

    requested_season = int(season)
    season_used = requested_season
    events = completed_events_for_season(requested_season)
    if not events:
        # Useful at the start of a new season before any league match has been played.
        season_used = requested_season - 1
        events = completed_events_for_season(season_used)

    if not events:
        raise SoccerDataError("ESPN direct fallback found no completed matches.")

    # Cap summary calls to keep Streamlit fast. Newest matches have priority.
    events = events[: min(36, max(12, int(recent_matches) * 8))]
    rows: list[dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue
        try:
            data = _direct_json(f"{base}/summary?event={event_id}")
        except Exception:
            continue

        rosters = data.get("rosters") or []
        box_form = ((data.get("boxscore") or {}).get("form") or [])
        event_date = pd.to_datetime(event.get("date"), errors="coerce", utc=True)

        for team_idx, roster_block in enumerate(rosters[:2]):
            roster = roster_block.get("roster") or []
            team_name = ""
            if team_idx < len(box_form):
                team_name = ((box_form[team_idx].get("team") or {}).get("displayName") or "")
            if not team_name:
                team_name = ((roster_block.get("team") or {}).get("displayName") or "")

            for p in roster:
                athlete = p.get("athlete") or {}
                name = athlete.get("displayName")
                if not name:
                    continue

                starter = bool(p.get("starter", False))
                sub_in_obj = p.get("subbedIn")
                sub_out_obj = p.get("subbedOut")

                def did_sub(obj: Any) -> bool:
                    if isinstance(obj, bool):
                        return obj
                    if isinstance(obj, dict):
                        return bool(obj.get("didSub"))
                    return False

                did_in = did_sub(sub_in_obj)
                did_out = did_sub(sub_out_obj)

                if starter:
                    minute_in = 0.0
                elif did_in:
                    minute_in = _sub_clock(sub_in_obj)
                else:
                    minute_in = None

                if (starter or did_in) and not did_out:
                    minute_out = 90.0
                elif did_out:
                    minute_out = _sub_clock(sub_out_obj)
                else:
                    minute_out = None

                if minute_in is None:
                    minutes = 0.0
                else:
                    minutes = max(0.0, min(90.0, (minute_out if minute_out is not None else 90.0) - minute_in))

                rows.append(
                    {
                        "name": str(name),
                        "club": str(team_name),
                        "game_key": event_id,
                        "event_date": event_date,
                        "started": float(starter),
                        "minutes": float(minutes),
                    }
                )

    lineup = pd.DataFrame(rows)
    if lineup.empty:
        raise SoccerDataError("ESPN direct fallback returned no usable lineup rows.")

    lineup = lineup.sort_values(["event_date", "game_key"], na_position="first")
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
                "espn_season_used": int(season_used),
                "espn_is_previous_season": bool(season_used != requested_season),
            }
        )
    return pd.DataFrame(result_rows)


def _direct_understat_players(competition: str, season: int) -> pd.DataFrame:
    slug = UNDERSTAT_DIRECT_LEAGUES.get(competition)
    if not slug:
        raise SoccerDataError(f"Understat direct fallback is not configured for {competition}.")

    # Understat's current JSON endpoints expect a normal session/cookies plus
    # X-Requested-With. This mirrors the current SoccerData implementation.
    session = requests.Session()
    headers = dict(_DIRECT_HEADERS)
    headers["X-Requested-With"] = "XMLHttpRequest"

    try:
        home = session.get("https://understat.com", headers=_DIRECT_HEADERS, timeout=20)
        home.raise_for_status()
    except Exception as exc:
        raise SoccerDataError(f"Understat homepage/cookie initialization failed: {exc}") from exc

    def get_json_session(url: str) -> dict[str, Any]:
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

    requested = int(season)
    season_to_use = requested

    # Discover seasons Understat actually exposes. At the opening of a season,
    # the requested new season can lag behind on Understat, so use the latest
    # available league season as a statistical prior instead of failing.
    try:
        stat_data = get_json_session("https://understat.com/getStatData")
        available: list[int] = []
        for item in stat_data.get("stat", []) or []:
            if str(item.get("league", "")).replace(" ", "_") != slug:
                continue
            year = int(item.get("year"))
            month = int(item.get("month"))
            sid = year if month >= 7 else year - 1
            available.append(sid)
        available = sorted(set(available))
        if available and requested not in available:
            eligible = [x for x in available if x <= requested]
            season_to_use = max(eligible) if eligible else max(available)
    except Exception:
        # Discovery is helpful but not mandatory. We still try requested season.
        season_to_use = requested

    def fetch_year(year: int) -> dict[str, Any]:
        return get_json_session(f"https://understat.com/getLeagueData/{slug}/{year}")

    try:
        data = fetch_year(season_to_use)
    except Exception as first_exc:
        if season_to_use != requested:
            raise SoccerDataError(
                f"Understat direct fallback failed for latest available season "
                f"{season_to_use}: {first_exc}"
            ) from first_exc
        # Last-resort previous-season prior.
        try:
            season_to_use = requested - 1
            data = fetch_year(season_to_use)
        except Exception as second_exc:
            raise SoccerDataError(
                f"Understat direct fallback failed for {requested} and {requested - 1}: "
                f"{first_exc} | {second_exc}"
            ) from second_exc

    players = data.get("players") or []
    if not players:
        raise SoccerDataError(
            f"Understat returned no player rows for season {season_to_use}."
        )

    rows: list[dict[str, Any]] = []
    for p in players:
        games = safe_float(p.get("games"), 0.0)
        minutes = safe_float(p.get("time"), 0.0)
        appearances = max(games, 0.0)
        starts_est = min(appearances, max(0.0, minutes / 75.0))
        team = str(p.get("team_title") or "")
        if "," in team:
            team = team.split(",", 1)[0].strip()
        rows.append(
            {
                "name": str(p.get("player_name") or ""),
                "club": team,
                "position": _position_group(p.get("position")),
                "minutes": minutes,
                "appearances": appearances,
                "starts": starts_est,
                "goals": safe_float(p.get("goals"), 0.0),
                "assists": safe_float(p.get("assists"), 0.0),
                "shots": safe_float(p.get("shots"), 0.0),
                "xg": safe_float(p.get("xG"), 0.0),
                "xa": safe_float(p.get("xA"), 0.0),
                "yellow_cards": safe_float(p.get("yellow_cards"), 0.0),
                "red_cards": safe_float(p.get("red_cards"), 0.0),
                "xg_chain": safe_float(p.get("xGChain"), 0.0),
                "xg_buildup": safe_float(p.get("xGBuildup"), 0.0),
                "understat_matches": appearances,
                "understat_season_used": season_to_use,
                "understat_is_previous_season": bool(season_to_use != requested),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["name"].astype(str).str.len().gt(0)].reset_index(drop=True)
    if out.empty:
        raise SoccerDataError("Understat direct fallback could not normalize player rows.")
    return out


def _season_matches_sofascore(raw: dict[str, Any], season: int) -> dict[str, Any] | None:
    target = int(season)
    yy = str(target)[-2:]
    next_yy = str(target + 1)[-2:]
    for item in raw.get("seasons", []) or []:
        candidates = " ".join(
            str(item.get(k, "")) for k in ("name", "year")
        )
        if str(target) in candidates or f"{yy}/{next_yy}" in candidates or f"{yy}-{next_yy}" in candidates:
            return item
    # Current seasons are normally returned first.
    seasons = raw.get("seasons", []) or []
    return seasons[0] if seasons else None


def _direct_sofascore(
    competition: str,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = SOFASCORE_DIRECT_NAMES.get(competition)
    if not wanted:
        raise SoccerDataError(f"No direct SofaScore mapping for {competition}.")

    base = "https://api.sofascore.com/api/v1"
    cfg = _direct_json(f"{base}/config/default-unique-tournaments/EN/football")
    tournaments = cfg.get("uniqueTournaments") or []

    tournament = None
    wanted_lower = {x.lower() for x in wanted}
    for item in tournaments:
        if str(item.get("name", "")).strip().lower() in wanted_lower:
            tournament = item
            break
    if tournament is None:
        raise SoccerDataError(f"SofaScore direct fallback could not locate {competition}.")

    tournament_id = int(tournament["id"])
    season_data = _direct_json(f"{base}/unique-tournament/{tournament_id}/seasons")
    chosen = _season_matches_sofascore(season_data, season)
    if chosen is None:
        raise SoccerDataError("SofaScore direct fallback could not identify the season.")
    season_id = int(chosen["id"])

    standings = _direct_json(
        f"{base}/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
    )
    table_rows: list[dict[str, Any]] = []
    blocks = standings.get("standings") or []
    for row in (blocks[0].get("rows", []) if blocks else []):
        team = row.get("team") or {}
        table_rows.append(
            {
                "team": team.get("name", ""),
                "MP": row.get("matches", 0),
                "W": row.get("wins", 0),
                "D": row.get("draws", 0),
                "L": row.get("losses", 0),
                "GF": row.get("scoresFor", 0),
                "GA": row.get("scoresAgainst", 0),
                "GD": safe_float(row.get("scoresFor"), 0.0) - safe_float(row.get("scoresAgainst"), 0.0),
                "Pts": row.get("points", 0),
            }
        )
    table = pd.DataFrame(table_rows)

    # Schedule is useful as a cross-check, but don't turn one optional schedule
    # endpoint into a total SofaScore failure.
    schedule_rows: list[dict[str, Any]] = []
    try:
        rounds_data = _direct_json(
            f"{base}/unique-tournament/{tournament_id}/season/{season_id}/rounds"
        )
        rounds = rounds_data.get("rounds") or []
        # Current/recent rounds only to keep requests low.
        round_numbers = [r.get("round") for r in rounds if r.get("round") is not None][-4:]
        for number in round_numbers:
            payload = _direct_json(
                f"{base}/unique-tournament/{tournament_id}/season/{season_id}/events/round/{number}"
            )
            for event in payload.get("events", []) or []:
                schedule_rows.append(
                    {
                        "round": number,
                        "week": ((event.get("roundInfo") or {}).get("round")),
                        "date": datetime.fromtimestamp(
                            safe_float(event.get("startTimestamp"), 0.0), tz=timezone.utc
                        ),
                        "home_team": ((event.get("homeTeam") or {}).get("name", "")),
                        "away_team": ((event.get("awayTeam") or {}).get("name", "")),
                        "game_id": event.get("id"),
                    }
                )
    except Exception:
        pass

    if table.empty and not schedule_rows:
        raise SoccerDataError("SofaScore direct fallback returned no table or schedule rows.")
    return table, pd.DataFrame(schedule_rows)


def _direct_clubelo(competition: str) -> pd.DataFrame:
    datestring = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    errors: list[str] = []
    content = None
    for scheme in ("http", "https"):
        try:
            response = requests.get(
                f"{scheme}://api.clubelo.com/{datestring}",
                headers=_DIRECT_HEADERS,
                timeout=8,
            )
            response.raise_for_status()
            content = response.text
            if content.strip():
                break
        except Exception as exc:
            errors.append(str(exc))
    if not content:
        raise SoccerDataError("ClubElo direct fallback failed: " + " | ".join(errors[-2:]))

    raw = pd.read_csv(StringIO(content))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    club_col = _pick_column(raw, "club", "team")
    elo_col = _pick_column(raw, "elo")
    country_col = _pick_column(raw, "country")
    level_col = _pick_column(raw, "level")
    if club_col is None or elo_col is None:
        raise SoccerDataError("ClubElo CSV did not contain club/team and Elo columns.")

    country = CLUBELO_COUNTRIES.get(competition)
    out = raw.copy()
    if country and country_col:
        filtered = out[out[country_col].astype(str).str.upper().eq(country)].copy()
        if level_col:
            level1 = filtered[pd.to_numeric(filtered[level_col], errors="coerce").eq(1)].copy()
            if not level1.empty:
                filtered = level1
        if not filtered.empty:
            out = filtered

    result = out[[club_col, elo_col]].rename(columns={club_col: "club", elo_col: "elo"})
    result["elo"] = pd.to_numeric(result["elo"], errors="coerce")
    result = result.dropna(subset=["elo"]).drop_duplicates("club", keep="last").reset_index(drop=True)
    if result.empty:
        raise SoccerDataError("ClubElo direct fallback returned no usable ratings.")
    return result


def _run_with_direct_fallback(
    source_name: str,
    soccerdata_call,
    direct_call,
    success_message: str,
) -> tuple[Any, dict[str, Any]]:
    first_error = None
    try:
        value = soccerdata_call()
        rows = len(value[0]) if isinstance(value, tuple) else len(value)
        return value, {
            "ok": True,
            "message": f"SoccerData path — {success_message}",
            "rows": rows,
            "route": "SoccerData",
        }
    except Exception as exc:
        first_error = str(exc)

    try:
        value = direct_call()
        rows = len(value[0]) if isinstance(value, tuple) else len(value)
        return value, {
            "ok": True,
            "message": f"Direct fallback — {success_message}",
            "rows": rows,
            "route": "Direct",
            "soccerdata_error": first_error,
        }
    except Exception as exc:
        return None, {
            "ok": False,
            "message": f"SoccerData failed: {first_error} | Direct fallback failed: {exc}",
            "rows": 0,
            "route": "Failed",
        }

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
            source: {"ok": False, "message": message, "rows": 0, "route": "Failed"}
            for source in ["ESPN", "Understat", "SofaScore", "ClubElo"]
        })

    # Import failure should not disable direct fallbacks.
    try:
        sd = _load_soccerdata()
        sd_error = None
    except SoccerDataError as exc:
        sd = None
        sd_error = str(exc)

    def unavailable():
        raise SoccerDataError(sd_error or "SoccerData import is unavailable.")

    understat_call = (
        (lambda: _fetch_understat(sd, league, season)) if sd is not None else unavailable
    )
    understat, status["Understat"] = _run_with_direct_fallback(
        "Understat",
        understat_call,
        lambda: _direct_understat_players(competition, season),
        "player xG/xA statistics loaded",
    )
    if understat is None:
        understat = empty

    espn_call = (
        (lambda: _fetch_espn_recent(sd, league, season, recent_matches))
        if sd is not None else unavailable
    )
    espn, status["ESPN"] = _run_with_direct_fallback(
        "ESPN",
        espn_call,
        lambda: _direct_espn_recent(competition, season, recent_matches),
        "recent actual lineup/minutes data loaded",
    )
    if espn is None:
        espn = empty

    sofa_call = (
        (lambda: _fetch_sofascore(sd, league, season)) if sd is not None else unavailable
    )
    sofa_value, status["SofaScore"] = _run_with_direct_fallback(
        "SofaScore",
        sofa_call,
        lambda: _direct_sofascore(competition, season),
        "league table/schedule data loaded (optional source)",
    )
    if sofa_value is None:
        sofa_table, sofa_schedule = empty, empty
    else:
        sofa_table, sofa_schedule = sofa_value

    elo_call = (
        (lambda: _fetch_clubelo(sd, league)) if sd is not None else unavailable
    )
    elo, status["ClubElo"] = _run_with_direct_fallback(
        "ClubElo",
        elo_call,
        lambda: _direct_clubelo(competition),
        "current club Elo ratings loaded (optional source)",
    )
    if elo is None:
        elo = empty

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
    right = _align_external_player_names(stats, players)
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
        # IMPORTANT: keep Understat in its own statistical window.
        # At season rollover FPL totals reset to zero while the Understat fallback
        # can be the previous season. Blending raw totals from those two windows
        # corrupts per-90 features (e.g. previous-season xG divided by 0/1 current
        # season minutes). Preserve the source values separately instead.
        for column in ["minutes", "goals", "assists", "shots", "xg", "xa"]:
            external_value = safe_float(external.get(column), np.nan)
            if np.isfinite(external_value):
                result.at[idx, f"understat_{column}"] = external_value

        result.at[idx, "understat_matches"] = safe_float(
            external.get("understat_matches"), 0.0
        )
        if pd.notna(external.get("understat_season_used", np.nan)):
            result.at[idx, "understat_season_used"] = safe_float(
                external.get("understat_season_used"), np.nan
            )
        result.at[idx, "understat_is_previous_season"] = bool(
            external.get("understat_is_previous_season", False)
        )
        result.at[idx, "soccerdata_understat"] = True
    return result.drop(columns=["_name_key", "_club_key"]), matched


def _merge_recent_espn(players: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if recent.empty:
        return players.copy(), 0
    from .lineup_intelligence import merge_recent_lineup_history

    recent = _align_external_player_names(recent, players)

    # Keep freshness metadata before the generic lineup merge (which intentionally
    # only copies lineup fields).
    metadata: dict[tuple[str, str], tuple[float | None, bool]] = {}
    for _, row in recent.iterrows():
        key = (normalize_name(row.get("name")), canonical_club_key(row.get("club")))
        season_value = safe_float(row.get("espn_season_used"), np.nan)
        metadata[key] = (
            season_value if np.isfinite(season_value) else None,
            bool(row.get("espn_is_previous_season", False)),
        )

    result = merge_recent_lineup_history(players, recent)
    matched = int(pd.to_numeric(
        result.get("recent_lineup_matches", pd.Series(0.0, index=result.index)), errors="coerce"
    ).fillna(0.0).gt(0).sum())
    if matched:
        signal = pd.to_numeric(
            result.get("recent_lineup_matches", pd.Series(0.0, index=result.index)), errors="coerce"
        ).fillna(0.0).gt(0)
        result.loc[signal, "soccerdata_espn"] = True

        for idx, row in result.loc[signal].iterrows():
            key = (normalize_name(row.get("name")), canonical_club_key(row.get("club")))
            if key not in metadata:
                continue
            season_value, stale = metadata[key]
            if season_value is not None:
                result.at[idx, "espn_season_used"] = season_value
            result.at[idx, "espn_is_previous_season"] = bool(stale)
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
            pool_strength_map: dict[str, float] = {}
            for club_name in result.get("club", pd.Series(dtype=str)).dropna().astype(str).unique():
                club_key = canonical_club_key(club_name)
                if club_key in lookup:
                    pool_strength_map[club_name] = float(lookup[club_key])
            if pool_strength_map:
                result["clubelo_strength_map"] = [pool_strength_map.copy() for _ in range(len(result))]

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
                    result.at[idx, "clubelo_opponent_strength"] = lookup[opponent]
                    result.at[idx, "opponent_strength"] = float(
                        np.clip(0.85 * current_opp + 0.15 * lookup[opponent], 0.0, 1.0)
                    )
    return result, matched_sofa, matched_elo


def apply_soccerdata_bundle(players: pd.DataFrame, bundle: SoccerDataBundle) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply successful sources and preserve base-provider values for late-fusion xP."""
    result = players.copy()

    # Keep an untouched snapshot of the primary provider inputs. For Premier
    # League this is Official FPL. These columns let model.py produce a genuine
    # base-provider xP instead of only predicting from already blended features.
    for column in [
        "xg",
        "xa",
        "start_probability",
        "team_strength",
        "opponent_strength",
    ]:
        if column in result.columns and f"base_{column}" not in result.columns:
            result[f"base_{column}"] = result[column]

    result, understat_matches = _merge_understat(result, bundle.understat_players)
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
