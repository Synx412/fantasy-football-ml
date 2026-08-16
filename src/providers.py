from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import requests
except ModuleNotFoundError:  # The standard-library fallback keeps local tests lightweight.
    requests = None

from .config import API_POSITION_MAP, FPL_POSITION_MAP

FPL_BASE = "https://fantasy.premierleague.com/api"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
COMPLETED_STATUSES = {"FT", "AET", "PEN"}


class DataProviderError(RuntimeError):
    pass


def _get_json(url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    request_headers = {"User-Agent": "FantasyXI-ML/2.0"}
    if headers:
        request_headers.update(headers)
    if requests is not None:
        try:
            response = requests.get(url, headers=request_headers, params=params, timeout=35)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise DataProviderError(f"Data request failed: {exc}") from exc
        except ValueError as exc:
            raise DataProviderError("The data provider returned invalid JSON.") from exc
    else:
        import json

        query = urlencode(params or {}, doseq=True)
        request_url = f"{url}?{query}" if query else url
        request = Request(request_url, headers=request_headers)
        try:
            with urlopen(request, timeout=35) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise DataProviderError(f"Data request failed: {exc}") from exc
        except (UnicodeDecodeError, ValueError) as exc:
            raise DataProviderError("The data provider returned invalid JSON.") from exc
    if not isinstance(payload, (dict, list)):
        raise DataProviderError("The data provider returned an unexpected response.")
    return payload


def _check_api_errors(payload: dict, label: str = "API-Football") -> None:
    errors = payload.get("errors")
    if errors:
        raise DataProviderError(f"{label} error: {errors}")


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_club(value: object) -> str:
    """Create a cross-provider club key without relying on provider-specific IDs."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    stop_words = {
        "fc", "cf", "afc", "ac", "as", "sc", "ss", "ssc", "bc", "cd",
        "club", "football", "futbol", "calcio", "de", "del", "the",
    }
    tokens = [token for token in re.findall(r"[a-z]+", text) if token not in stop_words]
    key = "".join(tokens)
    aliases = {
        "bayernmunchen": "bayernmunich",
        "internazionale": "inter",
        "internazionalemilano": "inter",
        "athletic": "athleticbilbao",
        "realbetisbalompie": "realbetis",
        "tottenhamhotspur": "tottenham",
        "newcastleunited": "newcastle",
        "westhamunited": "westham",
        "wolverhamptonwanderers": "wolves",
        "brightonhovealbion": "brighton",
        "olympiquemarseille": "marseille",
        "olympiquelyonnais": "lyon",
        "olympiquelyon": "lyon",
        "staderennais": "rennes",
        "lillelosc": "lille",
        "losclille": "lille",
        "ogcnice": "nice",
        "rclens": "lens",
        "rcstrasbourgalsace": "strasbourg",
        "sportingportugal": "sportingcp",
        "sportingclubeportugal": "sportingcp",
        "psveindhoven": "psv",
    }
    return aliases.get(key, key)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def estimate_fpl_appearances(
    minutes: float,
    starts: float,
    subbed_in: object = None,
    max_appearances: int = 38,
    team_matches: Optional[float] = None,
) -> float:
    """Estimate appearances without turning a rotation player into a 100% starter.

    FPL exposes starts and minutes but not a clean appearance counter.  The old
    starts/estimated-appearances ratio could badly overstate players with several
    starts plus substitute cameos.  Team matches cap the reconstruction and make
    the same logic usable for every player.
    """
    starts = max(safe_float(starts), 0.0)
    minutes = max(safe_float(minutes), 0.0)
    match_cap = max(
        int(round(safe_float(team_matches, max_appearances))),
        int(math.ceil(starts)),
        1,
    )
    match_cap = min(match_cap, max(int(max_appearances), 1))
    if subbed_in is not None:
        appearances = starts + max(safe_float(subbed_in), 0.0)
    else:
        if team_matches is None:
            # Backward-compatible fallback when team-match context is unavailable.
            estimated_starter_minutes = 90.0
        else:
            # A start is not normally 90 minutes. Assuming 90 can hide substitute
            # cameos for players who also start some matches.
            starter_share = starts / max(float(match_cap), 1.0)
            estimated_starter_minutes = 70.0 + 15.0 * float(
                np.clip((starter_share - 0.20) / 0.60, 0.0, 1.0)
            )
        substitute_minutes = max(minutes - estimated_starter_minutes * starts, 0.0)
        estimated_substitute_apps = math.ceil(substitute_minutes / 18.0)
        appearances = starts + estimated_substitute_apps
    return float(np.clip(max(appearances, 1.0), 1.0, match_cap))


def estimate_start_probability(starts: float, team_matches: float) -> float:
    """Smoothed unconditional probability that a player starts a team match."""
    starts = max(safe_float(starts), 0.0)
    observed = safe_float(team_matches, 0.0)
    # Before a new season has any completed matches, 0/0 means unknown, not 0%.
    if observed <= 0 and starts <= 0:
        return 0.5
    matches = max(observed, starts, 1.0)
    # Jeffreys-style smoothing prevents tiny samples from becoming 0% or 100%.
    return float(np.clip((starts + 0.5) / (matches + 1.0), 0.0, 1.0))


def fpl_team_matches_observed(bootstrap: dict, fixtures: list[dict]) -> dict[int, int]:
    """Count completed FPL team matches, with a rollover guard for stale stats."""
    team_ids = [int(team["id"]) for team in bootstrap.get("teams", [])]
    completed = {team_id: 0 for team_id in team_ids}
    for fixture in fixtures:
        if not fixture.get("finished"):
            continue
        for key in ("team_h", "team_a"):
            team_id = int(fixture.get(key) or 0)
            if team_id in completed:
                completed[team_id] += 1

    max_starts = {team_id: 0 for team_id in team_ids}
    for player in bootstrap.get("elements", []):
        team_id = int(player.get("team") or 0)
        if team_id in max_starts:
            max_starts[team_id] = max(max_starts[team_id], int(safe_float(player.get("starts"))))

    for team_id in team_ids:
        # Around season rollover the fixture list and bootstrap totals can briefly
        # disagree.  Never let a player with old totals become a fake 100% starter.
        if max_starts[team_id] > completed[team_id] + 1:
            completed[team_id] = 38
        completed[team_id] = max(completed[team_id], 0)
    return completed


def _parse_datetime(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _fpl_fixture_context(
    bootstrap: dict,
    fixtures: list[dict],
    horizon: int = 1,
) -> dict[int, dict[str, Any]]:
    team_names = {int(t["id"]): t.get("name", str(t["id"])) for t in bootstrap.get("teams", [])}
    events = bootstrap.get("events", [])
    next_event = next((e.get("id") for e in events if e.get("is_next")), None)

    unfinished = [f for f in fixtures if not f.get("finished") and f.get("event") is not None]
    if next_event is None and unfinished:
        next_event = min(int(f["event"]) for f in unfinished)
    event_ids = sorted({int(f["event"]) for f in unfinished})
    if next_event is not None:
        event_ids = [event for event in event_ids if event >= int(next_event)]
    event_ids = event_ids[: max(1, int(horizon))]

    context: dict[int, dict[str, Any]] = {}
    for team_id in team_names:
        scenarios: list[dict[str, Any]] = []
        for period_index, event_id in enumerate(event_ids):
            team_fixtures = [
                fixture
                for fixture in unfinished
                if int(fixture.get("event") or -1) == event_id
                and (fixture.get("team_h") == team_id or fixture.get("team_a") == team_id)
            ]
            for fixture in sorted(
                team_fixtures,
                key=lambda value: _parse_datetime(value.get("kickoff_time"))
                or datetime.max.replace(tzinfo=timezone.utc),
            ):
                is_home = fixture.get("team_h") == team_id
                opponent_id = fixture.get("team_a") if is_home else fixture.get("team_h")
                scenarios.append(
                    {
                        "fixture_difficulty": safe_float(
                            fixture.get("team_h_difficulty" if is_home else "team_a_difficulty"),
                            3.0,
                        ),
                        "home": 1.0 if is_home else 0.0,
                        "opponent_strength": (
                            safe_float(
                                fixture.get("team_h_difficulty" if is_home else "team_a_difficulty"),
                                3.0,
                            )
                            - 1.0
                        )
                        / 4.0,
                        "next_opponent": team_names.get(int(opponent_id), str(opponent_id)),
                        "next_kickoff": str(fixture.get("kickoff_time") or ""),
                        "fixture_count": 1.0,
                        "rest_days": 7.0,
                        "fixture_id": fixture.get("id"),
                        "period_index": period_index,
                    }
                )

        if not scenarios:
            context[team_id] = {
                "fixture_difficulty": 3.0,
                "home": 0.5,
                "next_opponent": "TBD",
                "next_kickoff": "",
                "fixture_count": 0.0,
                "next_fixture_ids": [],
                "fixture_scenarios": [],
            }
            continue
        first = scenarios[0]
        context[team_id] = {
            "fixture_difficulty": first["fixture_difficulty"],
            "home": first["home"],
            "opponent_strength": first["opponent_strength"],
            "next_opponent": " / ".join(scenario["next_opponent"] for scenario in scenarios),
            "next_kickoff": min(
                (scenario["next_kickoff"] for scenario in scenarios if scenario["next_kickoff"]),
                default="",
            ),
            "fixture_count": float(len(scenarios)),
            "next_fixture_ids": [scenario["fixture_id"] for scenario in scenarios if scenario.get("fixture_id")],
            "fixture_scenarios": scenarios,
        }
    return context


def fetch_fpl_players(horizon: int = 1) -> pd.DataFrame:
    bootstrap = _get_json(f"{FPL_BASE}/bootstrap-static/")
    fixtures_payload = _get_json(f"{FPL_BASE}/fixtures/")
    fixtures = fixtures_payload if isinstance(fixtures_payload, list) else []
    teams = {int(t["id"]): t for t in bootstrap.get("teams", [])}
    fixture_context = _fpl_fixture_context(bootstrap, fixtures, horizon=horizon)
    observed_matches = fpl_team_matches_observed(bootstrap, fixtures)
    rows: list[dict[str, Any]] = []

    for p in bootstrap.get("elements", []):
        team_id = int(p.get("team") or 0)
        team = teams.get(team_id, {})
        context = fixture_context.get(team_id, {})
        minutes = safe_float(p.get("minutes"))
        starts = safe_float(p.get("starts"))
        team_matches = observed_matches.get(team_id, 38)
        appearances = estimate_fpl_appearances(
            minutes,
            starts,
            p.get("subbed_in"),
            team_matches=team_matches,
        )
        chance = p.get("chance_of_playing_next_round")
        if chance is None:
            chance = 100 if p.get("status") == "a" else 50

        strength_values = [
            safe_float(team.get("strength_overall_home"), 1000.0),
            safe_float(team.get("strength_overall_away"), 1000.0),
        ]
        team_strength = float(np.mean(strength_values)) / 1500.0
        expected_goals = safe_float(p.get("expected_goals"))
        expected_assists = safe_float(p.get("expected_assists"))

        rows.append(
            {
                "player_id": p.get("id"),
                "name": p.get("web_name"),
                "club": team.get("name", str(team_id)),
                "position": FPL_POSITION_MAP.get(p.get("element_type"), "MID"),
                "price": safe_float(p.get("now_cost")) / 10.0,
                "minutes": minutes,
                "appearances": appearances,
                "starts": starts,
                "start_probability": estimate_start_probability(starts, team_matches),
                "team_matches_observed": float(team_matches),
                "rating": 5.8 + max(safe_float(p.get("ict_index_rank_type"), 300.0), 1.0) ** -0.18,
                "goals": safe_float(p.get("goals_scored")),
                "assists": safe_float(p.get("assists")),
                "clean_sheets": safe_float(p.get("clean_sheets")),
                "saves": safe_float(p.get("saves")),
                "yellow_cards": safe_float(p.get("yellow_cards")),
                "red_cards": safe_float(p.get("red_cards")),
                "xg": expected_goals,
                "xa": expected_assists,
                "form": safe_float(p.get("form")),
                "total_points": safe_float(p.get("total_points")),
                "bonus": safe_float(p.get("bonus")),
                "bps": safe_float(p.get("bps")),
                "ict_index": safe_float(p.get("ict_index")),
                "selected_by_percent": safe_float(p.get("selected_by_percent")),
                "chance_playing": safe_float(chance, 100.0) / 100.0,
                "injury_reason": p.get("news") or "",
                "fixture_difficulty": safe_float(context.get("fixture_difficulty"), 3.0),
                "home": safe_float(context.get("home"), 0.5),
                "next_opponent": context.get("next_opponent", "TBD"),
                "next_kickoff": context.get("next_kickoff", ""),
                "fixture_count": safe_float(context.get("fixture_count"), 1.0),
                "next_fixture_ids": context.get("next_fixture_ids", []),
                "fixture_scenarios": context.get("fixture_scenarios", []),
                "team_strength": team_strength,
                "team_form_points": 1.5,
                "team_attack_form": 1.35,
                "team_defence_form": 1.35,
                "opponent_strength": (safe_float(context.get("fixture_difficulty"), 3.0) - 1.0) / 4.0,
                "rest_days": 7.0,
                "lineup_status": "",
                "price_source": "Official FPL price",
                "data_source": "Official FPL live API",
            }
        )

    if not rows:
        raise DataProviderError("FPL returned no player data. The game may be between seasons.")
    return pd.DataFrame(rows)


def _api_headers(api_key: str) -> dict:
    if not api_key.strip():
        raise DataProviderError(
            "A free API-Football key is required for live non-FPL player data. "
            "Add API_FOOTBALL_KEY to Streamlit secrets or paste it in the sidebar."
        )
    return {"x-apisports-key": api_key.strip()}


def fetch_api_football_players(api_key: str, league_id: int, season: int, max_pages: int = 30) -> pd.DataFrame:
    headers = _api_headers(api_key)
    rows: list[dict[str, Any]] = []
    page = 1

    while page <= max_pages:
        payload = _get_json(
            f"{API_FOOTBALL_BASE}/players",
            headers=headers,
            params={"league": league_id, "season": season, "page": page},
        )
        _check_api_errors(payload)
        response_rows = payload.get("response", [])

        for item in response_rows:
            player = item.get("player", {})
            for s in item.get("statistics", []) or []:
                games = s.get("games", {}) or {}
                goals = s.get("goals", {}) or {}
                cards = s.get("cards", {}) or {}
                shots = s.get("shots", {}) or {}
                passes = s.get("passes", {}) or {}
                tackles = s.get("tackles", {}) or {}
                duels = s.get("duels", {}) or {}
                penalty = s.get("penalty", {}) or {}
                team = s.get("team", {}) or {}
                appearances = safe_float(games.get("appearences"))
                starts = safe_float(games.get("lineups"))
                minutes = safe_float(games.get("minutes"))
                rating = safe_float(games.get("rating"), 6.0)
                position = API_POSITION_MAP.get(games.get("position"), "MID")
                goals_total = safe_float(goals.get("total"))
                assists_total = safe_float(goals.get("assists"))
                shots_total = safe_float(shots.get("total"))
                shots_on_target = safe_float(shots.get("on"))
                key_passes = safe_float(passes.get("key"))
                # API-Football's free player feed does not expose provider xG/xA here.
                # These conservative volume proxies let the supervised FPL model use
                # the free shot and key-pass fields instead of treating every value as zero.
                xg_proxy = max(0.08 * shots_total + 0.12 * shots_on_target, 0.70 * goals_total)
                xa_proxy = max(0.11 * key_passes, 0.70 * assists_total)

                rows.append(
                    {
                        "player_id": player.get("id"),
                        "name": player.get("name"),
                        "club": team.get("name") or "Unknown",
                        "team_id": team.get("id"),
                        "position": position,
                        "price": np.nan,
                        "minutes": minutes,
                        "appearances": appearances,
                        "starts": starts,
                        "start_probability": starts / max(appearances, 1.0),
                        "rating": rating,
                        "goals": goals_total,
                        "assists": assists_total,
                        "clean_sheets": 0.0,
                        "saves": safe_float(goals.get("saves")),
                        "yellow_cards": safe_float(cards.get("yellow")) + safe_float(cards.get("yellowred")),
                        "red_cards": safe_float(cards.get("red")),
                        "xg": xg_proxy,
                        "xa": xa_proxy,
                        "form": rating,
                        "total_points": 0.0,
                        "bonus": 0.0,
                        "bps": 0.0,
                        "ict_index": 0.0,
                        "selected_by_percent": 0.0,
                        "shots": shots_total,
                        "shots_on_target": shots_on_target,
                        "key_passes": key_passes,
                        "tackles": safe_float(tackles.get("total")),
                        "interceptions": safe_float(tackles.get("interceptions")),
                        "duels_won": safe_float(duels.get("won")),
                        "penalties_scored": safe_float(penalty.get("scored")),
                        "penalties_missed": safe_float(penalty.get("missed")),
                        "saves": safe_float(goals.get("saves")),
                        "chance_playing": 1.0,
                        "injury_reason": "",
                        "fixture_difficulty": 3.0,
                        "home": 0.5,
                        "next_opponent": "TBD",
                        "next_kickoff": "",
                        "fixture_count": 1.0,
                        "next_fixture_ids": [],
                        "fixture_scenarios": [],
                        "team_strength": 0.5,
                        "team_form_points": 1.5,
                        "team_attack_form": 1.35,
                        "team_defence_form": 1.35,
                        "opponent_strength": 0.5,
                        "rest_days": 7.0,
                        "lineup_status": "",
                        "price_source": "Missing",
                        "data_source": "API-Football live",
                    }
                )

        paging = payload.get("paging", {}) or {}
        total_pages = int(paging.get("total", 1) or 1)
        if page >= total_pages or not response_rows:
            break
        page += 1

    if not rows:
        raise DataProviderError(
            "No player statistics were returned. Check the season, league coverage, and API plan."
        )
    result = pd.DataFrame(rows)
    result = (
        result.sort_values(["player_id", "minutes"], ascending=[True, False])
        .drop_duplicates(subset=["player_id"], keep="first")
    )
    return result.reset_index(drop=True)


def fetch_api_football_injuries(api_key: str, league_id: int, season: int) -> pd.DataFrame:
    headers = _api_headers(api_key)
    payload = _get_json(
        f"{API_FOOTBALL_BASE}/injuries",
        headers=headers,
        params={"league": league_id, "season": season},
    )
    _check_api_errors(payload, "API-Football injury feed")

    rows: list[dict[str, Any]] = []
    for item in payload.get("response", []):
        p = item.get("player", {}) or {}
        team = item.get("team", {}) or {}
        fixture = item.get("fixture", {}) or {}
        rows.append(
            {
                "player_id": p.get("id"),
                "name": p.get("name"),
                "club": team.get("name"),
                "injury_type": p.get("type") or "Unavailable",
                "injury_reason": p.get("reason") or "Not specified",
                "fixture_date": fixture.get("date"),
            }
        )
    return pd.DataFrame(rows)


def fetch_api_football_fixtures(api_key: str, league_id: int, season: int) -> list[dict[str, Any]]:
    headers = _api_headers(api_key)
    payload = _get_json(
        f"{API_FOOTBALL_BASE}/fixtures",
        headers=headers,
        params={"league": league_id, "season": season},
    )
    _check_api_errors(payload, "API-Football fixtures feed")
    return list(payload.get("response", []) or [])


def _football_data_headers(token: str) -> dict[str, str]:
    if not token.strip():
        raise DataProviderError(
            "A free football-data.org token is required for the secondary fixture feed."
        )
    return {"X-Auth-Token": token.strip()}


def fetch_football_data_matches(
    token: str,
    competition_code: str,
    season: int,
) -> list[dict[str, Any]]:
    payload = _get_json(
        f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/matches",
        headers=_football_data_headers(token),
        params={"season": int(season)},
    )
    if not isinstance(payload, dict):
        raise DataProviderError("football-data.org returned an unexpected response.")
    message = payload.get("message")
    if message and not payload.get("matches"):
        raise DataProviderError(f"football-data.org error: {message}")
    return list(payload.get("matches", []) or [])


def _football_data_as_fixtures(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for match in matches:
        home = match.get("homeTeam", {}) or {}
        away = match.get("awayTeam", {}) or {}
        score = (match.get("score", {}) or {}).get("fullTime", {}) or {}
        status = str(match.get("status") or "SCHEDULED").upper()
        fixtures.append(
            {
                "fixture": {
                    "id": match.get("id"),
                    "date": match.get("utcDate"),
                    "status": {"short": "FT" if status == "FINISHED" else status},
                },
                "league": {"round": f"Matchday {match.get('matchday')}"},
                "teams": {
                    "home": {"id": home.get("id"), "name": home.get("name") or home.get("shortName")},
                    "away": {"id": away.get("id"), "name": away.get("name") or away.get("shortName")},
                },
                "goals": {"home": score.get("home"), "away": score.get("away")},
            }
        )
    return fixtures


def build_football_data_context(
    matches: list[dict[str, Any]],
    horizon: int = 1,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    return build_team_context(
        _football_data_as_fixtures(matches),
        horizon=horizon,
        now=now,
    )


def _round_number(value: object) -> Optional[int]:
    numbers = re.findall(r"\d+", str(value or ""))
    return int(numbers[-1]) if numbers else None


def _periodize_upcoming(
    matches: list[dict[str, Any]],
    horizon: int,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(matches, key=lambda item: item["date"])
    periods: list[list[dict[str, Any]]] = []
    for match in ordered:
        if not periods:
            periods.append([match])
            continue
        previous = periods[-1][-1]
        same_round = str(match.get("round") or "") == str(previous.get("round") or "")
        days_apart = abs((match["date"] - previous["date"]).total_seconds()) / 86400.0
        current_round = _round_number(match.get("round"))
        previous_round = _round_number(previous.get("round"))
        rescheduled_double = (
            days_apart <= 5.0
            and current_round is not None
            and previous_round is not None
            and abs(current_round - previous_round) > 1
        )
        if same_round or rescheduled_double:
            periods[-1].append(match)
        else:
            periods.append([match])
    return periods[: max(1, int(horizon))]


def build_team_context(
    fixtures: list[dict[str, Any]],
    horizon: int = 1,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    teams: dict[int, str] = {}
    completed: dict[int, list[dict[str, Any]]] = {}
    upcoming: dict[int, list[dict[str, Any]]] = {}

    for item in fixtures:
        fixture = item.get("fixture", {}) or {}
        status = (fixture.get("status", {}) or {}).get("short")
        kickoff = _parse_datetime(fixture.get("date"))
        round_label = (item.get("league", {}) or {}).get("round") or fixture.get("round")
        team_block = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        home = team_block.get("home", {}) or {}
        away = team_block.get("away", {}) or {}
        if not home.get("id") or not away.get("id"):
            continue
        home_id, away_id = int(home["id"]), int(away["id"])
        teams[home_id], teams[away_id] = home.get("name", str(home_id)), away.get("name", str(away_id))

        if status in COMPLETED_STATUSES and goals.get("home") is not None and goals.get("away") is not None:
            home_goals, away_goals = safe_float(goals.get("home")), safe_float(goals.get("away"))
            home_points = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
            away_points = 3.0 if away_goals > home_goals else 1.0 if home_goals == away_goals else 0.0
            completed.setdefault(home_id, []).append(
                {"date": kickoff, "gf": home_goals, "ga": away_goals, "points": home_points}
            )
            completed.setdefault(away_id, []).append(
                {"date": kickoff, "gf": away_goals, "ga": home_goals, "points": away_points}
            )
        elif kickoff and kickoff >= now:
            upcoming.setdefault(home_id, []).append(
                {
                    "date": kickoff,
                    "opponent_id": away_id,
                    "home": 1.0,
                    "round": round_label,
                    "fixture_id": fixture.get("id"),
                }
            )
            upcoming.setdefault(away_id, []).append(
                {
                    "date": kickoff,
                    "opponent_id": home_id,
                    "home": 0.0,
                    "round": round_label,
                    "fixture_id": fixture.get("id"),
                }
            )

    raw_strength: dict[int, float] = {}
    summaries: dict[int, dict[str, float]] = {}
    for team_id in teams:
        matches = sorted(completed.get(team_id, []), key=lambda x: x["date"] or datetime.min.replace(tzinfo=timezone.utc))
        recent = matches[-5:]
        sample = recent or matches
        games = max(len(sample), 1)
        ppg = sum(m["points"] for m in sample) / games
        gfpg = sum(m["gf"] for m in sample) / games
        gapg = sum(m["ga"] for m in sample) / games
        raw_strength[team_id] = 0.52 * (ppg / 3.0) + 0.28 * min(gfpg / 2.5, 1.0) + 0.20 * (1.0 - min(gapg / 2.5, 1.0))
        summaries[team_id] = {"ppg": ppg, "gfpg": gfpg, "gapg": gapg}

    values = list(raw_strength.values())
    min_strength = min(values) if values else 0.0
    max_strength = max(values) if values else 1.0
    span = max(max_strength - min_strength, 1e-6)
    normalized = {team_id: (value - min_strength) / span for team_id, value in raw_strength.items()}

    rows: list[dict[str, Any]] = []
    for team_id, club in teams.items():
        next_matches = sorted(upcoming.get(team_id, []), key=lambda x: x["date"])
        played = sorted(completed.get(team_id, []), key=lambda x: x["date"] or datetime.min.replace(tzinfo=timezone.utc))
        last_date = played[-1]["date"] if played else None
        summary = summaries.get(team_id, {"ppg": 1.5, "gfpg": 1.35, "gapg": 1.35})

        scenarios: list[dict[str, Any]] = []
        previous_date = last_date
        for period_index, period in enumerate(_periodize_upcoming(next_matches, horizon)):
            for match in period:
                opponent_id = int(match["opponent_id"])
                home = safe_float(match.get("home"), 0.5)
                opponent_strength = normalized.get(opponent_id, 0.5)
                difficulty = float(
                    np.clip(
                        1.0 + 4.0 * opponent_strength - (0.25 if home == 1.0 else 0.0),
                        1.0,
                        5.0,
                    )
                )
                rest_days = (match["date"] - previous_date).days if previous_date else 7
                scenarios.append(
                    {
                        "fixture_difficulty": round(difficulty, 2),
                        "home": home,
                        "opponent_strength": opponent_strength,
                        "next_opponent": teams.get(opponent_id, "TBD"),
                        "next_kickoff": match["date"].isoformat(),
                        "fixture_count": 1.0,
                        "rest_days": float(np.clip(rest_days, 2, 21)),
                        "fixture_id": match.get("fixture_id"),
                        "period_index": period_index,
                    }
                )
                previous_date = match["date"]

        first = scenarios[0] if scenarios else None
        rows.append(
            {
                "club": club,
                "team_id": team_id,
                "team_strength": normalized.get(team_id, 0.5),
                "team_form_points": summary["ppg"],
                "team_attack_form": summary["gfpg"],
                "team_defence_form": summary["gapg"],
                "team_matches_observed": float(len(played)),
                "opponent_strength": first["opponent_strength"] if first else 0.5,
                "fixture_difficulty": first["fixture_difficulty"] if first else 3.0,
                "home": first["home"] if first else 0.5,
                "next_opponent": " / ".join(item["next_opponent"] for item in scenarios) or "TBD",
                "next_kickoff": min(
                    (item["next_kickoff"] for item in scenarios),
                    default="",
                ),
                "fixture_count": float(len(scenarios)),
                "rest_days": first["rest_days"] if first else 7.0,
                "next_fixture_ids": [
                    item["fixture_id"] for item in scenarios if item.get("fixture_id")
                ],
                "fixture_scenarios": scenarios,
            }
        )
    return pd.DataFrame(rows)


def apply_team_context(players: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if context is None or context.empty:
        return players.copy()
    result = players.copy()
    context = context.copy()
    context["_club_key"] = context["club"].map(normalize_club)
    result["_club_key"] = result["club"].map(normalize_club)
    update_cols = [
        "team_strength", "team_form_points", "team_attack_form", "team_defence_form",
        "opponent_strength", "fixture_difficulty", "home", "next_opponent", "next_kickoff",
        "fixture_count", "rest_days", "next_fixture_ids", "fixture_scenarios",
        "team_matches_observed",
    ]
    lookup = context.drop_duplicates("_club_key").set_index("_club_key")
    context_keys = [str(key) for key in lookup.index if key]

    def resolve_key(player_key: object) -> Optional[str]:
        key = str(player_key or "")
        if not key:
            return None
        if key in lookup.index:
            return key
        contained = [
            candidate
            for candidate in context_keys
            if min(len(key), len(candidate)) >= 5 and (key in candidate or candidate in key)
        ]
        if len(contained) == 1:
            return contained[0]
        scores = sorted(
            ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in context_keys),
            reverse=True,
        )
        if scores and scores[0][0] >= 0.82 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08):
            return scores[0][1]
        return None

    result["_context_key"] = result["_club_key"].map(resolve_key)
    for col in update_cols:
        if col not in lookup.columns:
            continue
        mapped = result["_context_key"].map(lookup[col])
        result.loc[mapped.notna(), col] = mapped[mapped.notna()]

    # Global correction: start probability is starts / TEAM MATCHES, not starts /
    # player appearances.  This is what prevents Marmoush/Jesus-style rotation
    # players from being promoted to near-certain starters.
    if {"starts", "team_matches_observed"}.issubset(result.columns):
        starts = pd.to_numeric(result["starts"], errors="coerce").fillna(0.0)
        matches = pd.to_numeric(result["team_matches_observed"], errors="coerce")
        valid = matches.notna() & (matches >= 0)
        result.loc[valid, "start_probability"] = [
            estimate_start_probability(st, mt)
            for st, mt in zip(starts[valid], matches[valid])
        ]
    return result.drop(columns=["_club_key", "_context_key"])


def merge_injuries(players: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    result = players.copy()
    if injuries is None or injuries.empty:
        return result

    injury_by_id = (
        injuries.dropna(subset=["player_id"])
        .sort_values("fixture_date", na_position="first")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")
    )
    for idx, row in result.iterrows():
        player_id = row.get("player_id")
        match = None
        if player_id in injury_by_id.index:
            match = injury_by_id.loc[player_id]
        else:
            norm = normalize_name(row.get("name"))
            candidates = injuries[injuries["name"].map(normalize_name) == norm]
            if not candidates.empty:
                match = candidates.iloc[-1]
        if match is not None:
            result.at[idx, "chance_playing"] = min(safe_float(row.get("chance_playing"), 1.0), 0.15)
            reason = str(match.get("injury_reason", "Unavailable"))
            injury_type = str(match.get("injury_type", "Unavailable"))
            result.at[idx, "injury_reason"] = f"{injury_type}: {reason}"
    return result



def fetch_api_football_lineups(api_key: str, fixture_ids: Iterable[int]) -> pd.DataFrame:
    """Fetch confirmed lineups for a small set of fixture IDs.

    API-Football publishes lineups shortly before kickoff when a competition
    supports them. Empty responses are normal before lineups are released.
    """
    headers = _api_headers(api_key)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_fixture_id in fixture_ids:
        try:
            fixture_id = int(raw_fixture_id)
        except (TypeError, ValueError):
            continue
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        payload = _get_json(
            f"{API_FOOTBALL_BASE}/fixtures/lineups",
            headers=headers,
            params={"fixture": fixture_id},
        )
        _check_api_errors(payload, "API-Football lineup feed")
        for team_lineup in payload.get("response", []) or []:
            team = team_lineup.get("team", {}) or {}
            club = team.get("name") or "Unknown"
            team_id = team.get("id")
            for status, key in (("starter", "startXI"), ("bench", "substitutes")):
                for item in team_lineup.get(key, []) or []:
                    player = item.get("player", {}) or {}
                    if not player.get("name"):
                        continue
                    rows.append(
                        {
                            "fixture_id": fixture_id,
                            "team_id": team_id,
                            "club": club,
                            "player_id": player.get("id"),
                            "name": player.get("name"),
                            "lineup_status": status,
                        }
                    )
    return pd.DataFrame(rows)


def near_kickoff_fixture_ids(
    fixtures: list[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    hours_ahead: float = 3.0,
) -> list[int]:
    """Return upcoming/recent fixture IDs worth checking for confirmed lineups."""
    now = now or datetime.now(timezone.utc)
    ids: list[int] = []
    for item in fixtures:
        fixture = item.get("fixture", {}) or {}
        kickoff = _parse_datetime(fixture.get("date"))
        status = str((fixture.get("status", {}) or {}).get("short") or "").upper()
        if not kickoff or status in COMPLETED_STATUSES:
            continue
        delta_hours = (kickoff - now).total_seconds() / 3600.0
        if -2.0 <= delta_hours <= float(hours_ahead):
            try:
                ids.append(int(fixture.get("id")))
            except (TypeError, ValueError):
                continue
    return ids

def merge_price_file(players: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return players
    columns = {str(c).strip().lower(): c for c in price_df.columns}
    name_col = columns.get("name") or columns.get("player") or columns.get("player_name")
    price_col = columns.get("price") or columns.get("value") or columns.get("cost")
    if name_col is None or price_col is None:
        raise DataProviderError("Price CSV must contain name and price columns.")

    lookup = price_df[[name_col, price_col]].copy()
    lookup["_key"] = lookup[name_col].map(normalize_name)
    lookup[price_col] = pd.to_numeric(lookup[price_col], errors="coerce")
    lookup = lookup.dropna(subset=[price_col]).drop_duplicates("_key", keep="last")
    price_map = lookup.set_index("_key")[price_col].to_dict()

    result = players.copy()
    keys = result["name"].map(normalize_name)
    matched = keys.map(price_map)
    result.loc[matched.notna(), "price"] = matched[matched.notna()].astype(float)
    result.loc[matched.notna(), "price_source"] = "Uploaded official fantasy price"
    return result


def estimate_missing_prices(players: pd.DataFrame, budget: float, squad_size: int) -> pd.DataFrame:
    result = players.copy()
    result["price"] = pd.to_numeric(result.get("price"), errors="coerce")
    missing = result["price"].isna() | (result["price"] <= 0)
    if not missing.any():
        return result

    def numeric_series(column: str, default: float) -> pd.Series:
        if column in result.columns:
            return pd.to_numeric(result[column], errors="coerce").fillna(default)
        return pd.Series(default, index=result.index, dtype=float)

    minutes = numeric_series("minutes", 0.0).clip(lower=1.0)
    appearances = numeric_series("appearances", 0.0).clip(lower=1.0)
    rating = numeric_series("rating", 6.0)
    goals = numeric_series("goals", 0.0)
    assists = numeric_series("assists", 0.0)
    starts = numeric_series("starts", 0.0)
    team_strength = numeric_series("team_strength", 0.5)
    goals_p90 = 90.0 * goals / minutes
    assists_p90 = 90.0 * assists / minutes
    start_rate = (starts / appearances).clip(0, 1)
    raw = 0.38 * (rating - 5.5).clip(lower=0.0) + 2.4 * goals_p90 + 1.6 * assists_p90 + 0.7 * start_rate + 0.5 * team_strength
    result["_price_score"] = raw

    average = budget / max(squad_size, 1)
    ranges = {
        "GK": (0.55 * average, 1.05 * average),
        "DEF": (0.58 * average, 1.28 * average),
        "MID": (0.62 * average, 1.95 * average),
        "FWD": (0.62 * average, 2.05 * average),
    }
    for position, group in result.groupby("position"):
        indexes = group.index.intersection(result.index[missing])
        if indexes.empty:
            continue
        percentile = result.loc[indexes, "_price_score"].rank(pct=True, method="average").fillna(0.5)
        floor, ceiling = ranges.get(str(position).upper(), (0.6 * average, 1.7 * average))
        values = floor + np.power(percentile, 1.45) * (ceiling - floor)
        result.loc[indexes, "price"] = values.clip(lower=3.5, upper=15.0).round(1)
        result.loc[indexes, "price_source"] = "Statistical fallback price — not official"

    return result.drop(columns=["_price_score"])


def load_uploaded_players(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    required = {"name", "club", "position", "price"}
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise DataProviderError(f"Player CSV is missing: {', '.join(sorted(missing))}")
    if df.empty:
        raise DataProviderError("Player CSV contains no player rows.")
    defaults: dict[str, Any] = {
        "player_id": range(1, len(df) + 1),
        "minutes": 0.0,
        "appearances": 0.0,
        "starts": 0.0,
        "start_probability": 0.5,
        "rating": 6.0,
        "goals": 0.0,
        "assists": 0.0,
        "clean_sheets": 0.0,
        "saves": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
        "xg": 0.0,
        "xa": 0.0,
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
        "next_fixture_ids": [],
        "fixture_scenarios": [],
        "team_strength": 0.5,
        "team_form_points": 1.5,
        "team_attack_form": 1.35,
        "team_defence_form": 1.35,
        "opponent_strength": 0.5,
        "rest_days": 7.0,
        "lineup_status": "",
        "price_source": "Uploaded",
        "data_source": "Uploaded",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            if isinstance(default, range):
                df[col] = list(default)
            elif isinstance(default, list):
                df[col] = [list(default) for _ in range(len(df))]
            else:
                df[col] = default
    df["position"] = df["position"].astype(str).str.upper().replace(
        {
            "ATT": "FWD", "FW": "FWD", "ST": "FWD",
            "AM": "MID", "CM": "MID", "DM": "MID", "RM": "MID", "LM": "MID",
            "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
            "GOALKEEPER": "GK",
        }
    )
    return df
