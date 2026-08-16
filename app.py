from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.config import COMPETITIONS
from src.demo import demo_history, demo_players
from src.history import load_bundled_history
from src.model import estimate_missing_prices_ml, predict_players
from src.optimizer import OptimizerError, optimize_team
from src.public_enrichment import merge_public_enrichment
from src.lineup_intelligence import merge_confirmed_lineups, merge_lineup_consensus, merge_recent_lineup_history
from src.providers import (
    DataProviderError,
    apply_team_context,
    build_football_data_context,
    build_team_context,
    estimate_missing_prices,
    fetch_api_football_fixtures,
    fetch_api_football_injuries,
    fetch_api_football_lineups,
    fetch_api_football_players,
    fetch_football_data_matches,
    fetch_fpl_players,
    load_uploaded_players,
    merge_injuries,
    merge_price_file,
    near_kickoff_fixture_ids,
)

st.set_page_config(page_title="Fantasy XI ML", page_icon="⚽", layout="wide")


def _configured_value(*names: str) -> str:
    env_value = next((os.getenv(name, "") for name in names if os.getenv(name, "")), "")
    try:
        secret_value = next((str(st.secrets.get(name, "")) for name in names if st.secrets.get(name, "")), "")
    except Exception:
        secret_value = ""
    return secret_value or env_value


@st.cache_data(ttl=900, show_spinner=False)
def cached_fpl_players(horizon: int) -> pd.DataFrame:
    return fetch_fpl_players(horizon=horizon)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_api_players(api_key: str, league_id: int, season: int) -> pd.DataFrame:
    # The player endpoint is paginated, so a long cache protects the free allowance.
    return fetch_api_football_players(api_key, league_id, season)


@st.cache_data(ttl=900, show_spinner=False)
def cached_injuries(api_key: str, league_id: int, season: int) -> pd.DataFrame:
    return fetch_api_football_injuries(api_key, league_id, season)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fixtures(api_key: str, league_id: int, season: int) -> list[dict]:
    return fetch_api_football_fixtures(api_key, league_id, season)


@st.cache_data(ttl=180, show_spinner=False)
def cached_confirmed_lineups(api_key: str, fixture_ids: tuple[int, ...]) -> pd.DataFrame:
    return fetch_api_football_lineups(api_key, fixture_ids)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_football_data_matches(token: str, competition_code: str, season: int) -> list[dict]:
    return fetch_football_data_matches(token, competition_code, season)


@st.cache_data(show_spinner=False)
def cached_bundled_history() -> pd.DataFrame:
    return load_bundled_history()


def csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))


def fixture_count_from_players(players: pd.DataFrame) -> int:
    fixture_ids: set[int] = set()
    if "next_fixture_ids" not in players.columns:
        return 0
    for values in players["next_fixture_ids"]:
        if isinstance(values, (list, tuple)):
            fixture_ids.update(int(value) for value in values if value)
    return len(fixture_ids)


def render_team_pitch(team: pd.DataFrame, captain: str, vice: str) -> None:
    for position in ["GK", "DEF", "MID", "FWD"]:
        row = team[team["position"] == position].sort_values("predicted_points", ascending=False)
        if row.empty:
            continue
        columns = st.columns(len(row))
        for column, (_, player) in zip(columns, row.iterrows()):
            marker = " (C)" if player["name"] == captain else " (VC)" if player["name"] == vice else ""
            opponent = player.get("next_opponent", "TBD")
            column.markdown(
                f"**{player['name']}{marker}**  \n"
                f"{player['club']} · vs {opponent}  \n"
                f"{player['price']:.1f} · {player['predicted_points']:.2f} pts  \n"
                f"Start {100 * player.get('start_probability', 0):.0f}%"
            )


def choose_history(
    local_file,
    global_history: pd.DataFrame | None,
    use_bundled: bool,
    source: str,
) -> pd.DataFrame | None:
    if local_file is not None:
        return read_uploaded_csv(local_file)
    if global_history is not None:
        return global_history.copy()
    if use_bundled:
        return cached_bundled_history().copy()
    if source == "Demo":
        return demo_history()
    return None


def render_competition(
    name: str,
    api_key: str,
    football_data_token: str,
    season: int,
    global_history: pd.DataFrame | None,
    use_api_injury_backup: bool,
    use_confirmed_lineups: bool,
) -> None:
    config = COMPETITIONS[name]
    st.subheader(name)

    top = st.columns(4)
    if name == "Premier League":
        source_options = ["Live official FPL", "Demo", "Upload player CSV"]
    else:
        source_options = ["Live API-Football", "Demo", "Upload player CSV"]
    source = top[0].selectbox("Player data", source_options, key=f"source_{config.key}")
    budget = top[1].number_input(
        "Budget",
        min_value=10.0,
        max_value=500.0,
        value=config.default_budget,
        step=0.5,
        key=f"budget_{config.key}",
    )
    horizon = top[2].slider(
        "Prediction horizon",
        1,
        6,
        1,
        key=f"horizon_{config.key}",
        help="The model predicts each future fixture separately; later matchdays receive slightly less weight.",
    )
    max_per_club = top[3].number_input(
        "Max per club",
        min_value=1,
        max_value=config.squad_size,
        value=config.max_per_club,
        step=1,
        key=f"clubmax_{config.key}",
    )

    uploaded_players = None
    price_file = None
    if source == "Upload player CSV":
        uploaded_players = st.file_uploader("Upload player pool CSV", type=["csv"], key=f"players_{config.key}")
    elif source == "Live API-Football":
        price_file = st.file_uploader(
            "Optional official fantasy prices (CSV: name, price)",
            type=["csv"],
            key=f"prices_{config.key}",
            help="Otherwise a separate ML model estimates non-official prices.",
        )

    history_columns = st.columns(2)
    local_history_file = history_columns[0].file_uploader(
        "Optional replacement training CSV",
        type=["csv"],
        key=f"history_{config.key}",
        help="Use next-match labels such as future_points, future_started, future_appearance and future_minutes.",
    )
    use_bundled = history_columns[1].checkbox(
        "Use bundled real 2024/25 FPL multitask training data",
        value=True,
        key=f"bundled_history_{config.key}",
        help=(
            "Direct historical training for the Premier League. Other leagues use transfer learning, so "
            "competition-specific history will usually improve accuracy."
        ),
    )
    history = choose_history(local_history_file, global_history, use_bundled, source)
    public_enrichment_file = st.file_uploader(
        "Optional public-stat enrichment CSV",
        type=["csv"],
        key=f"public_enrichment_{config.key}",
        help=(
            "Use a CSV produced by scripts/fetch_public_soccer_data.py to fill season totals such as "
            "minutes, starts, goals, assists, xG and xA without replacing live fixtures or availability."
        ),
    )
    lineup_intelligence_file = st.file_uploader(
        "Optional predicted-lineup consensus CSV",
        type=["csv"],
        key=f"lineup_intelligence_{config.key}",
        help=(
            "Either one row per source (source,name,club,status) or aggregated "
            "name,club,start_probability,source_count. It is fused for every matching player."
        ),
    )
    recent_lineup_file = st.file_uploader(
        "Optional recent actual-lineup history CSV",
        type=["csv"],
        key=f"recent_lineups_{config.key}",
        help=(
            "Generate with scripts/fetch_recent_lineup_history.py. Recent manager selections are blended "
            "into the start prior for every matched player."
        ),
    )

    try:
        with st.spinner("Loading players, availability and fixture context..."):
            fixture_rows = 0
            fixture_provider = "Provided CSV"
            api_fixtures_for_lineups: list[dict] = []

            if source == "Live official FPL":
                players = cached_fpl_players(int(horizon))
                fixture_rows = fixture_count_from_players(players)
                fixture_provider = "Official FPL"
                if api_key and use_api_injury_backup:
                    try:
                        injuries = cached_injuries(api_key, config.api_league_id, season)
                        players = merge_injuries(players, injuries)
                    except DataProviderError as exc:
                        st.warning(f"Official FPL loaded, but the optional second injury feed failed: {exc}")
                if api_key and use_confirmed_lineups:
                    try:
                        api_fixtures_for_lineups = cached_fixtures(api_key, config.api_league_id, season)
                    except DataProviderError as exc:
                        st.warning(f"Confirmed-lineup fixture lookup failed: {exc}")

            elif source == "Live API-Football":
                if not api_key.strip():
                    st.info(
                        "Live non-Premier-League player statistics need a free API-Football key. "
                        "Add it in the sidebar, or use Demo / Upload mode."
                    )
                    return
                players = cached_api_players(api_key, config.api_league_id, season)

                context = pd.DataFrame()
                if football_data_token.strip():
                    try:
                        matches = cached_football_data_matches(
                            football_data_token,
                            config.football_data_code,
                            season,
                        )
                        fixture_rows = len(matches)
                        context = build_football_data_context(matches, horizon=int(horizon))
                        fixture_provider = "football-data.org"
                    except DataProviderError as exc:
                        st.warning(f"The secondary fixture feed failed, so API-Football is being used: {exc}")

                if context.empty:
                    fixtures = cached_fixtures(api_key, config.api_league_id, season)
                    api_fixtures_for_lineups = fixtures
                    fixture_rows = len(fixtures)
                    context = build_team_context(fixtures, horizon=int(horizon))
                    fixture_provider = "API-Football"
                players = apply_team_context(players, context)

                try:
                    injuries = cached_injuries(api_key, config.api_league_id, season)
                    players = merge_injuries(players, injuries)
                except DataProviderError as exc:
                    st.warning(f"Players and fixtures loaded, but the injury feed failed: {exc}")

                if price_file is not None:
                    players = merge_price_file(players, read_uploaded_csv(price_file))
                if use_confirmed_lineups and not api_fixtures_for_lineups:
                    try:
                        api_fixtures_for_lineups = cached_fixtures(api_key, config.api_league_id, season)
                    except DataProviderError as exc:
                        st.warning(f"Confirmed-lineup fixture lookup failed: {exc}")

            elif source == "Upload player CSV":
                if uploaded_players is None:
                    st.info("Upload a player CSV to continue.")
                    return
                players = load_uploaded_players(io.BytesIO(uploaded_players.getvalue()))

            else:
                players = demo_players(name)
                fixture_rows = int(horizon)
                fixture_provider = "Demo scenarios"

            if public_enrichment_file is not None:
                players = merge_public_enrichment(players, read_uploaded_csv(public_enrichment_file))
                matched = int(players.get("public_data_match", pd.Series(False, index=players.index)).sum())
                st.caption(f"Public-stat enrichment matched {matched} players in this pool.")

            if recent_lineup_file is not None:
                players = merge_recent_lineup_history(players, read_uploaded_csv(recent_lineup_file))
                matched_recent = int(pd.to_numeric(
                    players.get("recent_lineup_matches", pd.Series(0.0, index=players.index)),
                    errors="coerce",
                ).fillna(0.0).gt(0).sum())
                st.caption(f"Recent actual-lineup history matched {matched_recent} players in this pool.")

            if lineup_intelligence_file is not None:
                players = merge_lineup_consensus(players, read_uploaded_csv(lineup_intelligence_file))
                matched = int(pd.to_numeric(
                    players.get("lineup_source_count", pd.Series(0.0, index=players.index)),
                    errors="coerce",
                ).fillna(0.0).gt(0).sum())
                st.caption(f"Predicted-lineup consensus matched {matched} players in this pool.")

            if api_key and use_confirmed_lineups and api_fixtures_for_lineups:
                near_ids = near_kickoff_fixture_ids(api_fixtures_for_lineups, hours_ahead=3.0)
                if near_ids:
                    try:
                        confirmed = cached_confirmed_lineups(api_key, tuple(near_ids[:12]))
                        if not confirmed.empty:
                            players = merge_confirmed_lineups(players, confirmed)
                            clubs = confirmed["club"].dropna().nunique()
                            st.caption(f"Confirmed lineup data applied for {clubs} club(s).")
                    except DataProviderError as exc:
                        st.warning(f"Confirmed-lineup lookup failed: {exc}")

            players = estimate_missing_prices_ml(players, history, budget, config.squad_size)
            players = estimate_missing_prices(players, budget, config.squad_size)

        with st.spinner("Loading the pre-trained ML projection..."):
            result = predict_players(players, history, horizon=int(horizon))
        predicted = result.players.copy()
        predicted["value_score"] = predicted["predicted_points"] / predicted["price"].clip(lower=0.1)

        controls = st.columns(2)
        minimum_start = controls[0].slider(
            "Minimum ML start probability",
            0.0,
            0.95,
            0.60,
            0.05,
            key=f"appearance_{config.key}",
            help="Filters substitutes, rotation options and likely absences before optimization.",
        )
        risk_aversion = controls[1].slider(
            "Uncertainty penalty",
            0.0,
            1.0,
            0.30,
            0.05,
            key=f"risk_{config.key}",
            help="Higher values prefer steadier projections over volatile upside.",
        )
        eligible = predicted[predicted["start_probability"] >= minimum_start].copy()

        price_sources = predicted["price_source"].fillna("").astype(str)
        exact_prices = (
            price_sources.str.contains("official fpl|uploaded", case=False, regex=True)
            & ~price_sources.str.contains("not official", case=False, regex=False)
        )
        metrics = st.columns(6)
        metrics[0].metric("Players", len(predicted))
        metrics[1].metric("Fixture rows", fixture_rows)
        metrics[2].metric("Official-price coverage", f"{100 * exact_prices.mean():.0f}%")
        metrics[3].metric("Training rows", f"{result.training_rows:,}")
        metrics[4].metric("Point MAE", f"{result.validation_mae:.2f}" if result.validation_mae is not None else "Fallback")
        metrics[5].metric("Start Brier", f"{result.validation_start_brier:.3f}" if result.validation_start_brier is not None else "N/A")
        second_metrics = st.columns(3)
        second_metrics[0].metric(
            "Appearance Brier",
            f"{result.validation_appearance_brier:.3f}" if result.validation_appearance_brier is not None else "N/A",
        )
        second_metrics[1].metric(
            "Minutes MAE",
            f"{result.validation_minutes_mae:.1f}" if result.validation_minutes_mae is not None else "N/A",
        )
        second_metrics[2].metric("Eligible players", len(eligible))
        st.caption(
            f"Model: {result.model_detail} · Fixtures: {fixture_provider} · "
            f"Refreshed {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )

        estimated_count = int((~exact_prices).sum())
        if estimated_count:
            st.warning(
                f"{estimated_count} prices are ML-estimated or synthetic, not official fantasy-game prices. "
                "Upload an official price file for exact budget compliance."
            )
        if name != "Premier League" and use_bundled:
            st.info(
                "The live features are competition-specific, but the supplied labels are Premier League data. "
                "This cross-league transfer model is useful as a baseline, not a league-specific accuracy guarantee."
            )

        if st.button("Generate best team", type="primary", key=f"optimize_{config.key}"):
            team = optimize_team(
                eligible,
                squad_size=config.squad_size,
                budget=budget,
                position_limits=config.position_limits,
                max_per_club=int(max_per_club),
                create_starting_xi=config.starting_xi,
                risk_aversion=float(risk_aversion),
            )
            st.session_state[f"team_{config.key}"] = team

        team = st.session_state.get(f"team_{config.key}")
        if team is not None:
            team_metrics = st.columns(3)
            team_metrics[0].metric("Squad cost", f"{team.total_cost:.1f} / {budget:.1f}")
            team_metrics[1].metric("Projected XI points", f"{team.expected_points:.2f}")
            team_metrics[2].metric("Captain", team.captain)
            st.markdown("#### Starting XI")
            render_team_pitch(team.starting_xi, team.captain, team.vice_captain)
            if not team.bench.empty:
                st.markdown("#### Bench")
                bench_columns = [
                    "name", "club", "position", "price", "predicted_points", "start_probability",
                    "appearance_probability", "expected_minutes_next", "next_opponent",
                ]
                st.dataframe(
                    team.bench[[column for column in bench_columns if column in team.bench.columns]],
                    use_container_width=True,
                    hide_index=True,
                )
            st.download_button(
                "Download optimized squad CSV",
                csv_download(team.squad),
                file_name=f"{config.key}_optimized_squad.csv",
                mime="text/csv",
                key=f"download_{config.key}",
            )

        with st.expander("ML player rankings", expanded=True):
            display_columns = [
                "name", "club", "position", "price", "predicted_points", "predicted_points_p10",
                "predicted_points_p90", "point_uncertainty", "value_score", "start_probability",
                "appearance_probability", "expected_minutes_next", "projection_confidence",
                "next_opponent", "fixture_difficulty", "team_matches_observed",
                "recent_start_rate", "recent_lineup_matches",
                "lineup_consensus_probability", "lineup_source_count", "lineup_status",
                "lineup_intelligence_note", "injury_reason", "price_source",
            ]
            st.dataframe(
                predicted.sort_values("predicted_points", ascending=False)[
                    [column for column in display_columns if column in predicted.columns]
                ].head(150),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Availability feed"):
            lineup_status = predicted.get("lineup_status", pd.Series("", index=predicted.index)).fillna("").astype(str)
            unavailable = predicted[
                (predicted["start_probability"] < 0.70)
                | (predicted["appearance_probability"] < 0.8)
                | predicted["injury_reason"].fillna("").astype(str).str.len().gt(0)
                | lineup_status.str.len().gt(0)
            ]
            if unavailable.empty:
                st.success("No availability warnings were returned by the selected source or ML model.")
            else:
                columns = [
                    "name", "club", "start_probability", "appearance_probability", "expected_minutes_next",
                    "lineup_consensus_probability", "lineup_source_count", "lineup_status",
                    "lineup_intelligence_note", "injury_reason", "next_opponent", "next_kickoff",
                ]
                st.dataframe(
                    unavailable[[column for column in columns if column in unavailable.columns]].sort_values(
                        "start_probability"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    except (DataProviderError, OptimizerError, ValueError, KeyError, FileNotFoundError) as exc:
        st.error(str(exc))


st.title("⚽ Fantasy XI Ensemble ML")
st.caption("Scikit-learn + XGBoost · Premier League · La Liga · Bundesliga · Serie A · Ligue 1 · Champions League")

with st.sidebar:
    st.header("Competition")
    selected_competition = st.selectbox(
        "Load one competition",
        list(COMPETITIONS.keys()),
        help="Only the selected competition is fetched, which protects free API quotas.",
    )

    st.header("Live data")
    api_key = st.text_input(
        "API-Football key",
        value=_configured_value("API_FOOTBALL_KEY", "API_SPORTS_KEY"),
        type="password",
        help="Live non-FPL players and injuries. Store as API_FOOTBALL_KEY in Streamlit secrets.",
    )
    football_data_token = st.text_input(
        "football-data.org token",
        value=_configured_value("FOOTBALL_DATA_TOKEN"),
        type="password",
        help="Optional second free fixture source. Store as FOOTBALL_DATA_TOKEN.",
    )
    season = st.number_input("Season start year", min_value=2020, max_value=2035, value=2026, step=1)
    use_api_injury_backup = st.checkbox(
        "Second EPL injury feed",
        value=False,
        help="Official FPL availability is already included; disabling this saves API calls.",
    )
    use_confirmed_lineups = st.checkbox(
        "Check confirmed lineups near kickoff",
        value=True,
        help=(
            "Uses API-Football only for fixtures within about 3 hours of kickoff. "
            "Confirmed starters/bench/out status overrides model uncertainty."
        ),
    )
    st.caption("Player statistics cache for 6 hours; fixtures 30–60 minutes; confirmed lineups 3 minutes.")
    if st.button("Refresh live data now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.header("Optional custom training")
    global_history_file = st.file_uploader("Historical CSV for this session", type=["csv"], key="global_history")
    global_history = read_uploaded_csv(global_history_file) if global_history_file is not None else None
    st.caption("The app includes 16,359 real next-fixture training rows with points, starts and minutes labels.")

    template = pd.DataFrame({"name": ["Example Player"], "price": [7.5]})
    st.download_button(
        "Download price CSV template",
        csv_download(template),
        "fantasy_price_template.csv",
        "text/csv",
    )

st.info(
    "Premier League live data works without a key. Other competitions use API-Football for player statistics; "
    "football-data.org can supply fixtures. The app is for free fantasy analysis, not betting."
)

render_competition(
    selected_competition,
    api_key,
    football_data_token,
    int(season),
    global_history,
    use_api_injury_backup,
    use_confirmed_lineups,
)
