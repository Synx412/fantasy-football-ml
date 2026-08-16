from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COMPETITIONS
from src.demo import demo_players
from src.history import load_bundled_history
from src.model import estimate_missing_prices_ml, predict_players
from src.optimizer import optimize_team
from src.public_enrichment import merge_public_enrichment
from src.lineup_intelligence import (
    build_lineup_consensus,
    merge_confirmed_lineups,
    merge_lineup_consensus,
    merge_recent_lineup_history,
)
from src.providers import (
    apply_team_context,
    build_team_context,
    estimate_fpl_appearances,
    estimate_missing_prices,
    estimate_start_probability,
)


def fixture(
    fixture_id: int,
    date: str,
    round_name: str,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
    *,
    status: str = "NS",
    home_goals: int | None = None,
    away_goals: int | None = None,
) -> dict:
    return {
        "fixture": {"id": fixture_id, "date": date, "status": {"short": status}},
        "league": {"round": round_name},
        "teams": {
            "home": {"id": home_id, "name": home_name},
            "away": {"id": away_id, "name": away_name},
        },
        "goals": {"home": home_goals, "away": away_goals},
    }


def test_fixture_periods() -> None:
    fixtures = [
        fixture(1, "2026-08-01T15:00:00+00:00", "Matchday 1", 1, "Alpha", 2, "Beta", status="FT", home_goals=2, away_goals=0),
        fixture(2, "2026-08-20T15:00:00+00:00", "Matchday 10", 1, "Alpha", 2, "Beta"),
        fixture(3, "2026-08-23T15:00:00+00:00", "Matchday 3", 3, "Gamma", 1, "Alpha"),
        fixture(4, "2026-08-30T15:00:00+00:00", "Matchday 11", 1, "Alpha", 4, "Delta"),
    ]
    context = build_team_context(
        fixtures,
        horizon=2,
        now=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )
    alpha = context[context["club"] == "Alpha"].iloc[0]
    scenarios = alpha["fixture_scenarios"]
    assert len(scenarios) == 3
    assert [item["period_index"] for item in scenarios] == [0, 0, 1]
    assert alpha["next_fixture_ids"] == [2, 3, 4]

    cross_provider_players = pd.DataFrame(
        {"club": ["Bayern Munich", "Inter"], "fixture_difficulty": [3.0, 3.0]}
    )
    cross_provider_context = pd.DataFrame(
        {
            "club": ["FC Bayern München", "FC Internazionale Milano"],
            "fixture_difficulty": [1.5, 4.2],
        }
    )
    matched = apply_team_context(cross_provider_players, cross_provider_context)
    assert matched["fixture_difficulty"].tolist() == [1.5, 4.2]


def main() -> None:
    history = load_bundled_history()
    assert len(history) == 16359
    required_targets = {
        "future_points", "future_started", "future_appearance", "future_minutes",
    }
    assert required_targets.issubset(history.columns)
    test_fixture_periods()
    assert estimate_fpl_appearances(90, 1) == 1
    assert estimate_fpl_appearances(418, 3) == 12
    assert estimate_fpl_appearances(60, 0) == 4
    assert estimate_fpl_appearances(2293, 28) == 28
    assert 0.20 < estimate_start_probability(8, 30) < 0.30
    assert estimate_start_probability(8, 30) < 8 / 8

    # Global lineup intelligence must affect matching players rather than one hard-coded name.
    intel_players = pd.DataFrame([
        {"player_id": 1, "name": "Rotation One", "club": "Alpha", "start_probability": 0.25, "lineup_status": ""},
        {"player_id": 2, "name": "Starter Two", "club": "Alpha", "start_probability": 0.80, "lineup_status": ""},
        {"player_id": 3, "name": "Squad Three", "club": "Alpha", "start_probability": 0.45, "lineup_status": ""},
    ])
    votes = pd.DataFrame([
        {"source": "A", "name": "Rotation One", "club": "Alpha", "status": "bench"},
        {"source": "B", "name": "Rotation One", "club": "Alpha", "status": "starter"},
        {"source": "C", "name": "Rotation One", "club": "Alpha", "status": "bench"},
    ])
    consensus = build_lineup_consensus(votes)
    assert abs(float(consensus.iloc[0]["lineup_consensus_probability"]) - 1 / 3) < 1e-9
    intelligence_merged = merge_lineup_consensus(intel_players, votes)
    assert float(intelligence_merged.loc[0, "lineup_source_count"]) == 3.0
    recent = pd.DataFrame([
        {"name": "Rotation One", "club": "Alpha", "recent_start_rate": 0.2, "recent_lineup_matches": 5, "recent_minutes": 28},
        {"name": "Starter Two", "club": "Alpha", "recent_start_rate": 1.0, "recent_lineup_matches": 5, "recent_minutes": 84},
    ])
    recent_merged = merge_recent_lineup_history(intel_players, recent)
    assert float(recent_merged.loc[0, "start_probability"]) < 0.25
    assert float(recent_merged.loc[1, "start_probability"]) > 0.80
    confirmed = pd.DataFrame([
        {"fixture_id": 10, "player_id": 1, "name": "Rotation One", "club": "Alpha", "lineup_status": "bench"},
        {"fixture_id": 10, "player_id": 2, "name": "Starter Two", "club": "Alpha", "lineup_status": "starter"},
    ])
    confirmed_merged = merge_confirmed_lineups(intel_players, confirmed)
    assert confirmed_merged.loc[0, "lineup_status"] == "bench"
    assert confirmed_merged.loc[1, "lineup_status"] == "starter"
    assert confirmed_merged.loc[2, "lineup_status"] == "not_in_squad"

    players = demo_players("Premier League")
    players.loc[0, "xg"] = 0.0
    sample = players.iloc[[0]][["name", "club"]].copy()
    sample["xg"] = 9.9
    enriched = merge_public_enrichment(players, sample)
    assert int(enriched["public_data_match"].sum()) == 1
    assert float(enriched.loc[0, "xg"]) == 9.9

    players["price"] = float("nan")
    players["price_source"] = "Missing"
    players = estimate_missing_prices_ml(players, history, 100.0, 15)
    assert players["price_source"].str.contains("ML-estimated", na=False).all()
    players = estimate_missing_prices(players, 100.0, 15)

    result = predict_players(players, history, horizon=2)
    assert result.mode == "multitask_ml"
    assert result.validation_mae is not None
    assert result.validation_start_brier is not None
    assert result.validation_appearance_brier is not None
    assert result.validation_minutes_mae is not None
    assert "XGBoost" in result.model_detail
    output_columns = {
        "predicted_points", "predicted_points_p10", "predicted_points_p90",
        "point_uncertainty", "start_probability", "appearance_probability",
        "expected_minutes_next", "expected_minutes", "horizon_fixture_count",
        "projection_confidence",
    }
    assert output_columns.issubset(result.players.columns)
    assert result.players["start_probability"].between(0, 1).all()
    assert result.players["appearance_probability"].between(0, 1).all()
    assert (result.players["start_probability"] <= result.players["appearance_probability"]).all()
    assert result.players["expected_minutes_next"].between(0, 90).all()
    assert (result.players["predicted_points_p10"] <= result.players["predicted_points"]).all()
    assert (result.players["predicted_points"] <= result.players["predicted_points_p90"]).all()

    expected_names = {
        "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Champions League",
    }
    assert set(COMPETITIONS) == expected_names
    assert all(config.football_data_code for config in COMPETITIONS.values())

    # The competitions share the requested 15-player fantasy rules. Reuse one ML pool
    # so the smoke test trains the full ensemble only once.
    for name, config in COMPETITIONS.items():
        assert len(demo_players(name)) >= 100
        eligible = result.players[result.players["start_probability"] >= 0.35].copy()
        team = optimize_team(
            eligible,
            squad_size=config.squad_size,
            budget=config.default_budget,
            position_limits=config.position_limits,
            max_per_club=config.max_per_club,
            create_starting_xi=config.starting_xi,
            risk_aversion=0.25,
        )
        assert len(team.squad) == config.squad_size
        assert len(team.starting_xi) == 11
        assert team.total_cost <= config.default_budget + 1e-6
        assert team.captain in set(team.starting_xi["name"])
        assert pd.to_numeric(team.squad["selection_score"], errors="coerce").notna().all()
        for position, (minimum, maximum) in config.position_limits.items():
            count = int((team.squad["position"] == position).sum())
            assert minimum <= count <= maximum
        print(f"{name}: valid risk-aware squad, cost={team.total_cost:.1f}, captain={team.captain}")

    print(
        "Validation: "
        f"points MAE={result.validation_mae:.3f}, "
        f"start Brier={result.validation_start_brier:.3f}, "
        f"appearance Brier={result.validation_appearance_brier:.3f}, "
        f"minutes MAE={result.validation_minutes_mae:.2f}"
    )


if __name__ == "__main__":
    main()
