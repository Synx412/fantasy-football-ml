from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COMPETITIONS
from src.demo import demo_players
from src.history import V8_HISTORY, load_bundled_history
from src.model import (
    POINT_V2_PATH,
    PRETRAINED_MODEL_PATH,
    _four_source_point_ensemble,
    _load_point_v2_artifact,
    _point_v2_predict,
    _pretrained_artifact_for,
    engineer_features,
    estimate_missing_prices_ml,
    predict_players,
)
from src.optimizer import optimize_team
from src.public_enrichment import merge_public_enrichment
from src.lineup_intelligence import (
    build_lineup_consensus,
    merge_confirmed_lineups,
    merge_lineup_consensus,
    merge_recent_lineup_history,
)
from src.providers import (
    _fpl_fixture_context,
    apply_team_context,
    build_team_context,
    estimate_fpl_appearances,
    estimate_missing_prices,
    estimate_start_probability,
)
from src.soccerdata_live import SoccerDataBundle, build_soccerdata_player_pool


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


def test_model_artifacts(history: pd.DataFrame):
    assert V8_HISTORY.exists(), "Missing data/fpl_multitask_training_2025_26.csv"
    assert PRETRAINED_MODEL_PATH.exists(), "Missing models/fpl_multitask_bundle.joblib"
    assert POINT_V2_PATH.exists(), "Missing models/fpl_points_v2.joblib"

    pretrained = _pretrained_artifact_for(history)
    assert pretrained is not None, "Multitask artifact is incompatible with bundled v8 history"
    assert pretrained.get("model_version") == "v8"
    assert pretrained.get("training_season") == "2025-26"
    bundle = pretrained["bundle"]
    assert bundle.start_model is not None
    assert bundle.appearance_model is not None
    assert bundle.minutes_model is not None

    points = _load_point_v2_artifact()
    assert points is not None, "v8 point artifact failed to load"
    assert points.get("kind") == "fpl_points_v2"
    assert points.get("schema_version") == 1
    assert points.get("model_version") == "v8"
    assert points.get("training_season") == "2025-26"
    assert set(points.get("models", {})) == {"GK", "DEF", "MID", "FWD"}
    features = set(points.get("features", []))
    required_v8_features = {
        "price", "xgc_per90", "threat_per90", "creativity_per90",
        "influence_per90", "defensive_contribution_per90", "cbi_per90",
        "recoveries_per90", "tackles_per90",
    }
    assert required_v8_features.issubset(features)
    assert 0.0 <= float(points.get("official_base_blend", -1.0)) <= 0.65
    validation = points.get("validation", {})
    assert float(validation.get("model_only_mae", 99.0)) < 2.0
    assert np.isfinite(float(validation.get("model_only_mean_gw_spearman", np.nan)))
    return bundle


def test_feature_semantics() -> None:
    # Live player appearances must not be used as the denominator when the team
    # has played many more matches. Training/inference both represent TEAM-match exposure.
    row = pd.DataFrame([
        {
            "position": "FWD",
            "minutes": 400,
            "appearances": 10,
            "starts": 8,
            "team_matches_observed": 30,
        }
    ])
    out = engineer_features(row).iloc[0]
    assert abs(float(out["minutes_per_appearance"]) - 400 / 30) < 1e-9
    assert abs(float(out["start_probability"]) - 8 / 30) < 1e-9


def test_fixture_periods_and_preseason_neutrality() -> None:
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

    # No completed league games must mean neutral/unknown form, not "every team is terrible"
    # and every fixture is FDR 1.
    preseason = [
        fixture(10, "2026-08-21T15:00:00+00:00", "Matchday 1", 1, "Alpha", 2, "Beta"),
        fixture(11, "2026-08-22T15:00:00+00:00", "Matchday 1", 3, "Gamma", 4, "Delta"),
    ]
    neutral = build_team_context(
        preseason,
        horizon=1,
        now=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert np.allclose(neutral["team_strength"], 0.5)
    assert np.allclose(neutral["team_form_points"], 1.5)
    assert np.allclose(neutral["team_attack_form"], 1.35)
    assert np.allclose(neutral["team_defence_form"], 1.35)
    assert np.allclose(neutral["opponent_strength"], 0.5)
    assert neutral["fixture_difficulty"].between(2.5, 3.1).all()

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


def test_fpl_fixture_strength_and_rest() -> None:
    bootstrap = {
        "teams": [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}],
        "events": [{"id": 1, "is_next": True}],
    }
    fixtures = [
        {
            "id": 1, "event": 0, "finished": True,
            "team_h": 1, "team_a": 2,
            "team_h_score": 1, "team_a_score": 0,
            "kickoff_time": "2026-08-10T15:00:00Z",
            "team_h_difficulty": 3, "team_a_difficulty": 3,
        },
        {
            "id": 2, "event": 1, "finished": False,
            "team_h": 1, "team_a": 2,
            "kickoff_time": "2026-08-15T15:00:00Z",
            "team_h_difficulty": 4, "team_a_difficulty": 2,
        },
    ]
    strengths = {1: {"team_strength": 0.90}, 2: {"team_strength": 0.20}}
    context = _fpl_fixture_context(
        bootstrap, fixtures, horizon=1, team_strength_context=strengths
    )
    alpha = context[1]["fixture_scenarios"][0]
    beta = context[2]["fixture_scenarios"][0]
    assert abs(float(alpha["opponent_strength"]) - 0.20) < 1e-9
    assert abs(float(beta["opponent_strength"]) - 0.90) < 1e-9
    assert float(alpha["fixture_difficulty"]) == 4.0  # Official FPL FDR remains separate.
    assert abs(float(alpha["rest_days"]) - 5.0) < 1e-9


def test_soccerdata_pool_rotation_denominator() -> None:
    understat = pd.DataFrame([
        {"name": "Regular", "club": "Alpha", "position": "FWD", "appearances": 35, "starts": 34, "minutes": 2900},
        {"name": "Rotation", "club": "Alpha", "position": "FWD", "appearances": 25, "starts": 8, "minutes": 900},
    ])
    empty = pd.DataFrame()
    bundle = SoccerDataBundle(understat, empty, empty, empty, empty, {})
    pool = build_soccerdata_player_pool(bundle)
    regular = pool[pool["name"] == "Regular"].iloc[0]
    rotation = pool[pool["name"] == "Rotation"].iloc[0]
    assert float(regular["team_matches_observed"]) == 35.0
    assert float(rotation["team_matches_observed"]) == 35.0
    expected_rotation = estimate_start_probability(8, 35)
    assert abs(float(rotation["start_probability"]) - expected_rotation) < 1e-9
    assert float(rotation["start_probability"]) < 0.25


def _source_test_row(position: str = "FWD") -> pd.DataFrame:
    pool = demo_players("Premier League")
    row = pool[pool["position"] == position].iloc[[0]].copy().reset_index(drop=True)
    row["official_fpl_xp"] = 4.0
    row["fixture_count"] = 1.0
    row["team_matches_observed"] = 10.0
    row["team_strength"] = 0.70
    row["opponent_strength"] = 0.35
    row["fixture_difficulty"] = 2.5
    row["home"] = 1.0
    row["rest_days"] = 7.0
    row["base_start_probability"] = row["start_probability"]
    row["base_team_strength"] = row["team_strength"]
    row["base_opponent_strength"] = row["opponent_strength"]
    row["base_xg"] = row["xg"]
    row["base_xa"] = row["xa"]

    row["soccerdata_espn"] = True
    row["recent_lineup_matches"] = 5.0
    row["recent_start_rate"] = 0.80
    row["recent_minutes"] = 72.0
    row["espn_is_previous_season"] = False

    row["soccerdata_understat"] = True
    row["understat_matches"] = 12.0
    row["understat_minutes"] = 900.0
    row["understat_goals"] = 6.0
    row["understat_assists"] = 3.0
    row["understat_xg"] = 6.5
    row["understat_xa"] = 3.2
    row["understat_is_previous_season"] = False

    row["soccerdata_clubelo"] = True
    row["clubelo_strength"] = 0.72
    row["clubelo_opponent_strength"] = 0.30
    return row


def _engineered_source_rows(row: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    frames = []
    for period in periods:
        copy = row.copy()
        copy["_source_index"] = 0
        copy["_period_index"] = period
        copy["_scenario_weight"] = 1.0
        frames.append(copy)
    return engineer_features(pd.concat(frames, ignore_index=True)).reset_index(drop=True)


def test_v7_source_logic(bundle) -> None:
    engineered = _engineered_source_rows(_source_test_row("FWD"), [0])
    total, xp, _ = _four_source_point_ensemble(bundle, engineered)
    weights = xp["w_base"] + xp["w_espn"] + xp["w_understat"] + xp["w_clubelo"]
    assert np.allclose(weights, 1.0)
    assert np.isfinite(total).all()

    # Official FPL xP may change BASE only. ESPN/Understat/ClubElo must remain independent.
    low = engineered.copy()
    high = engineered.copy()
    low["official_fpl_xp"] = 2.0
    high["official_fpl_xp"] = 6.0
    _, xp_low, _ = _four_source_point_ensemble(bundle, low)
    _, xp_high, _ = _four_source_point_ensemble(bundle, high)
    assert not np.allclose(xp_low["base"], xp_high["base"])
    for source in ["espn", "understat", "clubelo"]:
        assert np.allclose(xp_low[source], xp_high[source], equal_nan=True), source

    # Understat attacking data is not a goalkeeper fantasy signal.
    gk = _engineered_source_rows(_source_test_row("GK"), [0])
    _, xp_gk, _ = _four_source_point_ensemble(bundle, gk)
    assert np.allclose(xp_gk["w_understat"], 0.0)
    assert np.isnan(xp_gk["understat"]).all()

    # Harder ClubElo opponent must not improve the same player's ClubElo xP.
    easy = engineered.copy()
    hard = engineered.copy()
    easy["clubelo_opponent_strength"] = 0.20
    hard["clubelo_opponent_strength"] = 0.90
    _, xp_easy, _ = _four_source_point_ensemble(bundle, easy)
    _, xp_hard, _ = _four_source_point_ensemble(bundle, hard)
    assert float(xp_hard["clubelo"][0]) <= float(xp_easy["clubelo"][0]) + 1e-9

    # ep_next is a GAMEWEEK total: a DGW must not duplicate the Official-FPL component.
    dgw0 = _engineered_source_rows(_source_test_row("FWD"), [0, 0])
    dgw4 = dgw0.copy()
    dgw0["official_fpl_xp"] = 0.0
    dgw4["official_fpl_xp"] = 4.0
    _, xp0, _ = _four_source_point_ensemble(bundle, dgw0)
    _, xp4, _ = _four_source_point_ensemble(bundle, dgw4)
    artifact = _load_point_v2_artifact()
    blend = float(artifact.get("official_base_blend", 0.50))
    observed_delta = float(np.sum(xp4["base"]) - np.sum(xp0["base"]))
    assert abs(observed_delta - blend * 4.0) < 1e-6

    # Official ep_next is next-GW only; future horizon rows remain model-only.
    horizon_low = _engineered_source_rows(_source_test_row("FWD"), [0, 1])
    horizon_high = horizon_low.copy()
    horizon_low["official_fpl_xp"] = 1.0
    horizon_high["official_fpl_xp"] = 7.0
    _, h_low, _ = _four_source_point_ensemble(bundle, horizon_low)
    _, h_high, _ = _four_source_point_ensemble(bundle, horizon_high)
    assert abs(float(h_low["base"][1]) - float(h_high["base"][1])) < 1e-9


def test_v8_scoring_features(history: pd.DataFrame) -> None:
    assert history["position"].astype(str).str.upper().isin({"GK", "DEF", "MID", "FWD"}).all()
    assert int(pd.to_numeric(history["GW"], errors="coerce").max()) >= 38
    assert "target_defensive_contribution" in history.columns
    dc_target = pd.to_numeric(history["target_defensive_contribution"], errors="coerce").fillna(0.0)
    assert float(dc_target.sum()) > 0.0, "2025/26 defensive-contribution labels are all zero"

    artifact = _load_point_v2_artifact()
    features = list(artifact["features"])
    defender = demo_players("Premier League")
    defender = defender[defender["position"].eq("DEF")].iloc[[0]].copy().reset_index(drop=True)
    defender["official_fpl_xp"] = 4.0
    defender["fixture_count"] = 1.0
    defender["team_matches_observed"] = 20.0
    defender["defensive_contribution"] = 0.0
    low = engineer_features(defender)
    high = defender.copy()
    high["defensive_contribution"] = 100.0
    high = engineer_features(high)
    low_xp = _point_v2_predict(low)
    high_xp = _point_v2_predict(high)
    assert low_xp is not None and high_xp is not None
    assert float(high_xp[0]) + 1e-9 >= float(low_xp[0]), (features, low_xp, high_xp)


def test_lineup_intelligence() -> None:
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


def main() -> None:
    history = load_bundled_history()
    assert V8_HISTORY.exists()
    assert len(history) > 15000
    required_targets = {"future_points", "future_started", "future_appearance", "future_minutes"}
    assert required_targets.issubset(history.columns)

    bundle = test_model_artifacts(history)
    test_feature_semantics()
    test_fixture_periods_and_preseason_neutrality()
    test_fpl_fixture_strength_and_rest()
    test_soccerdata_pool_rotation_denominator()
    test_v7_source_logic(bundle)
    test_v8_scoring_features(history)
    test_lineup_intelligence()

    assert estimate_fpl_appearances(90, 1) == 1
    assert estimate_fpl_appearances(418, 3) == 12
    assert estimate_fpl_appearances(60, 0) == 4
    assert estimate_fpl_appearances(2293, 28) == 28
    assert 0.20 < estimate_start_probability(8, 30) < 0.30

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
    assert "v8" in result.model_detail.lower()
    assert "2025-26" in result.model_detail
    assert result.validation_mae is not None
    assert result.validation_start_brier is not None
    assert result.validation_appearance_brier is not None
    assert result.validation_minutes_mae is not None
    output_columns = {
        "predicted_points", "predicted_points_p10", "predicted_points_p90",
        "point_uncertainty", "start_probability", "appearance_probability",
        "expected_minutes_next", "expected_minutes", "horizon_fixture_count",
        "projection_confidence", "xp_base_provider", "xp_espn", "xp_understat",
        "xp_clubelo", "xp_weight_base", "xp_weight_espn", "xp_weight_understat",
        "xp_weight_clubelo",
    }
    assert output_columns.issubset(result.players.columns)
    assert result.players["start_probability"].between(0, 1).all()
    assert result.players["appearance_probability"].between(0, 1).all()
    assert (result.players["start_probability"] <= result.players["appearance_probability"]).all()
    assert result.players["expected_minutes_next"].between(0, 90).all()
    assert (result.players["predicted_points_p10"] <= result.players["predicted_points"]).all()
    assert (result.players["predicted_points"] <= result.players["predicted_points_p90"]).all()

    expected_names = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Champions League"}
    assert set(COMPETITIONS) == expected_names
    assert all(config.football_data_code for config in COMPETITIONS.values())

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

    print("v8 source-logic invariants: PASS")
    print("pre-season context/denominator regressions: PASS")
    point_validation = _load_point_v2_artifact().get("validation", {})
    print(
        "v8 point model held-out: "
        f"model-only MAE={point_validation.get('model_only_mae', float('nan')):.3f}, "
        f"mean-GW Spearman={point_validation.get('model_only_mean_gw_spearman', float('nan')):.3f}, "
        f"FPL/Base blend MAE={point_validation.get('base_fpl_50_50_single_fixture_mae', float('nan')):.3f}"
    )
    print(
        "Availability bundle validation: "
        f"displayed point MAE={result.validation_mae:.3f}, "
        f"start Brier={result.validation_start_brier:.3f}, "
        f"appearance Brier={result.validation_appearance_brier:.3f}, "
        f"minutes MAE={result.validation_minutes_mae:.2f}"
    )


if __name__ == "__main__":
    main()
