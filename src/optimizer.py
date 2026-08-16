from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


class OptimizerError(RuntimeError):
    pass


@dataclass
class OptimizedTeam:
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    bench: pd.DataFrame
    captain: str
    vice_captain: str
    total_cost: float
    expected_points: float


def _solve_selection(
    players: pd.DataFrame,
    *,
    size: int,
    budget: float,
    position_limits: Dict[str, Tuple[int, int]],
    max_per_club: int,
) -> pd.DataFrame:
    pool = players.copy().reset_index(drop=True)
    score_column = "selection_score" if "selection_score" in pool.columns else "predicted_points"
    pool = pool[
        pool["position"].isin(position_limits)
        & pool["price"].notna()
        & (pool["price"] > 0)
        & pool["predicted_points"].notna()
        & pool[score_column].notna()
    ].reset_index(drop=True)
    if len(pool) < size:
        raise OptimizerError(f"Only {len(pool)} eligible players are available for a {size}-player team.")

    n = len(pool)
    rows = []
    lower = []
    upper = []

    # Exact squad size.
    rows.append(np.ones(n))
    lower.append(float(size))
    upper.append(float(size))

    # Budget cap.
    rows.append(pool["price"].astype(float).to_numpy())
    lower.append(-np.inf)
    upper.append(float(budget))

    # Position rules.
    for position, (minimum, maximum) in position_limits.items():
        mask = (pool["position"] == position).astype(float).to_numpy()
        rows.append(mask)
        lower.append(float(minimum))
        upper.append(float(maximum))

    # Maximum players from one club.
    if max_per_club < size:
        for club in pool["club"].dropna().unique():
            mask = (pool["club"] == club).astype(float).to_numpy()
            rows.append(mask)
            lower.append(-np.inf)
            upper.append(float(max_per_club))

    constraints = LinearConstraint(csr_matrix(np.vstack(rows)), np.array(lower), np.array(upper))
    objective = -pool[score_column].astype(float).to_numpy()
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": 20.0},
    )

    if not result.success or result.x is None:
        raise OptimizerError(
            "No valid team fits these constraints. Increase the budget, allow more players per club, "
            "or verify that every position has enough correctly priced players."
        )

    selected = np.flatnonzero(result.x > 0.5)
    if len(selected) != size:
        raise OptimizerError("The solver did not return a complete team. Check the player data and constraints.")
    return pool.iloc[selected].sort_values(score_column, ascending=False).reset_index(drop=True)


def optimize_team(
    players: pd.DataFrame,
    *,
    squad_size: int,
    budget: float,
    position_limits: Dict[str, Tuple[int, int]],
    max_per_club: int,
    create_starting_xi: bool,
    risk_aversion: float = 0.0,
) -> OptimizedTeam:
    prepared = players.copy()
    predicted = pd.to_numeric(prepared["predicted_points"], errors="coerce")
    uncertainty = pd.to_numeric(
        prepared.get("point_uncertainty", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    appearance = pd.to_numeric(
        prepared.get("appearance_probability", pd.Series(1.0, index=prepared.index)),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    start = pd.to_numeric(
        prepared.get("start_probability", appearance),
        errors="coerce",
    ).fillna(appearance).clip(0.0, 1.0)
    risk = float(np.clip(risk_aversion, 0.0, 1.0))
    prepared["selection_score"] = predicted - risk * (
        uncertainty + predicted.clip(lower=0.0) * (1.0 - start)
    )
    captain_risk = max(risk, 0.20)
    prepared["captain_score"] = predicted - captain_risk * uncertainty - (
        0.5 * predicted.clip(lower=0.0) * (1.0 - start)
    )

    squad = _solve_selection(
        prepared,
        size=squad_size,
        budget=budget,
        position_limits=position_limits,
        max_per_club=max_per_club,
    )

    if create_starting_xi and squad_size > 11:
        xi_limits = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
        starting_xi = _solve_selection(
            squad,
            size=11,
            budget=float(squad["price"].sum()) + 1.0,
            position_limits=xi_limits,
            max_per_club=11,
        )
        bench = squad[~squad["player_id"].isin(starting_xi["player_id"])].copy()
    else:
        starting_xi = squad.copy()
        bench = squad.iloc[0:0].copy()

    ranked = starting_xi.sort_values("captain_score", ascending=False)
    captain = str(ranked.iloc[0]["name"])
    vice = str(ranked.iloc[1]["name"]) if len(ranked) > 1 else captain
    expected = float(starting_xi["predicted_points"].sum() + ranked.iloc[0]["predicted_points"])

    return OptimizedTeam(
        squad=squad,
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        total_cost=float(squad["price"].sum()),
        expected_points=expected,
    )
