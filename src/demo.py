from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np
import pandas as pd

CLUBS = {
    "Premier League": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Newcastle", "Spurs", "Aston Villa"],
    "La Liga": ["Barcelona", "Real Madrid", "Atletico", "Villarreal", "Betis", "Athletic Club", "Real Sociedad", "Sevilla"],
    "Bundesliga": ["Bayern", "Dortmund", "Leverkusen", "Leipzig", "Frankfurt", "Stuttgart", "Freiburg", "Wolfsburg"],
    "Serie A": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Atalanta", "Lazio", "Fiorentina"],
    "Ligue 1": ["PSG", "Marseille", "Monaco", "Lyon", "Lille", "Nice", "Rennes", "Lens"],
    "Champions League": ["Barcelona", "Real Madrid", "Bayern", "PSG", "Liverpool", "Arsenal", "Inter", "Man City"],
}
FIRST = ["Alex", "Leo", "Mateo", "Daniel", "Noah", "Hugo", "Lucas", "Rayan", "Marco", "Ethan", "Sami", "Nico"]
LAST = ["Silva", "Martin", "Garcia", "Costa", "Muller", "Rossi", "Santos", "Lopez", "Torres", "Kane", "Diaz", "Alvarez"]


def demo_players(competition: str, n: int = 180) -> pd.DataFrame:
    seed = int(hashlib.sha256(competition.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    positions = rng.choice(["GK", "DEF", "MID", "FWD"], size=n, p=[0.12, 0.34, 0.34, 0.20])
    rows = []
    for i in range(n):
        pos = positions[i]
        apps = int(rng.integers(1, 28))
        mins = float(apps * rng.integers(45, 91))
        starts = int(np.clip(round(apps * rng.uniform(0.45, 1.0)), 0, apps))
        attack_factor = {"GK": 0.02, "DEF": 0.15, "MID": 0.48, "FWD": 0.65}[pos]
        goals = int(rng.poisson(apps * attack_factor * 0.22))
        assists = int(rng.poisson(apps * attack_factor * 0.18))
        clean = int(rng.poisson(apps * ({"GK": 0.28, "DEF": 0.28, "MID": 0.08, "FWD": 0.02}[pos])))
        rating = float(np.clip(rng.normal(6.7 + 0.04 * goals, 0.45), 5.4, 8.8))
        price = float(np.clip(3.8 + 0.55 * (rating - 5.5) + 0.25 * goals + 0.15 * assists + rng.normal(0, .6), 3.5, 15.0))
        chance = float(rng.choice([1.0, 1.0, 1.0, 0.75, 0.25], p=[.72, .1, .08, .06, .04]))
        scenarios = [
            {
                "fixture_difficulty": float(rng.integers(1, 6)),
                "home": float(rng.integers(0, 2)),
                "opponent_strength": float(rng.uniform(0.15, 0.9)),
                "rest_days": float(rng.integers(4, 10)),
                "next_opponent": "Demo opponent",
                "next_kickoff": "",
                "fixture_count": 1.0,
                "period_index": period,
            }
            for period in range(6)
        ]
        rows.append({
            "player_id": i + 1,
            "name": f"{random.choice(FIRST)} {random.choice(LAST)} {i+1}",
            "club": random.choice(CLUBS[competition]),
            "position": pos,
            "price": round(price, 1),
            "minutes": mins,
            "appearances": apps,
            "starts": starts,
            "start_probability": starts / max(apps, 1),
            "rating": rating,
            "goals": goals,
            "assists": assists,
            "clean_sheets": clean,
            "saves": int(rng.poisson(apps * 2.4)) if pos == "GK" else 0,
            "yellow_cards": int(rng.poisson(apps * 0.08)),
            "red_cards": int(rng.binomial(1, 0.025)),
            "xg": max(0.0, goals + rng.normal(0, 1.2)),
            "xa": max(0.0, assists + rng.normal(0, 1.0)),
            "form": float(np.clip(rng.normal(4.0 + 0.35 * goals, 2.0), 0, 12)),
            "total_points": 0.0,
            "chance_playing": chance,
            "injury_reason": "Demo injury" if chance < .5 else "",
            "fixture_scenarios": scenarios,
            "lineup_status": "",
            "price_source": "Demo",
            "data_source": "Demo",
        })
    return pd.DataFrame(rows)


def demo_history(n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(20260730)
    positions = rng.choice(["GK", "DEF", "MID", "FWD"], size=n, p=[.12, .34, .34, .20])
    apps = rng.integers(1, 16, size=n)
    minutes = apps * rng.integers(45, 91, size=n)
    goals = rng.poisson(np.where(positions == "FWD", .35, np.where(positions == "MID", .22, .06)) * apps)
    assists = rng.poisson(np.where(positions == "MID", .25, np.where(positions == "FWD", .18, .08)) * apps)
    clean = rng.poisson(np.where(np.isin(positions, ["GK", "DEF"]), .3, .06) * apps)
    rating = np.clip(rng.normal(6.7, .5, size=n) + .05 * goals, 5.2, 9.0)
    chance = rng.choice([1.0, .75, .25], size=n, p=[.9, .07, .03])
    fixture = rng.integers(1, 6, size=n)
    home = rng.integers(0, 2, size=n)
    future = (
        1.3 + 1.2 * goals / np.maximum(apps, 1) + .9 * assists / np.maximum(apps, 1)
        + .35 * clean / np.maximum(apps, 1) + .55 * (rating - 6)
        + .25 * home - .18 * (fixture - 3)
    ) * chance + rng.normal(0, 1.1, size=n)
    return pd.DataFrame({
        "position": positions,
        "minutes": minutes,
        "appearances": apps,
        "goals": goals,
        "assists": assists,
        "clean_sheets": clean,
        "saves": np.where(positions == "GK", rng.poisson(2.4 * apps), 0),
        "rating": rating,
        "form": np.clip(rng.normal(4.5, 2.0, size=n), 0, 12),
        "xg": np.maximum(0, goals + rng.normal(0, 1, size=n)),
        "xa": np.maximum(0, assists + rng.normal(0, 1, size=n)),
        "yellow_cards": rng.poisson(.08 * apps),
        "red_cards": rng.binomial(1, .02, size=n),
        "chance_playing": chance,
        "fixture_difficulty": fixture,
        "home": home,
        "future_points": np.maximum(0, future),
    })
