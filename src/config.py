from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CompetitionConfig:
    key: str
    name: str
    api_league_id: int
    football_data_code: str
    default_budget: float
    squad_size: int
    position_limits: Dict[str, Tuple[int, int]]
    max_per_club: int
    starting_xi: bool = True


COMPETITIONS = {
    "Premier League": CompetitionConfig(
        key="epl",
        name="Premier League",
        api_league_id=39,
        football_data_code="PL",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
    "La Liga": CompetitionConfig(
        key="laliga",
        name="La Liga",
        api_league_id=140,
        football_data_code="PD",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
    "Bundesliga": CompetitionConfig(
        key="bundesliga",
        name="Bundesliga",
        api_league_id=78,
        football_data_code="BL1",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
    "Serie A": CompetitionConfig(
        key="seriea",
        name="Serie A",
        api_league_id=135,
        football_data_code="SA",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
    "Ligue 1": CompetitionConfig(
        key="ligue1",
        name="Ligue 1",
        api_league_id=61,
        football_data_code="FL1",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
    "Champions League": CompetitionConfig(
        key="ucl",
        name="Champions League",
        api_league_id=2,
        football_data_code="CL",
        default_budget=100.0,
        squad_size=15,
        position_limits={"GK": (2, 2), "DEF": (5, 5), "MID": (5, 5), "FWD": (3, 3)},
        max_per_club=3,
    ),
}

FPL_POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
API_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Attacker": "FWD",
}
