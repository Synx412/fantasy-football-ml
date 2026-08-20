from __future__ import annotations

import numpy as np
import pandas as pd

from src.preseason_prior import blend_preseason_availability_prior


def main() -> None:
    frame = pd.DataFrame(
        [
            {
                "name": "Gabriel",
                "full_name": "Gabriel dos Santos Magalhães",
                "position": "DEF",
                "official_fpl_xp": 5.15,
                "team_matches_observed": 38,
                "team_form_points": 1.5,
                "team_attack_form": 1.35,
                "team_defence_form": 1.35,
            }
        ]
    )
    start, app = blend_preseason_availability_prior(
        frame, np.array([0.58]), np.array([0.66])
    )
    assert 0.60 < float(start[0]) < 0.90, start
    assert float(app[0]) >= float(start[0]), (start, app)

    later = frame.copy()
    later["team_matches_observed"] = 8
    later["team_form_points"] = 1.8
    later["team_attack_form"] = 1.6
    later["team_defence_form"] = 1.1
    later_start, _ = blend_preseason_availability_prior(
        later, np.array([0.58]), np.array([0.66])
    )
    assert abs(float(later_start[0]) - 0.58) < abs(float(start[0]) - 0.58)

    non_fpl = frame.drop(columns=["official_fpl_xp"])
    untouched_start, untouched_app = blend_preseason_availability_prior(
        non_fpl, np.array([0.42]), np.array([0.55])
    )
    assert np.allclose(untouched_start, [0.42])
    assert np.allclose(untouched_app, [0.55])

    print(
        "v8.2 preseason availability prior: PASS | "
        f"Gabriel 58% -> {100*float(start[0]):.1f}%"
    )


if __name__ == "__main__":
    main()
