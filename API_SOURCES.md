# Free data-source design

## Active live sources

| Source | App use | Key | Quota strategy |
|---|---|---|---|
| Official FPL public API | Premier League players, official prices, form, availability and fixtures | None | 15-minute cache |
| API-Football | Non-FPL player statistics and injuries; fixture fallback | `API_FOOTBALL_KEY` | One selected competition; players cached 6 hours |
| football-data.org | Fixtures, results and team/opponent context | `FOOTBALL_DATA_TOKEN` | 60-minute cache; API-Football fallback |
| Bundled FPL history | Supervised points/start/appearance/minutes/price labels | None | Local CSV |

## Offline public-data layer

The training stack adds `soccerdata`, which can retrieve association-football data from sources such as FBref and Understat. The included fetcher currently uses FBref season statistics and writes a normalized enrichment CSV for the Streamlit app.

```bash
python scripts/fetch_public_soccer_data.py --season 2025-26
```

This runs offline/on Colab rather than on every Streamlit refresh. That keeps deployment fast and avoids hammering public websites.

`nflverse` and `nfl_data_py` are not used because they target American football, not association football.

## Why two fixture feeds?

football-data.org covers schedules/results without spending API-Football requests. API-Football remains the fixture fallback and supplies player/injury fields that football-data.org does not provide.

Confirmed-lineup polling is not continuously active because it can consume a large number of requests and is useful only shortly before kickoff. The model predicts starts, appearances and minutes from labelled history and current availability.

## Why non-FPL prices are estimated

Football-statistics APIs do not provide the official virtual price from every fantasy game. The app trains a price regressor on historical data and labels its output as non-official. Uploaded official prices always override estimates.

## Historical event data

StatsBomb Open Data can be useful for event-level research, but raw events are not automatically equivalent to fantasy-game labels. They need a stable identity map and a scoring-label pipeline before they should be added to supervised training.

## No odds data

Odds feeds are intentionally excluded. This repository is for free fantasy analysis rather than betting or paid-entry contests.

## v5 lineup evidence

- API-Football confirmed lineups are queried only for fixtures close to kickoff and cached for three minutes.
- Recent actual lineup history can be collected offline with `scripts/fetch_recent_lineup_history.py` using `soccerdata` + FBref.
- Predicted-lineup consensus is accepted as a user-supplied CSV so the model can combine several independent sources without hard-coding fragile site-specific HTML selectors into the deployed app.
- No odds, sportsbook, or paid-entry data is used.
