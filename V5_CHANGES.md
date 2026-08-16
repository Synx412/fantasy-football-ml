# v5 global availability + lineup intelligence upgrade

This build keeps the v4 scikit-learn + XGBoost point model and adds a stronger availability layer.

## Global fixes

- Replaced the fragile `starts / player appearances` starter prior with a smoothed `starts / team matches` prior wherever team-match context is available.
- The correction applies to the entire player pool. There are no Marmoush-, Gabriel Jesus-, club-, or league-specific hard-coded exceptions.
- Added a guard that prevents the ML classifier from turning a low real-world start prior into an unsupported ~95% start probability when there is enough team-match history and no external lineup evidence.
- New-season `0 starts / 0 team matches` is treated as unknown (50%), not as 0%.
- The 60% minimum-start filter still applies to the whole 15-player squad, including the bench.
- Default uncertainty penalty is now 0.30.

## Live lineup intelligence

- Added API-Football confirmed-lineup polling for fixtures close to kickoff.
- Confirmed starter -> 100% start/appearance availability for the next fixture.
- Confirmed bench -> 0% start, 100% squad appearance availability before minutes modelling.
- Player absent from a team whose full lineup is confirmed -> `not_in_squad` and 0 next-fixture availability.
- Confirmed data overrides model uncertainty only for the next fixture; later horizon fixtures stay model-predicted.

## Multi-source predicted-lineup consensus

The app accepts either:

```csv
source,name,club,status,source_weight,confirmed
Source A,Example Player,Example Club,starter,1.0,false
Source B,Example Player,Example Club,bench,1.0,false
```

or an already aggregated file:

```csv
name,club,start_probability,source_count,expected_minutes
Example Player,Example Club,0.67,6,71
```

One source has limited weight. More independent sources gain influence, capped so unconfirmed web predictions never fully replace the trained model.

## Recent actual-lineup scraper

`scripts/fetch_recent_lineup_history.py` uses `soccerdata` + FBref to collect recent actual lineups and creates per-player recent start rate and recent minutes. Upload that CSV in the app to give recent manager selections more weight.

Example:

```bash
python -m pip install -r requirements-training.txt
python scripts/fetch_recent_lineup_history.py --season 2026-27 --matches-per-team 6
```

## Still included from v4

- XGBoost + scikit-learn ensemble
- separate start / appearance / minutes / point models
- FPL live feed
- API-Football player + injury feed
- football-data.org fixture context
- FBref public-stat enrichment
- double/rescheduled matchday handling
- 15-player optimizer, XI, bench, captain and vice-captain
- no betting odds or paid-entry features
