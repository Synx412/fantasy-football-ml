# v4 upgrade notes

## Implemented

- Added XGBoost 3.1.3 to the deployed ML stack.
- Added blended scikit-learn + XGBoost classifiers/regressors.
- Tuned ensemble weights on a time-ordered validation split.
- Reduced the starter/substitute role-model blend from 50% to 20% because the direct point model validated better overall.
- Rebuilt the bundled pre-trained model artifact (schema v3).
- Added portable CUDA training support through `FANTASY_XGB_DEVICE=cuda`; saved artifacts are switched back to CPU inference.
- Added a Google Colab training notebook.
- Added a separate training requirements file with Jupyter and `soccerdata`.
- Added an offline FBref/soccerdata public-stat fetcher.
- Added optional public-stat CSV enrichment in the Streamlit app.
- Added public-enrichment tests and XGBoost model-detail checks.
- Kept odds/betting data out of the project.

## Validation comparison on the bundled 2024/25 holdout

| Metric | Previous v3 | v4 ensemble |
|---|---:|---:|
| Points MAE | 1.391 | 1.379 |
| Start Brier | 0.093 | 0.092 |
| Appearance Brier | 0.091 | 0.090 |
| Minutes MAE | 14.58 | 14.30 |

Lower is better for all four metrics.

## Important limitation

The bundled supervised labels are still Premier League/FPL history. Public FBref enrichment improves feature coverage for other competitions, but truly league-specific supervised accuracy requires league-specific fantasy-point labels or a verified scoring-label builder.
