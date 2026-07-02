# Active Model Reports

Generated at `2026-07-02T08:45:26.083449+07:00`.

This directory is overwritten by the local publish step.
Detailed raw training histories remain in `out/analysis`; this repo path stores lightweight latest forecasts and metrics.

- `vic/summary.json`: compact VIC forecast summary.
- `vic/vic_single_model_current.csv`: the only active VIC model output.
- `vic/vic_single_model_holdout.csv`: last-5-session holdout backtest.
- `vic/vic_single_model_walkback.csv`: pre-holdout walkback backtest.
- `vic/vic_single_model_candidates.csv`: feature/model candidate audit table.
- `vic/vic_single_model_feature_engineering.csv`: retained feature-set and engineered-feature audit table.
- `raw/`: copied latest selector artifacts before filtering.
