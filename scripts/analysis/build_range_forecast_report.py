from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _attach_actual_return_columns,
    _normalise_ticker,
    _reconstruct_ohlc_frame,
    build_multi_ticker_sample,
)
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache, first_timestamp_and_count, last_timestamp


REPO_ROOT = Path(__file__).resolve().parents[2]
VN_TZ = timezone(timedelta(hours=7))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_HISTORY_CALENDAR_DAYS = 800
DEFAULT_MAX_HORIZON = 10
DEFAULT_HOLDOUT_DATES = 30
DEFAULT_RECENT_FOCUS_DATES = 252
DEFAULT_MAX_REPORT_TICKERS = 60
DEFAULT_DYNAMIC_BUY_TICKERS = 12
DEFAULT_DYNAMIC_SELL_TICKERS = 12
KEY_HORIZONS = (1, 5, 10)
VARIANTS = ("full_2y", "recent_focus")
DEFAULT_REPORT_TICKERS = [
    "VNINDEX",
    "ACB",
    "BCM",
    "BID",
    "CTG",
    "FPT",
    "FRT",
    "GAS",
    "GMD",
    "GVR",
    "HDB",
    "HPG",
    "MBB",
    "MSN",
    "MWG",
    "NT2",
    "PC1",
    "PLX",
    "PNJ",
    "POW",
    "PVD",
    "PVT",
    "REE",
    "SAB",
    "SHB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VCI",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VND",
    "VNM",
    "VPB",
    "VRE",
]


def _load_universe_tickers(universe_csv: Path) -> List[str]:
    universe_df = pd.read_csv(universe_csv)
    if "Ticker" not in universe_df.columns:
        return []
    return _ordered_unique(["VNINDEX", *universe_df["Ticker"].astype(str).tolist()])


def build_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    def make_numeric_pipeline(model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "ridge": lambda: make_numeric_pipeline(Ridge(alpha=1.0)),
        "random_forest": lambda: make_numeric_pipeline(
            RandomForestRegressor(
                n_estimators=24,
                max_depth=8,
                max_features="sqrt",
                min_samples_leaf=4,
                n_jobs=4,
                random_state=42,
            )
        ),
        "hist_gbm": lambda: make_numeric_pipeline(
            HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=220,
                random_state=42,
            )
        ),
    }


def _fit_predict_target(
    model_factory: Callable[[], Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> np.ndarray:
    model = model_factory()
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df[target_column].astype(float))
    return model.predict(test_df[list(FEATURE_COLUMNS)])


def _variant_train_slice(
    train_df: pd.DataFrame,
    variant: str,
    recent_focus_dates: int,
    quarter_focus_dates: int = 63,
) -> pd.DataFrame:
    if variant == "full_2y":
        return train_df
    if variant == "recent_focus":
        unique_dates = list(pd.Index(sorted(train_df["Date"].unique())))
        if len(unique_dates) <= recent_focus_dates:
            return train_df
        cutoff = unique_dates[-recent_focus_dates]
        return train_df[train_df["Date"] >= cutoff].copy()
    if variant == "quarter_focus":
        unique_dates = list(pd.Index(sorted(train_df["Date"].unique())))
        if len(unique_dates) <= quarter_focus_dates:
            return train_df
        cutoff = unique_dates[-quarter_focus_dates]
        return train_df[train_df["Date"] >= cutoff].copy()
    raise ValueError(f"Unsupported variant: {variant}")


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for raw in values:
        ticker = _normalise_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered


def resolve_report_tickers(
    *,
    tickers: Sequence[str] | None,
    universe_csv: Path | None,
    max_report_tickers: int,
    dynamic_buy_tickers: int,
    dynamic_sell_tickers: int,
) -> List[str]:
    if tickers:
        resolved = _ordered_unique(tickers)
        if max_report_tickers > 0:
            resolved = resolved[:max_report_tickers]
        return resolved

    if universe_csv is not None and universe_csv.exists():
        universe_tickers = _load_universe_tickers(universe_csv)
        if universe_tickers:
            return universe_tickers

    runtime_tickers: List[str] = list(DEFAULT_REPORT_TICKERS)
    return _ordered_unique(runtime_tickers)


def _validate_prediction_file_coverage(
    prediction_df: pd.DataFrame,
    expected_tickers: Sequence[str],
    variant: str,
) -> None:
    expected_pairs = {(ticker, int(horizon)) for ticker in _ordered_unique(expected_tickers) for horizon in KEY_HORIZONS}
    actual_pairs = {
        (_normalise_ticker(ticker), int(horizon))
        for ticker, horizon in prediction_df[["Ticker", "Horizon"]].itertuples(index=False, name=None)
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if not missing_pairs:
        return

    preview = ", ".join(f"{ticker}:T+{horizon}" for ticker, horizon in missing_pairs[:12])
    raise RuntimeError(
        f"ML prediction coverage incomplete for variant={variant}; missing {len(missing_pairs)} ticker/horizon pairs. "
        f"Examples: {preview}"
    )


def refresh_history_cache(tickers: Sequence[str], history_dir: Path, history_calendar_days: int) -> pd.DataFrame:
    history_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for ticker in _ordered_unique(["VNINDEX", "VN30", *tickers]):
        normalized = _normalise_ticker(ticker)
        ensure_ohlc_cache(normalized, outdir=str(history_dir), min_days=history_calendar_days, resolution="D")
        cache_path = history_dir / f"{normalized}_daily.csv"
        first_ts, count = first_timestamp_and_count(cache_path)
        last_ts = last_timestamp(cache_path)
        first_date = datetime.fromtimestamp(first_ts, tz=VN_TZ).strftime("%Y-%m-%d") if first_ts else ""
        last_date = datetime.fromtimestamp(last_ts, tz=VN_TZ).strftime("%Y-%m-%d") if last_ts else ""
        rows.append(
            {
                "Ticker": normalized,
                "Bars": int(count),
                "FirstDate": first_date,
                "LastDate": last_date,
            }
        )
    return pd.DataFrame(rows)


def _recent_focus_weight(horizon: int) -> float:
    if horizon <= 1:
        return 0.75
    if horizon >= 10:
        return 0.40
    return float(0.75 - ((horizon - 1) * (0.35 / 9.0)))


def _build_variant_prediction_file(easy_view_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if easy_view_df.empty:
        return pd.DataFrame()

    scoped = easy_view_df[easy_view_df["Variant"] == variant].copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped["RecentFocusWeight"] = scoped["Horizon"].astype(int).map(_recent_focus_weight)
    scoped["Full2YWeight"] = 1.0 - scoped["RecentFocusWeight"]
    return scoped[
        [
            "SnapshotDate",
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "Base",
            "Low",
            "Mid",
            "High",
            "PredLowRetPct",
            "PredMidRetPct",
            "PredHighRetPct",
            "RecentFocusWeight",
            "Full2YWeight",
            "CloseMAEPct",
            "RangeMAEPct",
            "CloseDirHitPct",
        ]
    ].sort_values(["Ticker", "Horizon"]).reset_index(drop=True)


def _score_range_prediction_frame(eval_out: pd.DataFrame) -> Dict[str, float]:
    close_mae = float(mean_absolute_error(eval_out["ActualCloseRetPct"], eval_out["PredCloseRetPct"]))
    range_mae = float(mean_absolute_error(eval_out["ActualRangePct"], eval_out["PredRangePct"]))
    close_dir = float(
        (
            np.sign(eval_out["ActualCloseRetPct"].astype(float))
            == np.sign(eval_out["PredCloseRetPct"].astype(float))
        ).mean()
        * 100.0
    )
    selection_score = float(close_mae + (0.5 * range_mae) - (0.01 * close_dir))
    return {
        "CloseMAEPct": close_mae,
        "RangeMAEPct": range_mae,
        "CloseDirHitPct": close_dir,
        "SelectionScore": selection_score,
    }


def _select_best_range_model(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.sort_values(
            ["Ticker", "Variant", "Horizon", "SelectionScore", "CloseMAEPct", "RangeMAEPct", "CloseDirHitPct", "Model"],
            ascending=[True, True, True, True, True, True, False, True],
        )
        .groupby(["Ticker", "Variant", "Horizon"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def run_report(
    tickers: Sequence[str] | None,
    universe_csv: Path | None,
    history_dir: Path,
    output_dir: Path,
    history_calendar_days: int,
    max_horizon: int,
    holdout_dates: int,
    recent_focus_dates: int,
    max_report_tickers: int,
    dynamic_buy_tickers: int,
    dynamic_sell_tickers: int,
) -> Dict[str, object]:
    normalized_tickers = resolve_report_tickers(
        tickers=tickers,
        universe_csv=universe_csv,
        max_report_tickers=max_report_tickers,
        dynamic_buy_tickers=dynamic_buy_tickers,
        dynamic_sell_tickers=dynamic_sell_tickers,
    )
    history_summary = refresh_history_cache(normalized_tickers, history_dir, history_calendar_days)
    sample_df = build_multi_ticker_sample(normalized_tickers, history_dir, max_horizon=max_horizon)
    latest_date = pd.Timestamp(sample_df["Date"].max())
    model_factories = build_model_factories()
    fallback_model_name = next(iter(model_factories))

    all_metric_rows: List[Dict[str, object]] = []
    forecast_frames: List[pd.DataFrame] = []
    for variant in VARIANTS:
        for horizon in range(1, max_horizon + 1):
            scoped = sample_df[sample_df["Horizon"] == horizon].copy()
            for ticker_name, ticker_scoped in scoped.groupby("Ticker", sort=False):
                labeled = ticker_scoped[ticker_scoped["TargetOpenRetPct"].notna()].copy()
                unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
                selected_model_name = fallback_model_name
                if unique_dates and len(unique_dates) > holdout_dates:
                    eval_start = unique_dates[-holdout_dates]
                    train_eval_full = labeled[labeled["Date"] < eval_start].copy()
                    test_eval = labeled[labeled["Date"] >= eval_start].copy()
                    train_eval = _variant_train_slice(train_eval_full, variant, recent_focus_dates)
                    if not train_eval.empty and not test_eval.empty:
                        for model_name, factory in model_factories.items():
                            eval_out = test_eval[
                                [
                                    "Date",
                                    "Ticker",
                                    "Horizon",
                                    "ForecastWindow",
                                    "BaseClose",
                                    "ForecastDate",
                                    "ActualOpen",
                                    "ActualHigh",
                                    "ActualLow",
                                    "ActualClose",
                                ]
                                + TARGET_COLUMNS
                            ].copy()
                            eval_out["Variant"] = variant
                            eval_out["Model"] = model_name
                            for target_column in TARGET_COLUMNS:
                                eval_out[f"Pred{target_column}"] = _fit_predict_target(factory, train_eval, test_eval, target_column)
                            eval_out = pd.concat([eval_out, _reconstruct_ohlc_frame(eval_out)], axis=1)
                            eval_out = _attach_actual_return_columns(eval_out)
                            metric_row = {
                                "Ticker": ticker_name,
                                "Variant": variant,
                                "Model": model_name,
                                "Horizon": int(horizon),
                                "ForecastWindow": f"T+{horizon}",
                                "EvalRows": int(eval_out.shape[0]),
                            }
                            metric_row.update(_score_range_prediction_frame(eval_out))
                            all_metric_rows.append(metric_row)
                        ticker_metrics = [row for row in all_metric_rows if row["Ticker"] == ticker_name and row["Variant"] == variant and row["Horizon"] == int(horizon)]
                        if ticker_metrics:
                            selected_model_name = str(
                                _select_best_range_model(pd.DataFrame(ticker_metrics)).iloc[0]["Model"]
                            )

                train_full = _variant_train_slice(labeled[labeled["Date"] < latest_date].copy(), variant, recent_focus_dates)
                current_rows = ticker_scoped[ticker_scoped["Date"] == latest_date].copy()
                if train_full.empty or current_rows.empty:
                    continue
                forecast = current_rows[["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"]].copy()
                forecast["Variant"] = variant
                forecast["Model"] = selected_model_name
                for target_column in TARGET_COLUMNS:
                    forecast[f"Pred{target_column}"] = _fit_predict_target(
                        model_factories[selected_model_name],
                        train_full,
                        current_rows,
                        target_column,
                    )
                forecast = pd.concat([forecast, _reconstruct_ohlc_frame(forecast)], axis=1)
                forecast_frames.append(forecast)

    metrics_df = pd.DataFrame(all_metric_rows)
    selected_metrics_df = _select_best_range_model(metrics_df)
    forecasts_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()

    if not forecasts_df.empty:
        forecasts_df["SnapshotDate"] = latest_date.strftime("%Y-%m-%d")
        forecasts_df["Base"] = forecasts_df["BaseClose"].astype(float)
        forecasts_df["Low"] = forecasts_df["PredLow"].astype(float)
        forecasts_df["Mid"] = forecasts_df["PredClose"].astype(float)
        forecasts_df["High"] = forecasts_df["PredHigh"].astype(float)
        forecasts_df["PredLowRetPct"] = ((forecasts_df["Low"] / forecasts_df["Base"]) - 1.0) * 100.0
        forecasts_df["PredMidRetPct"] = ((forecasts_df["Mid"] / forecasts_df["Base"]) - 1.0) * 100.0
        forecasts_df["PredHighRetPct"] = ((forecasts_df["High"] / forecasts_df["Base"]) - 1.0) * 100.0
        forecasts_df = forecasts_df.merge(
            selected_metrics_df[
                [
                    "Ticker",
                    "Variant",
                    "Horizon",
                    "Model",
                    "EvalRows",
                    "SelectionScore",
                    "CloseMAEPct",
                    "RangeMAEPct",
                    "CloseDirHitPct",
                ]
            ],
            on=["Ticker", "Variant", "Horizon", "Model"],
            how="left",
        )

    easy_view_df = forecasts_df[forecasts_df["Horizon"].isin(KEY_HORIZONS)].copy()
    easy_view_df = easy_view_df[
        [
            "SnapshotDate",
            "Variant",
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "Base",
            "Low",
            "Mid",
            "High",
            "PredLowRetPct",
            "PredMidRetPct",
            "PredHighRetPct",
            "CloseMAEPct",
            "RangeMAEPct",
            "CloseDirHitPct",
        ]
    ].sort_values(["Variant", "Ticker", "Horizon"]).reset_index(drop=True)
    full_prediction_file_df = _build_variant_prediction_file(easy_view_df, "full_2y")
    recent_prediction_file_df = _build_variant_prediction_file(easy_view_df, "recent_focus")
    _validate_prediction_file_coverage(full_prediction_file_df, normalized_tickers, "full_2y")
    _validate_prediction_file_coverage(recent_prediction_file_df, normalized_tickers, "recent_focus")

    t10_df = forecasts_df[forecasts_df["Horizon"] == 10].copy()
    top_upside_df = (
        t10_df.sort_values(["Variant", "PredMidRetPct", "PredLowRetPct"], ascending=[True, False, False])
        .groupby("Variant", sort=False)
        .head(10)
        .copy()
    )
    top_upside_df["Bucket"] = "top_upside_t10"
    weakest_df = (
        t10_df.sort_values(["Variant", "PredMidRetPct", "PredLowRetPct"], ascending=[True, True, True])
        .groupby("Variant", sort=False)
        .head(10)
        .copy()
    )
    weakest_df["Bucket"] = "weakest_t10"
    ranking_df = pd.concat([top_upside_df, weakest_df], ignore_index=True)[
        [
            "SnapshotDate",
            "Variant",
            "Bucket",
            "Ticker",
            "Base",
            "Low",
            "Mid",
            "High",
            "PredLowRetPct",
            "PredMidRetPct",
            "PredHighRetPct",
        ]
    ].sort_values(["Variant", "Bucket", "PredMidRetPct"], ascending=[True, True, False]).reset_index(drop=True)

    comparison_df = easy_view_df[easy_view_df["Ticker"].isin(["FPT", "HPG"])].copy()
    horizon_metrics_df = (
        selected_metrics_df.groupby(["Variant", "Horizon", "ForecastWindow"], as_index=False)[["CloseMAEPct", "RangeMAEPct", "CloseDirHitPct"]]
        .mean()
        .sort_values(["Variant", "Horizon"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_summary.to_csv(output_dir / "ml_range_2y_history_summary.csv", index=False)
    forecasts_df.to_csv(output_dir / "ml_range_2y_all_horizons.csv", index=False)
    easy_view_df.to_csv(output_dir / "ml_range_2y_easy_view.csv", index=False)
    metrics_df.to_csv(output_dir / "ml_range_2y_model_metrics.csv", index=False)
    selected_metrics_df.to_csv(output_dir / "ml_range_2y_selected_models.csv", index=False)
    full_prediction_file_df.to_csv(output_dir / "ml_range_predictions_full_2y.csv", index=False)
    recent_prediction_file_df.to_csv(output_dir / "ml_range_predictions_recent_focus.csv", index=False)
    ranking_df.to_csv(output_dir / "ml_range_2y_top_bottom.csv", index=False)
    comparison_df.to_csv(output_dir / "ml_range_2y_fpt_hpg_comparison.csv", index=False)
    horizon_metrics_df.to_csv(output_dir / "ml_range_2y_horizon_metrics.csv", index=False)
    (output_dir / "ml_range_2y_summary.json").write_text(
        json.dumps(
            {
                "SnapshotDate": latest_date.strftime("%Y-%m-%d"),
                "HistoryCalendarDays": int(history_calendar_days),
                "HistorySummary": history_summary.to_dict(orient="records"),
                "Variants": list(VARIANTS),
                "UniverseSize": int(len(normalized_tickers)),
                "SelectedTickers": normalized_tickers,
                "ModelSelectionMode": "per_ticker",
                "CandidateModels": list(model_factories.keys()),
                "MaxHorizon": int(max_horizon),
                "KeyHorizons": list(KEY_HORIZONS),
                "RecentFocusWeightByKeyHorizon": {f"T+{horizon}": _recent_focus_weight(horizon) for horizon in KEY_HORIZONS},
                "Full2YWeightByKeyHorizon": {f"T+{horizon}": 1.0 - _recent_focus_weight(horizon) for horizon in KEY_HORIZONS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "ml_range_predictions_summary.json").write_text(
        json.dumps(
            {
                "SnapshotDate": latest_date.strftime("%Y-%m-%d"),
                "PredictionFiles": {
                    "full_2y": "ml_range_predictions_full_2y.csv",
                    "recent_focus": "ml_range_predictions_recent_focus.csv",
                },
                "ModelMetricsFile": "ml_range_2y_model_metrics.csv",
                "SelectedModelsFile": "ml_range_2y_selected_models.csv",
                "UniverseSize": int(len(normalized_tickers)),
                "SelectedTickers": normalized_tickers,
                "RecentFocusWeightByHorizon": {str(horizon): _recent_focus_weight(horizon) for horizon in range(1, max_horizon + 1)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "SnapshotDate": latest_date.strftime("%Y-%m-%d"),
        "HistorySummary": history_summary,
        "Forecasts": forecasts_df,
        "ModelMetrics": metrics_df,
        "SelectedModels": selected_metrics_df,
        "EasyView": easy_view_df,
        "FullPredictionFile": full_prediction_file_df,
        "RecentPredictionFile": recent_prediction_file_df,
        "TopBottom": ranking_df,
        "Comparison": comparison_df,
        "HorizonMetrics": horizon_metrics_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a ~2-year per-ticker ML range forecast report for VNINDEX and the selected runtime tickers."
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Dedicated cache directory for ~2y daily history.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for report artifacts.")
    parser.add_argument("--history-calendar-days", type=int, default=DEFAULT_HISTORY_CALENDAR_DAYS, help="Calendar-day lookback window used to seed the dedicated history cache.")
    parser.add_argument("--max-horizon", type=int, default=DEFAULT_MAX_HORIZON, help="Maximum forecast horizon T+N.")
    parser.add_argument("--holdout-dates", type=int, default=DEFAULT_HOLDOUT_DATES, help="Last N trading dates reserved for holdout metrics.")
    parser.add_argument("--recent-focus-dates", type=int, default=DEFAULT_RECENT_FOCUS_DATES, help="Training-date cap used by the recent-focus ML variant.")
    parser.add_argument("--universe-csv", type=Path, default=None, help="Optional engine universe snapshot. When provided, the runtime ML universe must cover every ticker in that live Codex universe.")
    parser.add_argument("--max-report-tickers", type=int, default=DEFAULT_MAX_REPORT_TICKERS, help="Cap for explicit/offline runtime ticker lists. Ignored when universe-backed live coverage is requested via --universe-csv.")
    parser.add_argument("--dynamic-buy-tickers", type=int, default=DEFAULT_DYNAMIC_BUY_TICKERS, help="How many objectively strong names from universe.csv to add to the ML runtime universe.")
    parser.add_argument("--dynamic-sell-tickers", type=int, default=DEFAULT_DYNAMIC_SELL_TICKERS, help="How many objectively weak names from universe.csv to add to the ML runtime universe.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Explicit universe override. If omitted, use the default core watchlist plus dynamic names from universe.csv when provided.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_report(
        tickers=args.tickers,
        universe_csv=args.universe_csv,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        history_calendar_days=int(args.history_calendar_days),
        max_horizon=int(args.max_horizon),
        holdout_dates=int(args.holdout_dates),
        recent_focus_dates=int(args.recent_focus_dates),
        max_report_tickers=int(args.max_report_tickers),
        dynamic_buy_tickers=int(args.dynamic_buy_tickers),
        dynamic_sell_tickers=int(args.dynamic_sell_tickers),
    )
    print(f"SnapshotDate: {result['SnapshotDate']}")
    print(f"UniverseSize: {int(result['HistorySummary'].shape[0])}")
    print(f"Wrote {(args.output_dir / 'ml_range_2y_easy_view.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_predictions_full_2y.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_predictions_recent_focus.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_2y_top_bottom.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_2y_fpt_hpg_comparison.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_2y_horizon_metrics.csv')}")
    print(f"Wrote {(args.output_dir / 'ml_range_2y_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
