from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _normalise_ticker,
    _attach_actual_return_columns,
    _reconstruct_ohlc_frame,
    build_multi_ticker_sample,
)
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
VN_TZ = timezone(timedelta(hours=7))
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_HISTORY_CALENDAR_DAYS = 800
DEFAULT_MIN_TRAIN_DATES = 120
DEFAULT_HOLDOUT_DATES = 30
DEFAULT_HORIZON = 1
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 20)
OUTPUT_FILE_NAME = "ml_ohlc_next_session.csv"
MULTI_OUTPUT_FILE_NAME = "ml_ohlc_multi_session.csv"
METRICS_OUTPUT_FILE_NAME = "ml_ohlc_model_metrics.csv"
STATE_SIGNAL_COLUMNS = [
    "TickerColorStreakState",
    "TickerLimitProxyState",
    "TickerShockState1D",
    "TickerImpulseState3D",
    "TickerWideRangeState",
    "TickerTrendRegimeState",
    "TickerCompressionState",
    "TickerReclaimState",
    "TickerRelativeRotationState",
    "TickerExhaustionState",
    "IndexColorStreakState",
    "VN30ColorStreakState",
]
REQUIRED_OUTPUT_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "Horizon",
    "ForecastWindow",
    "Base",
    "ForecastDate",
    "Model",
    "ModelFamily",
    "ModelClass",
    "EvalRows",
    "OpenMAEPct",
    "HighMAEPct",
    "LowMAEPct",
    "CumHighMAEPct",
    "CumLowMAEPct",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
    "SelectionScore",
    "ForecastOpen",
    "ForecastHigh",
    "ForecastLow",
    "ForecastCumHigh",
    "ForecastCumLow",
    "ForecastClose",
    "ForecastCloseRetPct",
    "ForecastCumHighRetPct",
    "ForecastCumLowRetPct",
    "ForecastRangePct",
    "ForecastCandleBias",
]

MODEL_METADATA = {
    "hist_gbm": {
        "ModelFamily": "ML-tree-boosting",
        "ModelClass": "HistGradientBoostingRegressor",
    },
}


def _model_metadata(model_name: object) -> Dict[str, str]:
    name = str(model_name)
    return dict(MODEL_METADATA.get(name, {"ModelFamily": "ML", "ModelClass": name}))


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _mean_upside_miss_mae(actual_high: pd.Series, predicted_high: pd.Series) -> float:
    return float((actual_high.astype(float) - predicted_high.astype(float)).clip(lower=0.0).mean())


def _mean_downside_miss_mae(actual_low: pd.Series, predicted_low: pd.Series) -> float:
    return float((predicted_low.astype(float) - actual_low.astype(float)).clip(lower=0.0).mean())


def _load_universe_tickers(universe_csv: Path) -> List[str]:
    universe_df = pd.read_csv(universe_csv, usecols=["Ticker"])
    tickers: List[str] = []
    for raw in universe_df["Ticker"].dropna().tolist():
        ticker = _normalise_ticker(raw)
        if not ticker or ticker == "VNINDEX" or ticker in tickers:
            continue
        tickers.append(ticker)
    if not tickers:
        raise RuntimeError(f"{universe_csv} did not provide any non-index ticker")
    return tickers


def refresh_history_cache(tickers: Sequence[str], history_dir: Path, history_calendar_days: int) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ["VNINDEX", "VN30", *tickers]:
        ensure_ohlc_cache(
            _normalise_ticker(ticker),
            outdir=str(history_dir),
            min_days=int(history_calendar_days),
            resolution="D",
        )


def summarise_ohlc_model_metrics_by_ticker(
    prediction_history: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()

    _require_columns(
        prediction_history,
        [
            "Ticker",
            "Model",
            "Horizon",
            "ActualOpenRetPct",
            "PredOpenRetPct",
            "ActualHighRetPct",
            "PredHighRetPct",
            "ActualLowRetPct",
            "PredLowRetPct",
            "ActualCloseRetPct",
            "PredCloseRetPct",
            "ActualRangePct",
            "PredRangePct",
        ],
        "OHLC prediction history",
    )
    scoped = prediction_history[prediction_history["Horizon"] == int(horizon)].copy()
    if scoped.empty:
        return pd.DataFrame()
    if "ActualCumHighRetPct" not in scoped.columns:
        scoped["ActualCumHighRetPct"] = scoped["ActualHighRetPct"].astype(float)
    if "PredCumHighRetPct" not in scoped.columns:
        scoped["PredCumHighRetPct"] = scoped["PredHighRetPct"].astype(float)
    if "ActualCumLowRetPct" not in scoped.columns:
        scoped["ActualCumLowRetPct"] = scoped["ActualLowRetPct"].astype(float)
    if "PredCumLowRetPct" not in scoped.columns:
        scoped["PredCumLowRetPct"] = scoped["PredLowRetPct"].astype(float)

    rows: List[Dict[str, object]] = []
    for (ticker, model_name), group in scoped.groupby(["Ticker", "Model"], sort=False):
        close_mae = float(mean_absolute_error(group["ActualCloseRetPct"], group["PredCloseRetPct"]))
        range_mae = float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"]))
        upside_miss_mae = _mean_upside_miss_mae(group["ActualHighRetPct"], group["PredHighRetPct"])
        downside_miss_mae = _mean_downside_miss_mae(group["ActualLowRetPct"], group["PredLowRetPct"])
        cum_high_mae = float(mean_absolute_error(group["ActualCumHighRetPct"], group["PredCumHighRetPct"]))
        cum_low_mae = float(mean_absolute_error(group["ActualCumLowRetPct"], group["PredCumLowRetPct"]))
        cum_upside_miss_mae = _mean_upside_miss_mae(group["ActualCumHighRetPct"], group["PredCumHighRetPct"])
        cum_downside_miss_mae = _mean_downside_miss_mae(group["ActualCumLowRetPct"], group["PredCumLowRetPct"])
        row = {
            "Ticker": ticker,
            "Model": model_name,
            **_model_metadata(model_name),
            "Horizon": int(horizon),
            "EvalRows": int(group.shape[0]),
            "OpenMAEPct": float(mean_absolute_error(group["ActualOpenRetPct"], group["PredOpenRetPct"])),
            "HighMAEPct": float(mean_absolute_error(group["ActualHighRetPct"], group["PredHighRetPct"])),
            "LowMAEPct": float(mean_absolute_error(group["ActualLowRetPct"], group["PredLowRetPct"])),
            "CumHighMAEPct": cum_high_mae,
            "CumLowMAEPct": cum_low_mae,
            "CloseMAEPct": close_mae,
            "RangeMAEPct": range_mae,
            "UpsideMissMAEPct": upside_miss_mae,
            "DownsideMissMAEPct": downside_miss_mae,
            "CumUpsideMissMAEPct": cum_upside_miss_mae,
            "CumDownsideMissMAEPct": cum_downside_miss_mae,
            "CloseDirHitPct": float(
                (
                    np.sign(group["ActualCloseRetPct"].astype(float))
                    == np.sign(group["PredCloseRetPct"].astype(float))
                ).mean()
                * 100.0
            ),
        }
        row["SelectionScore"] = float(
            row["CloseMAEPct"]
            + (0.35 * row["RangeMAEPct"])
            + (0.90 * row["UpsideMissMAEPct"])
            + (0.60 * row["DownsideMissMAEPct"])
            + (0.90 * row["CumUpsideMissMAEPct"])
            + (0.60 * row["CumDownsideMissMAEPct"])
            - (0.01 * row["CloseDirHitPct"])
        )
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    return metrics_df.sort_values(
        [
            "Ticker",
            "SelectionScore",
            "UpsideMissMAEPct",
            "DownsideMissMAEPct",
            "CloseMAEPct",
            "RangeMAEPct",
            "CloseDirHitPct",
        ],
        ascending=[True, True, True, True, True, True, False],
    ).reset_index(drop=True)


def _fit_predict_target(model_factory, train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> np.ndarray:
    model = model_factory()
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df[target_column].astype(float))
    return model.predict(test_df[list(FEATURE_COLUMNS)])


def build_live_model_factories():
    def make_numeric_pipeline(model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "hist_gbm": lambda: make_numeric_pipeline(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=120,
                random_state=42,
            )
        ),
    }


def generate_ohlc_next_session_predictions(
    sample_df: pd.DataFrame,
    *,
    min_train_dates: int,
    holdout_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_factories = build_live_model_factories()
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

    for (ticker, horizon), scoped_df in sample_df.groupby(["Ticker", "Horizon"], sort=False):
        labeled = scoped_df[scoped_df["TargetOpenRetPct"].notna()].sort_values("Date").copy()
        current_row = scoped_df[scoped_df["Date"] == scoped_df["Date"].max()].copy()
        if labeled.shape[0] < int(min_train_dates) + int(holdout_dates):
            continue

        train_df = labeled.iloc[:-int(holdout_dates)].copy()
        holdout_df = labeled.iloc[-int(holdout_dates):].copy()
        if train_df.empty or holdout_df.empty or current_row.empty:
            continue

        for model_name, factory in model_factories.items():
            out = holdout_df[
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
                    *TARGET_COLUMNS,
                ]
            ].copy()
            out["Model"] = model_name
            for target_column in TARGET_COLUMNS:
                out[f"Pred{target_column}"] = _fit_predict_target(factory, train_df, holdout_df, target_column)
            out = pd.concat([out, _reconstruct_ohlc_frame(out)], axis=1)
            out = _attach_actual_return_columns(out)
            history_frames.append(out)

            forecast_columns = ["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"]
            forecast_columns.extend(column for column in STATE_SIGNAL_COLUMNS if column in current_row.columns)
            forecast = current_row[forecast_columns].copy()
            forecast["Model"] = model_name
            for target_column in TARGET_COLUMNS:
                forecast[f"Pred{target_column}"] = _fit_predict_target(factory, labeled, current_row, target_column)
            forecast = pd.concat([forecast, _reconstruct_ohlc_frame(forecast)], axis=1)
            forecast["ForecastBodyPct"] = ((forecast["PredClose"] - forecast["PredOpen"]) / forecast["BaseClose"]) * 100.0
            forecast["ForecastCandleBias"] = np.where(
                forecast["PredClose"] > forecast["PredOpen"],
                "BULLISH",
                np.where(forecast["PredClose"] < forecast["PredOpen"], "BEARISH", "DOJI"),
            )
            current_frames.append(forecast)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    return history_df, current_df


def select_best_next_session_forecasts(
    prediction_history: pd.DataFrame,
    current_forecasts: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    metrics_df = summarise_ohlc_model_metrics_by_ticker(prediction_history, horizon=int(horizon))
    if metrics_df.empty or current_forecasts.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    current_forecasts = current_forecasts.copy()
    if "PredCumHigh" not in current_forecasts.columns and "PredHigh" in current_forecasts.columns:
        current_forecasts["PredCumHigh"] = current_forecasts["PredHigh"].astype(float)
    if "PredCumLow" not in current_forecasts.columns and "PredLow" in current_forecasts.columns:
        current_forecasts["PredCumLow"] = current_forecasts["PredLow"].astype(float)
    if "PredCumHighRetPct" not in current_forecasts.columns and "PredHighRetPct" in current_forecasts.columns:
        current_forecasts["PredCumHighRetPct"] = current_forecasts["PredHighRetPct"].astype(float)
    if "PredCumLowRetPct" not in current_forecasts.columns and "PredLowRetPct" in current_forecasts.columns:
        current_forecasts["PredCumLowRetPct"] = current_forecasts["PredLowRetPct"].astype(float)
    if "PredCumHighRetPct" not in current_forecasts.columns:
        current_forecasts["PredCumHighRetPct"] = (
            (current_forecasts["PredCumHigh"].astype(float) / current_forecasts["BaseClose"].astype(float)) - 1.0
        ) * 100.0
    if "PredCumLowRetPct" not in current_forecasts.columns:
        current_forecasts["PredCumLowRetPct"] = (
            (current_forecasts["PredCumLow"].astype(float) / current_forecasts["BaseClose"].astype(float)) - 1.0
        ) * 100.0

    _require_columns(
        current_forecasts,
        [
            "Date",
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "ForecastDate",
            "BaseClose",
            "Model",
            "PredOpen",
            "PredHigh",
            "PredLow",
            "PredCumHigh",
            "PredCumLow",
            "PredClose",
            "PredCloseRetPct",
            "PredCumHighRetPct",
            "PredCumLowRetPct",
            "PredRangePct",
            "ForecastCandleBias",
        ],
        "OHLC current forecasts",
    )
    current_scoped = current_forecasts[current_forecasts["Horizon"] == int(horizon)].copy()
    if current_scoped.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    for column in STATE_SIGNAL_COLUMNS:
        if column not in current_scoped.columns:
            current_scoped[column] = np.nan

    best_metrics = metrics_df.drop_duplicates(subset=["Ticker", "Horizon"], keep="first")
    merged = best_metrics.merge(
        current_scoped,
        on=["Ticker", "Horizon", "Model"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    snapshot_date = pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    forecast_date = pd.to_datetime(merged["ForecastDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    report_df = pd.DataFrame(
        {
            "SnapshotDate": snapshot_date,
            "Ticker": merged["Ticker"],
            "Horizon": merged["Horizon"].astype(int),
            "ForecastWindow": merged["ForecastWindow"],
            "Base": merged["BaseClose"].astype(float),
            "ForecastDate": forecast_date,
            "Model": merged["Model"],
            "ModelFamily": merged["ModelFamily"],
            "ModelClass": merged["ModelClass"],
            "EvalRows": merged["EvalRows"].astype(int),
            "OpenMAEPct": merged["OpenMAEPct"].astype(float),
            "HighMAEPct": merged["HighMAEPct"].astype(float),
            "LowMAEPct": merged["LowMAEPct"].astype(float),
            "CumHighMAEPct": merged["CumHighMAEPct"].astype(float),
            "CumLowMAEPct": merged["CumLowMAEPct"].astype(float),
            "CloseMAEPct": merged["CloseMAEPct"].astype(float),
            "RangeMAEPct": merged["RangeMAEPct"].astype(float),
            "CloseDirHitPct": merged["CloseDirHitPct"].astype(float),
            "SelectionScore": merged["SelectionScore"].astype(float),
            "ForecastOpen": merged["PredOpen"].astype(float),
            "ForecastHigh": merged["PredHigh"].astype(float),
            "ForecastLow": merged["PredLow"].astype(float),
            "ForecastCumHigh": merged["PredCumHigh"].astype(float),
            "ForecastCumLow": merged["PredCumLow"].astype(float),
            "ForecastClose": merged["PredClose"].astype(float),
            "ForecastCloseRetPct": merged["PredCloseRetPct"].astype(float),
            "ForecastCumHighRetPct": merged["PredCumHighRetPct"].astype(float),
            "ForecastCumLowRetPct": merged["PredCumLowRetPct"].astype(float),
            "ForecastRangePct": merged["PredRangePct"].astype(float),
            "ForecastCandleBias": merged["ForecastCandleBias"],
            "TickerColorStreakState": merged["TickerColorStreakState"].astype(float),
            "TickerLimitProxyState": merged["TickerLimitProxyState"].astype(float),
            "TickerShockState1D": merged["TickerShockState1D"].astype(float),
            "TickerImpulseState3D": merged["TickerImpulseState3D"].astype(float),
            "TickerWideRangeState": merged["TickerWideRangeState"].astype(float),
            "TickerTrendRegimeState": merged["TickerTrendRegimeState"].astype(float),
            "TickerCompressionState": merged["TickerCompressionState"].astype(float),
            "TickerReclaimState": merged["TickerReclaimState"].astype(float),
            "TickerRelativeRotationState": merged["TickerRelativeRotationState"].astype(float),
            "TickerExhaustionState": merged["TickerExhaustionState"].astype(float),
            "IndexColorStreakState": merged["IndexColorStreakState"].astype(float),
            "VN30ColorStreakState": merged["VN30ColorStreakState"].astype(float),
        }
    )
    _require_columns(report_df, REQUIRED_OUTPUT_COLUMNS, "OHLC next-session report")
    return report_df.sort_values(["Ticker"]).reset_index(drop=True)


def select_best_multi_session_forecasts(
    prediction_history: pd.DataFrame,
    current_forecasts: pd.DataFrame,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for horizon in horizons:
        frame = select_best_next_session_forecasts(
            prediction_history,
            current_forecasts,
            horizon=int(horizon),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Horizon"]).reset_index(drop=True)


def _validate_coverage(report_df: pd.DataFrame, tickers: Sequence[str]) -> None:
    actual = {
        _normalise_ticker(ticker)
        for ticker in report_df["Ticker"].dropna().tolist()
        if _normalise_ticker(ticker)
    }
    expected = {_normalise_ticker(ticker) for ticker in tickers if _normalise_ticker(ticker)}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(
            f"OHLC next-session report does not cover every synced ticker. Missing: {', '.join(missing)}"
        )


def _validate_horizon_coverage(report_df: pd.DataFrame, tickers: Sequence[str], horizons: Sequence[int]) -> None:
    for horizon in horizons:
        scoped = report_df.loc[pd.to_numeric(report_df["Horizon"], errors="coerce").eq(int(horizon))]
        _validate_coverage(scoped, tickers)


def _parse_horizons(values: Sequence[int] | None, fallback: int) -> List[int]:
    raw_values = list(values or [])
    if not raw_values:
        raw_values = list(DEFAULT_HORIZONS)
    if int(fallback) not in raw_values:
        raw_values.append(int(fallback))
    horizons: List[int] = []
    for value in raw_values:
        horizon = int(value)
        if horizon <= 0:
            raise ValueError("OHLC horizons must be positive integers")
        if horizon not in horizons:
            horizons.append(horizon)
    return sorted(horizons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a live T+1 OHLC forecast file by selecting the best per-ticker daily model."
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=REPO_ROOT / "out" / "universe.csv",
        help="Universe CSV used to resolve the working ticker set.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Directory containing per-ticker daily OHLC caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the live OHLC forecast artifact.",
    )
    parser.add_argument(
        "--history-calendar-days",
        type=int,
        default=DEFAULT_HISTORY_CALENDAR_DAYS,
        help="Calendar-day lookback to maintain in the daily cache.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=DEFAULT_MIN_TRAIN_DATES,
        help="Minimum labeled rows before walk-forward evaluation begins.",
    )
    parser.add_argument(
        "--holdout-dates",
        type=int,
        default=DEFAULT_HOLDOUT_DATES,
        help="Number of latest labeled rows used as holdout when ranking per-ticker models.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Daily forecast horizon to export (default: T+1).",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Daily forecast horizons to export to the multi-session artifact (default: T+1 T+2 T+3 T+5 T+10 T+15 T+20).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = _load_universe_tickers(args.universe_csv)
    horizons = _parse_horizons(args.horizons, int(args.horizon))
    refresh_history_cache(tickers, args.history_dir, int(args.history_calendar_days))

    sample_df = build_multi_ticker_sample(tickers, args.history_dir, max_horizon=max(horizons))
    prediction_history, current_forecasts = generate_ohlc_next_session_predictions(
        sample_df,
        min_train_dates=int(args.min_train_dates),
        holdout_dates=int(args.holdout_dates),
    )
    report_df = select_best_next_session_forecasts(
        prediction_history,
        current_forecasts,
        horizon=int(args.horizon),
    )
    if report_df.empty:
        raise RuntimeError("OHLC next-session report is empty; no forecast rows were produced.")
    _validate_coverage(report_df, tickers)
    multi_df = select_best_multi_session_forecasts(
        prediction_history,
        current_forecasts,
        horizons=horizons,
    )
    if multi_df.empty:
        raise RuntimeError("OHLC multi-session report is empty; no forecast rows were produced.")
    _validate_horizon_coverage(multi_df, tickers, horizons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_FILE_NAME
    multi_output_path = args.output_dir / MULTI_OUTPUT_FILE_NAME
    report_df.to_csv(output_path, index=False)
    multi_df.to_csv(multi_output_path, index=False)
    metrics_frames = [
        summarise_ohlc_model_metrics_by_ticker(prediction_history, horizon=int(horizon))
        for horizon in horizons
    ]
    metrics_frames = [frame for frame in metrics_frames if not frame.empty]
    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    metrics_output_path = args.output_dir / METRICS_OUTPUT_FILE_NAME
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"Wrote {output_path}")
    print(f"Wrote {multi_output_path}")
    print(f"Wrote {metrics_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
