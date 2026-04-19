from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.build_range_forecast_report import _variant_train_slice
from scripts.analysis.evaluate_ohlc_models import (
    FEATURE_COLUMNS,
    _load_daily_ohlcv,
    _normalise_ticker,
    build_ticker_ohlc_sample,
)
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_HISTORY_CALENDAR_DAYS = 900
DEFAULT_MIN_TRAIN_DATES = 120
DEFAULT_HOLDOUT_DATES = 40
DEFAULT_RECENT_FOCUS_DATES = 252
DEFAULT_QUARTER_FOCUS_DATES = 63
DEFAULT_HORIZONS = (3, 5, 10)
VARIANTS = ("full_2y", "recent_focus", "quarter_focus")
TARGET_COLUMNS = (
    "TargetPeakRetPct",
    "TargetPeakDay",
    "TargetDrawdownPct",
    "TargetCloseRetPct",
)
OUTPUT_FILE_NAME = "ml_single_name_timing.csv"
REQUIRED_OUTPUT_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "Horizon",
    "ForecastWindow",
    "Base",
    "ForecastDate",
    "Variant",
    "Model",
    "EvalRows",
    "PeakRetMAEPct",
    "PeakDayMAE",
    "DrawdownMAEPct",
    "CloseMAEPct",
    "TradeScoreMAEPct",
    "TradeScoreHitPct",
    "SelectionScore",
    "PredPeakRetPct",
    "PredPeakDay",
    "PredDrawdownPct",
    "PredCloseRetPct",
    "PredPeakPrice",
    "PredDrawdownPrice",
    "PredClosePrice",
    "PredRewardRisk",
    "PredTradeScore",
    "PredNetEdgePct",
    "PredCapitalEfficiencyPctPerDay",
]
TRADE_FEE_ROUND_TRIP_PCT = 0.06


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _load_universe_tickers(universe_csv: Path) -> List[str]:
    frame = pd.read_csv(universe_csv, usecols=["Ticker"])
    tickers: List[str] = []
    for raw in frame["Ticker"].dropna().tolist():
        ticker = _normalise_ticker(raw)
        if not ticker or ticker == "VNINDEX" or ticker in tickers:
            continue
        tickers.append(ticker)
    if not tickers:
        raise RuntimeError(f"{universe_csv} did not provide any non-index ticker")
    return tickers


def refresh_history_cache(tickers: Sequence[str], history_dir: Path, history_calendar_days: int) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ["VNINDEX", *tickers]:
        ensure_ohlc_cache(
            _normalise_ticker(ticker),
            outdir=str(history_dir),
            min_days=int(history_calendar_days),
            resolution="D",
        )


def _compute_trade_efficiency_targets(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon_days: int,
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")

    high_windows = pd.concat(
        [((high.shift(-day) / close) - 1.0) * 100.0 for day in range(1, horizon_days + 1)],
        axis=1,
    )
    high_windows.columns = list(range(1, horizon_days + 1))
    low_windows = pd.concat(
        [((low.shift(-day) / close) - 1.0) * 100.0 for day in range(1, horizon_days + 1)],
        axis=1,
    )
    low_windows.columns = list(range(1, horizon_days + 1))

    peak_ret = high_windows.max(axis=1)
    peak_day = high_windows.apply(
        lambda row: float(row.idxmax()) if row.notna().any() else np.nan,
        axis=1,
    )
    drawdown = low_windows.min(axis=1)
    close_ret = ((close.shift(-horizon_days) / close) - 1.0) * 100.0

    return pd.DataFrame(
        {
            "TargetPeakRetPct": peak_ret,
            "TargetPeakDay": peak_day,
            "TargetDrawdownPct": drawdown,
            "TargetCloseRetPct": close_ret,
        },
        index=close.index,
    )


def build_ticker_single_name_sample(
    ticker: str,
    history_dir: Path,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    normalized = _normalise_ticker(ticker)
    base_frame = build_ticker_ohlc_sample(normalized, history_dir, max_horizon=1)
    base_frame = (
        base_frame[["Date", "Ticker", "BaseClose"] + list(FEATURE_COLUMNS)]
        .drop_duplicates(subset=["Date"])
        .copy()
    )
    base_frame["Date"] = pd.to_datetime(base_frame["Date"], errors="coerce")
    base_frame = base_frame.dropna(subset=["Date"]).set_index("Date", drop=False).sort_index()

    price_frame = _load_daily_ohlcv(normalized, history_dir)
    forecast_dates = pd.Series(price_frame.index, index=price_frame.index)

    frames: List[pd.DataFrame] = []
    for horizon in horizons:
        targets = _compute_trade_efficiency_targets(
            close=price_frame["Close"],
            high=price_frame["High"],
            low=price_frame["Low"],
            horizon_days=int(horizon),
        )
        frame = base_frame.copy()
        frame["Horizon"] = int(horizon)
        frame["ForecastWindow"] = f"T+{int(horizon)}"
        frame["ForecastDate"] = forecast_dates.shift(-int(horizon)).reindex(frame.index)
        for column in TARGET_COLUMNS:
            frame[column] = targets[column].reindex(frame.index)
        frames.append(frame.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker", "Horizon"]).reset_index(drop=True)


def build_multi_ticker_single_name_sample(
    tickers: Sequence[str],
    history_dir: Path,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    frames = [
        build_ticker_single_name_sample(ticker, history_dir, horizons=horizons)
        for ticker in tickers
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        "hist_gbm": lambda: make_numeric_pipeline(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=180,
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


def _clip_peak_day(values: pd.Series, horizon_days: int) -> pd.Series:
    return values.astype(float).clip(lower=1.0, upper=float(horizon_days))


def _trade_score(peak_ret_pct: pd.Series | float, drawdown_pct: pd.Series | float) -> pd.Series | float:
    return peak_ret_pct - np.abs(drawdown_pct)


def _net_edge_pct(peak_ret_pct: pd.Series | float, drawdown_pct: pd.Series | float) -> pd.Series | float:
    return _trade_score(peak_ret_pct, drawdown_pct) - TRADE_FEE_ROUND_TRIP_PCT


def _reward_risk_ratio(peak_ret_pct: pd.Series, drawdown_pct: pd.Series) -> pd.Series:
    denominator = np.abs(drawdown_pct.astype(float))
    denominator = denominator.where(denominator > 1e-9, np.nan)
    return peak_ret_pct.astype(float) / denominator


def _score_eval_frame(eval_out: pd.DataFrame) -> pd.DataFrame:
    actual_trade_score = _trade_score(eval_out["TargetPeakRetPct"], eval_out["TargetDrawdownPct"]).astype(float)
    pred_trade_score = _trade_score(eval_out["PredTargetPeakRetPct"], eval_out["PredTargetDrawdownPct"]).astype(float)
    actual_net_edge = _net_edge_pct(eval_out["TargetPeakRetPct"], eval_out["TargetDrawdownPct"]).astype(float)
    pred_net_edge = _net_edge_pct(eval_out["PredTargetPeakRetPct"], eval_out["PredTargetDrawdownPct"]).astype(float)
    return pd.DataFrame(
        [
            {
                "EvalRows": int(eval_out.shape[0]),
                "PeakRetMAEPct": float(mean_absolute_error(eval_out["TargetPeakRetPct"], eval_out["PredTargetPeakRetPct"])),
                "PeakDayMAE": float(mean_absolute_error(eval_out["TargetPeakDay"], eval_out["PredTargetPeakDay"])),
                "DrawdownMAEPct": float(mean_absolute_error(eval_out["TargetDrawdownPct"], eval_out["PredTargetDrawdownPct"])),
                "CloseMAEPct": float(mean_absolute_error(eval_out["TargetCloseRetPct"], eval_out["PredTargetCloseRetPct"])),
                "TradeScoreMAEPct": float(mean_absolute_error(actual_trade_score, pred_trade_score)),
                "TradeScoreHitPct": float((np.sign(actual_net_edge) == np.sign(pred_net_edge)).mean() * 100.0),
            }
        ]
    )


def select_best_single_name_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.sort_values(
            [
                "Ticker",
                "Horizon",
                "SelectionScore",
                "TradeScoreMAEPct",
                "PeakRetMAEPct",
                "DrawdownMAEPct",
                "CloseMAEPct",
                "TradeScoreHitPct",
                "Variant",
                "Model",
            ],
            ascending=[True, True, True, True, True, True, True, False, True, True],
        )
        .groupby(["Ticker", "Horizon"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def _validate_output_coverage(
    report_df: pd.DataFrame,
    expected_tickers: Sequence[str],
    horizons: Sequence[int],
) -> None:
    expected_pairs = {(ticker, int(horizon)) for ticker in expected_tickers for horizon in horizons}
    actual_pairs = {
        (_normalise_ticker(ticker), int(horizon))
        for ticker, horizon in report_df[["Ticker", "Horizon"]].itertuples(index=False, name=None)
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if not missing_pairs:
        return
    preview = ", ".join(f"{ticker}:T+{horizon}" for ticker, horizon in missing_pairs[:12])
    raise RuntimeError(
        f"Single-name timing coverage incomplete; missing {len(missing_pairs)} ticker/horizon pairs. "
        f"Examples: {preview}"
    )


def run_report(
    *,
    universe_csv: Path,
    history_dir: Path,
    output_dir: Path,
    history_calendar_days: int,
    min_train_dates: int,
    holdout_dates: int,
    recent_focus_dates: int,
    quarter_focus_dates: int,
    horizons: Sequence[int],
) -> Dict[str, object]:
    tickers = _load_universe_tickers(universe_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    refresh_history_cache(tickers, history_dir, history_calendar_days)
    sample_df = build_multi_ticker_single_name_sample(tickers, history_dir, horizons=horizons)
    if sample_df.empty:
        raise RuntimeError("Single-name timing sample is empty; cannot build report")

    latest_date = pd.Timestamp(sample_df["Date"].max())
    model_factories = build_model_factories()
    metric_frames: List[pd.DataFrame] = []

    for variant in VARIANTS:
        for horizon in horizons:
            print(f"[single-name-timing] eval variant={variant} horizon=T+{int(horizon)}", flush=True)
            scoped = sample_df[sample_df["Horizon"].astype(int) == int(horizon)].copy()
            for ticker_name, ticker_scoped in scoped.groupby("Ticker", sort=False):
                labeled = ticker_scoped.dropna(subset=list(TARGET_COLUMNS)).sort_values("Date").copy()
                unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
                if len(unique_dates) < int(min_train_dates) + int(holdout_dates):
                    continue
                eval_start = unique_dates[-int(holdout_dates)]
                train_eval_full = labeled[labeled["Date"] < eval_start].copy()
                test_eval = labeled[labeled["Date"] >= eval_start].copy()
                train_eval = _variant_train_slice(
                    train_eval_full,
                    variant,
                    recent_focus_dates,
                    quarter_focus_dates=quarter_focus_dates,
                )
                if train_eval.empty or test_eval.empty:
                    continue
                for model_name, factory in model_factories.items():
                    eval_out = test_eval[
                        [
                            "Date",
                            "Ticker",
                            "Horizon",
                            "ForecastWindow",
                            "BaseClose",
                            "ForecastDate",
                        ]
                        + list(TARGET_COLUMNS)
                    ].copy()
                    for target_column in TARGET_COLUMNS:
                        eval_out[f"Pred{target_column}"] = _fit_predict_target(factory, train_eval, test_eval, target_column)
                    eval_out["PredTargetPeakDay"] = _clip_peak_day(eval_out["PredTargetPeakDay"], int(horizon))
                    metric_row = _score_eval_frame(eval_out)
                    metric_row["Ticker"] = ticker_name
                    metric_row["Variant"] = variant
                    metric_row["Model"] = model_name
                    metric_row["Horizon"] = int(horizon)
                    metric_row["ForecastWindow"] = f"T+{int(horizon)}"
                    metric_frames.append(metric_row)

    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    if metrics_df.empty:
        raise RuntimeError("Single-name timing metrics are empty; insufficient holdout coverage")

    metrics_df["SelectionScore"] = (
        metrics_df["PeakRetMAEPct"].astype(float)
        + (0.15 * metrics_df["PeakDayMAE"].astype(float))
        + (0.50 * metrics_df["DrawdownMAEPct"].astype(float))
        + (0.50 * metrics_df["CloseMAEPct"].astype(float))
        + (0.75 * metrics_df["TradeScoreMAEPct"].astype(float))
        - (0.01 * metrics_df["TradeScoreHitPct"].astype(float))
    )
    selected_df = select_best_single_name_configs(metrics_df)

    current_frames: List[pd.DataFrame] = []
    for config in selected_df.to_dict(orient="records"):
        ticker_name = str(config["Ticker"])
        variant = str(config["Variant"])
        model_name = str(config["Model"])
        horizon = int(config["Horizon"])
        print(
            f"[single-name-timing] current ticker={ticker_name} variant={variant} horizon=T+{horizon} model={model_name}",
            flush=True,
        )
        scoped = sample_df[
            (sample_df["Ticker"].astype(str) == ticker_name)
            & (sample_df["Horizon"].astype(int) == horizon)
        ].copy()
        labeled = scoped.dropna(subset=list(TARGET_COLUMNS)).sort_values("Date").copy()
        train_full = _variant_train_slice(
            labeled[labeled["Date"] < latest_date].copy(),
            variant,
            recent_focus_dates,
            quarter_focus_dates=quarter_focus_dates,
        )
        current_rows = scoped[scoped["Date"] == latest_date].copy()
        if train_full.empty or current_rows.empty:
            continue
        factory = model_factories[model_name]
        forecast = current_rows[
            ["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"]
        ].copy()
        forecast["Variant"] = variant
        forecast["Model"] = model_name
        for target_column in TARGET_COLUMNS:
            forecast[f"Pred{target_column}"] = _fit_predict_target(factory, train_full, current_rows, target_column)
        forecast["PredTargetPeakDay"] = _clip_peak_day(forecast["PredTargetPeakDay"], horizon)
        current_frames.append(forecast)

    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    if current_df.empty:
        raise RuntimeError("Single-name timing current forecast is empty; unable to score latest rows")

    merged = selected_df.merge(
        current_df,
        on=["Ticker", "Horizon", "ForecastWindow", "Variant", "Model"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("Single-name timing merge between metrics and current forecast is empty")

    snapshot_date = pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    forecast_date = pd.to_datetime(merged["ForecastDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    base = merged["BaseClose"].astype(float)
    pred_peak_ret = merged["PredTargetPeakRetPct"].astype(float)
    pred_peak_day = merged["PredTargetPeakDay"].astype(float).clip(lower=1.0, upper=merged["Horizon"].astype(float)).round(1)
    pred_drawdown = merged["PredTargetDrawdownPct"].astype(float)
    pred_close_ret = merged["PredTargetCloseRetPct"].astype(float)
    pred_trade_score = _trade_score(pred_peak_ret, pred_drawdown).astype(float)
    pred_net_edge = _net_edge_pct(pred_peak_ret, pred_drawdown).astype(float)
    pred_capital_efficiency = pred_net_edge / pred_peak_day.clip(lower=1.0)

    report_df = pd.DataFrame(
        {
            "SnapshotDate": snapshot_date,
            "Ticker": merged["Ticker"],
            "Horizon": merged["Horizon"].astype(int),
            "ForecastWindow": merged["ForecastWindow"],
            "Base": base,
            "ForecastDate": forecast_date,
            "Variant": merged["Variant"],
            "Model": merged["Model"],
            "EvalRows": merged["EvalRows"].astype(int),
            "PeakRetMAEPct": merged["PeakRetMAEPct"].astype(float),
            "PeakDayMAE": merged["PeakDayMAE"].astype(float),
            "DrawdownMAEPct": merged["DrawdownMAEPct"].astype(float),
            "CloseMAEPct": merged["CloseMAEPct"].astype(float),
            "TradeScoreMAEPct": merged["TradeScoreMAEPct"].astype(float),
            "TradeScoreHitPct": merged["TradeScoreHitPct"].astype(float),
            "SelectionScore": merged["SelectionScore"].astype(float),
            "PredPeakRetPct": pred_peak_ret,
            "PredPeakDay": pred_peak_day,
            "PredDrawdownPct": pred_drawdown,
            "PredCloseRetPct": pred_close_ret,
            "PredPeakPrice": base * (1.0 + (pred_peak_ret / 100.0)),
            "PredDrawdownPrice": base * (1.0 + (pred_drawdown / 100.0)),
            "PredClosePrice": base * (1.0 + (pred_close_ret / 100.0)),
            "PredRewardRisk": _reward_risk_ratio(pred_peak_ret, pred_drawdown),
            "PredTradeScore": pred_trade_score,
            "PredNetEdgePct": pred_net_edge,
            "PredCapitalEfficiencyPctPerDay": pred_capital_efficiency,
        }
    )
    _require_columns(report_df, REQUIRED_OUTPUT_COLUMNS, "Single-name timing report")
    report_df = report_df.sort_values(["Ticker", "Horizon"]).reset_index(drop=True)
    _validate_output_coverage(report_df, tickers, horizons)

    metrics_path = output_dir / "ml_single_name_timing_model_metrics.csv"
    selected_path = output_dir / "ml_single_name_timing_selected_models.csv"
    output_path = output_dir / OUTPUT_FILE_NAME
    metrics_df.to_csv(metrics_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    report_df.to_csv(output_path, index=False)

    return {
        "output_path": output_path,
        "metrics_path": metrics_path,
        "selected_path": selected_path,
        "row_count": int(report_df.shape[0]),
        "ticker_count": int(report_df["Ticker"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-ticker short-horizon single-name timing report for concentrated trade decisions.",
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        required=True,
        help="Path to out/universe.csv used to resolve live tickers.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Directory containing cached daily OHLCV data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where live single-name timing artifacts will be written.",
    )
    parser.add_argument(
        "--history-calendar-days",
        type=int,
        default=DEFAULT_HISTORY_CALENDAR_DAYS,
        help="Minimum number of calendar days of daily history to cache per ticker.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=DEFAULT_MIN_TRAIN_DATES,
        help="Minimum number of dated observations required before reserving holdout rows.",
    )
    parser.add_argument(
        "--holdout-dates",
        type=int,
        default=DEFAULT_HOLDOUT_DATES,
        help="Number of dated observations reserved for holdout evaluation per ticker/horizon.",
    )
    parser.add_argument(
        "--recent-focus-dates",
        type=int,
        default=DEFAULT_RECENT_FOCUS_DATES,
        help="Maximum number of dated observations kept in the recent_focus training slice.",
    )
    parser.add_argument(
        "--quarter-focus-dates",
        type=int,
        default=DEFAULT_QUARTER_FOCUS_DATES,
        help="Maximum number of dated observations kept in the quarter_focus training slice.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Short horizons (in trading sessions) used to score single-name timing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_report(
        universe_csv=args.universe_csv,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        history_calendar_days=args.history_calendar_days,
        min_train_dates=args.min_train_dates,
        holdout_dates=args.holdout_dates,
        recent_focus_dates=args.recent_focus_dates,
        quarter_focus_dates=args.quarter_focus_dates,
        horizons=tuple(int(value) for value in args.horizons),
    )
    print(f"Wrote {result['output_path']}")
    print(f"Wrote {result['metrics_path']}")
    print(f"Wrote {result['selected_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
