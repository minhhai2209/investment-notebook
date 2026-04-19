from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.build_range_forecast_report import _variant_train_slice
from scripts.analysis.evaluate_ohlc_models import (
    FEATURE_COLUMNS,
    _load_daily_ohlcv,
    _normalise_ticker,
    build_ticker_ohlc_sample,
)
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache, first_timestamp_and_count, last_timestamp
from scripts.data_fetching.market_members import fetch_vn30_members


VN_TZ = timezone(timedelta(hours=7))
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_HISTORY_CALENDAR_DAYS = 1100
DEFAULT_RECENT_FOCUS_DATES = 252
DEFAULT_QUARTER_FOCUS_DATES = 63
DEFAULT_HOLDOUT_DATES = 40
DEFAULT_SELL_DELAY_DAYS = 3
DEFAULT_MONTH_HORIZONS = (1, 2, 3, 4, 5, 6)
VARIANTS = ("full_2y", "recent_focus", "quarter_focus")
NUMERIC_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
CATEGORICAL_FEATURE_COLUMNS = ["Ticker"]
MODEL_NAMES = ("ridge", "random_forest", "hist_gbm")
TARGET_COLUMNS = ("TargetPeakRetPct", "TargetPeakDay", "TargetDrawdownPct")


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


def _month_to_trading_days(months: int) -> int:
    return int(months) * 21


def _resolve_tickers(tickers: Sequence[str] | None) -> List[str]:
    if tickers:
        return _ordered_unique(tickers)
    return _ordered_unique(fetch_vn30_members(timeout=30))


def _load_tickers_from_universe_csv(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {path}")
    df = pd.read_csv(path, usecols=["Ticker"])
    if "Ticker" not in df.columns:
        raise ValueError(f"{path} missing required column 'Ticker'")
    return _ordered_unique(df["Ticker"].dropna().astype(str).tolist())


def refresh_history_cache(tickers: Sequence[str], history_dir: Path, history_calendar_days: int) -> pd.DataFrame:
    history_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for ticker in _ordered_unique(["VNINDEX", *tickers]):
        ensure_ohlc_cache(ticker, outdir=str(history_dir), min_days=history_calendar_days, resolution="D")
        cache_path = history_dir / f"{ticker}_daily.csv"
        first_ts, count = first_timestamp_and_count(cache_path)
        last_ts = last_timestamp(cache_path)
        rows.append(
            {
                "Ticker": ticker,
                "Bars": int(count),
                "FirstDate": datetime.fromtimestamp(first_ts, tz=VN_TZ).strftime("%Y-%m-%d") if first_ts else "",
                "LastDate": datetime.fromtimestamp(last_ts, tz=VN_TZ).strftime("%Y-%m-%d") if last_ts else "",
            }
        )
    return pd.DataFrame(rows)


def _compute_cycle_targets(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon_days: int,
    sell_delay_days: int,
) -> pd.DataFrame:
    if horizon_days < sell_delay_days:
        raise ValueError("horizon_days must be >= sell_delay_days")

    high_windows = pd.concat(
        [((high.shift(-day) / close) - 1.0) * 100.0 for day in range(sell_delay_days, horizon_days + 1)],
        axis=1,
    )
    high_windows.columns = list(range(sell_delay_days, horizon_days + 1))
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

    return pd.DataFrame(
        {
            "TargetPeakRetPct": peak_ret,
            "TargetPeakDay": peak_day,
            "TargetDrawdownPct": drawdown,
        },
        index=close.index,
    )


def build_ticker_cycle_sample(
    ticker: str,
    history_dir: Path,
    horizon_months: Sequence[int],
    sell_delay_days: int,
) -> pd.DataFrame:
    normalized = _normalise_ticker(ticker)
    base_frame = build_ticker_ohlc_sample(normalized, history_dir, max_horizon=1)
    base_frame = (
        base_frame[["Date", "Ticker", "BaseClose"] + FEATURE_COLUMNS]
        .drop_duplicates(subset=["Date"])
        .copy()
    )
    base_frame["Date"] = pd.to_datetime(base_frame["Date"], errors="coerce")
    base_frame = base_frame.dropna(subset=["Date"]).set_index("Date", drop=False).sort_index()

    price_frame = _load_daily_ohlcv(normalized, history_dir)
    forecast_dates = pd.Series(price_frame.index, index=price_frame.index)

    frames: List[pd.DataFrame] = []
    for months in horizon_months:
        horizon_days = _month_to_trading_days(int(months))
        targets = _compute_cycle_targets(
            close=price_frame["Close"],
            high=price_frame["High"],
            low=price_frame["Low"],
            horizon_days=horizon_days,
            sell_delay_days=sell_delay_days,
        )
        frame = base_frame.copy()
        frame["HorizonMonths"] = int(months)
        frame["HorizonDays"] = int(horizon_days)
        frame["ForecastWindow"] = f"{int(months)}M"
        frame["ForecastDate"] = forecast_dates.shift(-horizon_days).reindex(frame.index)
        for column in TARGET_COLUMNS:
            frame[column] = targets[column].reindex(frame.index)
        frames.append(frame.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker", "HorizonDays"]).reset_index(drop=True)


def build_multi_ticker_cycle_sample(
    tickers: Sequence[str],
    history_dir: Path,
    horizon_months: Sequence[int],
    sell_delay_days: int,
) -> pd.DataFrame:
    frames = [
        build_ticker_cycle_sample(ticker, history_dir, horizon_months=horizon_months, sell_delay_days=sell_delay_days)
        for ticker in tickers
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    def make_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(NUMERIC_FEATURE_COLUMNS),
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    list(CATEGORICAL_FEATURE_COLUMNS),
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    return {
        "ridge": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=24,
                        max_depth=8,
                        max_features="sqrt",
                        min_samples_leaf=4,
                        n_jobs=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gbm": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=60,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _fit_predict_target(
    model_factory: Callable[[], Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> np.ndarray:
    model = model_factory()
    feature_columns = list(NUMERIC_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS)
    model.fit(train_df[feature_columns], train_df[target_column].astype(float))
    return model.predict(test_df[feature_columns])


def _clip_peak_day(values: pd.Series, sell_delay_days: int, horizon_days: int) -> pd.Series:
    return values.astype(float).clip(lower=float(sell_delay_days), upper=float(horizon_days))


def select_best_cycle_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.sort_values(
            ["Ticker", "HorizonMonths", "PeakRetMAEPct", "PeakDayMAE", "DrawdownMAEPct", "Variant", "Model"],
            ascending=[True, True, True, True, True, True, True],
        )
        .groupby(["Ticker", "HorizonMonths"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def build_selected_cycle_matrix(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()

    matrix = pd.DataFrame({"Ticker": sorted(selected_df["Ticker"].astype(str).unique())})
    fields = [
        "Variant",
        "Model",
        "PredPeakRetPct",
        "PredPeakDays",
        "PredPeakPrice",
        "PredDrawdownPct",
        "PredDrawdownPrice",
        "PeakRetMAEPct",
        "PeakDayMAE",
        "DrawdownMAEPct",
    ]
    for months in sorted(selected_df["HorizonMonths"].astype(int).unique()):
        scoped = selected_df[selected_df["HorizonMonths"].astype(int) == int(months)][["Ticker"] + fields].copy()
        scoped = scoped.rename(
            columns={
                field: f"{field}_{int(months)}M"
                for field in fields
            }
        )
        matrix = matrix.merge(scoped, on="Ticker", how="left")
    return matrix


def build_best_horizon_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    order_columns = [
        "Ticker",
        "SelectionScore",
        "PeakRetMAEPct",
        "PeakDayMAE",
        "DrawdownMAEPct",
        "HorizonMonths",
    ]
    available = [column for column in order_columns if column in selected_df.columns]
    return (
        selected_df.sort_values(available, ascending=[True, True, True, True, True, True][: len(available)])
        .groupby("Ticker", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def run_report(
    *,
    tickers: Sequence[str] | None,
    universe_csv: Path | None,
    history_dir: Path,
    output_dir: Path,
    history_calendar_days: int,
    horizon_months: Sequence[int],
    holdout_dates: int,
    recent_focus_dates: int,
    quarter_focus_dates: int,
    sell_delay_days: int,
) -> Dict[str, object]:
    if universe_csv is not None:
        normalized_tickers = _load_tickers_from_universe_csv(universe_csv)
    else:
        normalized_tickers = _resolve_tickers(tickers)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_summary = refresh_history_cache(normalized_tickers, history_dir, history_calendar_days)
    sample_df = build_multi_ticker_cycle_sample(
        normalized_tickers,
        history_dir=history_dir,
        horizon_months=horizon_months,
        sell_delay_days=sell_delay_days,
    )
    if sample_df.empty:
        raise RuntimeError("Cycle sample is empty; cannot build cycle forecast report")

    latest_date = pd.Timestamp(sample_df["Date"].max())
    model_factories = build_model_factories()
    metric_frames: List[pd.DataFrame] = []

    for variant in VARIANTS:
        for months in horizon_months:
            print(f"[cycle-forecast] eval variant={variant} horizon={int(months)}M", flush=True)
            scoped = sample_df[sample_df["HorizonMonths"].astype(int) == int(months)].copy()
            for ticker_name, ticker_scoped in scoped.groupby("Ticker", sort=False):
                labeled = ticker_scoped.dropna(subset=list(TARGET_COLUMNS)).copy()
                unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
                if not unique_dates:
                    continue
                if len(unique_dates) > holdout_dates:
                    eval_start = unique_dates[-holdout_dates]
                    train_eval_full = labeled[labeled["Date"] < eval_start].copy()
                    test_eval = labeled[labeled["Date"] >= eval_start].copy()
                    train_eval = _variant_train_slice(
                        train_eval_full,
                        variant,
                        recent_focus_dates,
                        quarter_focus_dates=quarter_focus_dates,
                    )
                    if not train_eval.empty and not test_eval.empty:
                        for model_name, factory in model_factories.items():
                            eval_out = test_eval[
                                [
                                    "Date",
                                    "Ticker",
                                    "HorizonMonths",
                                    "HorizonDays",
                                    "ForecastWindow",
                                    "BaseClose",
                                ]
                                + list(TARGET_COLUMNS)
                            ].copy()
                            for target_column in TARGET_COLUMNS:
                                eval_out[f"Pred{target_column}"] = _fit_predict_target(factory, train_eval, test_eval, target_column)
                            eval_out["PredTargetPeakDay"] = _clip_peak_day(
                                eval_out["PredTargetPeakDay"],
                                sell_delay_days=sell_delay_days,
                                horizon_days=int(ticker_scoped["HorizonDays"].iloc[0]),
                            )
                            metric_frames.append(
                                pd.DataFrame(
                                    [
                                        {
                                            "Ticker": ticker_name,
                                            "Variant": variant,
                                            "Model": model_name,
                                            "HorizonMonths": int(months),
                                            "HorizonDays": int(eval_out["HorizonDays"].iloc[0]),
                                            "ForecastWindow": str(eval_out["ForecastWindow"].iloc[0]),
                                            "EvalRows": int(eval_out.shape[0]),
                                            "PeakRetMAEPct": float(
                                                mean_absolute_error(
                                                    eval_out["TargetPeakRetPct"],
                                                    eval_out["PredTargetPeakRetPct"],
                                                )
                                            ),
                                            "PeakDayMAE": float(
                                                mean_absolute_error(
                                                    eval_out["TargetPeakDay"],
                                                    eval_out["PredTargetPeakDay"],
                                                )
                                            ),
                                            "DrawdownMAEPct": float(
                                                mean_absolute_error(
                                                    eval_out["TargetDrawdownPct"],
                                                    eval_out["PredTargetDrawdownPct"],
                                                )
                                            ),
                                        }
                                    ]
                                )
                            )

    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    if metrics_df.empty:
        raise RuntimeError("Cycle forecast metrics are empty; insufficient holdout coverage")

    metrics_df["SelectionScore"] = (
        metrics_df["PeakRetMAEPct"].astype(float)
        + (0.15 * metrics_df["PeakDayMAE"].astype(float))
        + (0.50 * metrics_df["DrawdownMAEPct"].astype(float))
    )
    best_configs_df = select_best_cycle_configs(metrics_df)

    current_frames: List[pd.DataFrame] = []
    for config in best_configs_df.to_dict(orient="records"):
        ticker_name = str(config["Ticker"])
        variant = str(config["Variant"])
        model_name = str(config["Model"])
        months = int(config["HorizonMonths"])
        print(
            f"[cycle-forecast] current ticker={ticker_name} variant={variant} horizon={months}M model={model_name}",
            flush=True,
        )
        scoped = sample_df[
            (sample_df["Ticker"].astype(str) == ticker_name)
            & (sample_df["HorizonMonths"].astype(int) == months)
        ].copy()
        labeled = scoped.dropna(subset=list(TARGET_COLUMNS)).copy()
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
            [
                "Date",
                "Ticker",
                "HorizonMonths",
                "HorizonDays",
                "ForecastWindow",
                "BaseClose",
                "ForecastDate",
            ]
        ].copy()
        forecast["Variant"] = variant
        forecast["Model"] = model_name
        for target_column in TARGET_COLUMNS:
            forecast[f"Pred{target_column}"] = _fit_predict_target(factory, train_full, current_rows, target_column)
        forecast["PredTargetPeakDay"] = _clip_peak_day(
            forecast["PredTargetPeakDay"],
            sell_delay_days=sell_delay_days,
            horizon_days=int(current_rows["HorizonDays"].iloc[0]),
        )
        current_frames.append(forecast)

    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    if current_df.empty:
        raise RuntimeError("Cycle current forecast is empty; unable to score latest rows")

    current_df["SnapshotDate"] = latest_date.strftime("%Y-%m-%d")
    current_df["PredPeakRetPct"] = current_df["PredTargetPeakRetPct"].astype(float)
    current_df["PredPeakDays"] = current_df["PredTargetPeakDay"].round(1)
    current_df["PredDrawdownPct"] = current_df["PredTargetDrawdownPct"].astype(float)
    current_df["PredPeakPrice"] = current_df["BaseClose"].astype(float) * (1.0 + (current_df["PredPeakRetPct"] / 100.0))
    current_df["PredDrawdownPrice"] = current_df["BaseClose"].astype(float) * (1.0 + (current_df["PredDrawdownPct"] / 100.0))

    selected_current_df = current_df.merge(
        best_configs_df[
            [
                "Ticker",
                "HorizonMonths",
                "Variant",
                "Model",
                "EvalRows",
                "PeakRetMAEPct",
                "PeakDayMAE",
                "DrawdownMAEPct",
                "SelectionScore",
            ]
        ],
        on=["Ticker", "HorizonMonths", "Variant", "Model"],
        how="inner",
    )
    selected_current_df = selected_current_df[
        [
            "SnapshotDate",
            "Ticker",
            "HorizonMonths",
            "HorizonDays",
            "ForecastWindow",
            "BaseClose",
            "Variant",
            "Model",
            "EvalRows",
            "PeakRetMAEPct",
            "PeakDayMAE",
            "DrawdownMAEPct",
            "SelectionScore",
            "PredPeakRetPct",
            "PredPeakDays",
            "PredPeakPrice",
            "PredDrawdownPct",
            "PredDrawdownPrice",
        ]
    ].sort_values(["Ticker", "HorizonMonths"]).reset_index(drop=True)

    ticker_matrix_df = build_selected_cycle_matrix(selected_current_df)
    best_horizon_df = build_best_horizon_summary(selected_current_df)

    history_summary.to_csv(output_dir / "cycle_forecast_history_summary.csv", index=False)
    metrics_df.sort_values(["Ticker", "HorizonMonths", "PeakRetMAEPct", "PeakDayMAE", "DrawdownMAEPct"]).to_csv(
        output_dir / "cycle_forecast_model_metrics.csv",
        index=False,
    )
    best_configs_df.to_csv(output_dir / "cycle_forecast_best_configs.csv", index=False)
    current_df.sort_values(["Ticker", "HorizonMonths", "Variant", "Model"]).to_csv(
        output_dir / "cycle_forecast_current_all.csv",
        index=False,
    )
    selected_current_df.to_csv(output_dir / "cycle_forecast_current_selected.csv", index=False)
    ticker_matrix_df.to_csv(output_dir / "cycle_forecast_ticker_matrix.csv", index=False)
    best_horizon_df.to_csv(output_dir / "cycle_forecast_best_horizon_by_ticker.csv", index=False)
    (output_dir / "cycle_forecast_summary.json").write_text(
        json.dumps(
            {
                "SnapshotDate": latest_date.strftime("%Y-%m-%d"),
                "UniverseSize": int(len(normalized_tickers)),
                "Tickers": normalized_tickers,
                "Variants": list(VARIANTS),
                "HorizonMonths": [int(month) for month in horizon_months],
                "SellDelayDays": int(sell_delay_days),
                "OutputFiles": {
                    "HistorySummary": "cycle_forecast_history_summary.csv",
                    "ModelMetrics": "cycle_forecast_model_metrics.csv",
                    "BestConfigs": "cycle_forecast_best_configs.csv",
                    "CurrentAll": "cycle_forecast_current_all.csv",
                    "CurrentSelected": "cycle_forecast_current_selected.csv",
                    "TickerMatrix": "cycle_forecast_ticker_matrix.csv",
                    "BestHorizonByTicker": "cycle_forecast_best_horizon_by_ticker.csv",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "SnapshotDate": latest_date.strftime("%Y-%m-%d"),
        "HistorySummary": history_summary,
        "Metrics": metrics_df,
        "BestConfigs": best_configs_df,
        "CurrentAll": current_df,
        "CurrentSelected": selected_current_df,
        "TickerMatrix": ticker_matrix_df,
        "BestHorizonByTicker": best_horizon_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-ticker cycle forecast report that estimates achievable peak return, "
            "time-to-peak, and drawdown within 1M..6M windows."
        )
    )
    parser.add_argument("--tickers", nargs="*", default=None, help="Explicit ticker list. Default: live VN30.")
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=None,
        help="Optional universe CSV whose Ticker column defines the report universe.",
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Daily cache directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output artifacts.")
    parser.add_argument(
        "--history-calendar-days",
        type=int,
        default=DEFAULT_HISTORY_CALENDAR_DAYS,
        help="Calendar-day lookback used to seed the daily cache.",
    )
    parser.add_argument(
        "--holdout-dates",
        type=int,
        default=DEFAULT_HOLDOUT_DATES,
        help="Last N trading dates reserved for holdout metrics.",
    )
    parser.add_argument(
        "--recent-focus-dates",
        type=int,
        default=DEFAULT_RECENT_FOCUS_DATES,
        help="Training-date cap used by the recent-focus variant.",
    )
    parser.add_argument(
        "--quarter-focus-dates",
        type=int,
        default=DEFAULT_QUARTER_FOCUS_DATES,
        help="Training-date cap used by the quarter-focus variant.",
    )
    parser.add_argument(
        "--sell-delay-days",
        type=int,
        default=DEFAULT_SELL_DELAY_DAYS,
        help="Earliest trading day that is considered sellable inside each cycle window.",
    )
    parser.add_argument(
        "--horizon-months",
        nargs="*",
        type=int,
        default=list(DEFAULT_MONTH_HORIZONS),
        help="Cycle windows in month units (converted to ~21 trading days each).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_report(
        tickers=args.tickers,
        universe_csv=args.universe_csv,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        history_calendar_days=int(args.history_calendar_days),
        horizon_months=[int(value) for value in args.horizon_months],
        holdout_dates=int(args.holdout_dates),
        recent_focus_dates=int(args.recent_focus_dates),
        quarter_focus_dates=int(args.quarter_focus_dates),
        sell_delay_days=int(args.sell_delay_days),
    )
    output_dir = Path(args.output_dir)
    print(f"SnapshotDate: {results['SnapshotDate']}")
    print(f"Wrote {output_dir / 'cycle_forecast_history_summary.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_model_metrics.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_best_configs.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_current_all.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_current_selected.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_ticker_matrix.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_best_horizon_by_ticker.csv'}")
    print(f"Wrote {output_dir / 'cycle_forecast_summary.json'}")


if __name__ == "__main__":
    main()
