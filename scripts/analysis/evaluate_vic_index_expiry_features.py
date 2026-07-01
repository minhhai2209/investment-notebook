from __future__ import annotations

import argparse
import json
import warnings
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _attach_actual_return_columns,
    _load_daily_ohlcv,
    _load_optional_daily_ohlcv,
    _normalise_ticker,
    _reconstruct_ohlc_frame,
    build_ticker_ohlc_sample,
)


DEFAULT_TICKER = "VIC"
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 20)
DEFAULT_MIN_TRAIN_DATES = 120
DEFAULT_RETRAIN_EVERY = 20
DEFAULT_MODELS = ("random_forest",)
VIN_FAMILY_TICKERS = ("VIC", "VHM", "VRE", "VPL")
INDEX_TICKERS = ("VNINDEX", "VN30", "VN100")
OUTPUT_CURRENT = "vic_index_expiry_current_forecasts.csv"
OUTPUT_CURRENT_CANDIDATES = "vic_index_expiry_current_candidates.csv"
OUTPUT_METRICS = "vic_index_expiry_model_metrics.csv"
OUTPUT_HISTORY = "vic_index_expiry_prediction_history.csv"
OUTPUT_FEATURES = "vic_index_expiry_latest_features.csv"
OUTPUT_SUMMARY = "vic_index_expiry_summary.json"


def build_experiment_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    def make_numeric_pipeline(model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "ridge": lambda: make_numeric_pipeline(Ridge(alpha=1.0)),
        "random_forest": lambda: make_numeric_pipeline(
            RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            )
        ),
        "hist_gbm": lambda: make_numeric_pipeline(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=120,
                random_state=42,
            )
        ),
    }


def _third_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (3 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def _month_iter(start: pd.Timestamp, end: pd.Timestamp) -> List[tuple[int, int]]:
    months: List[tuple[int, int]] = []
    year = int(start.year)
    month = int(start.month)
    while (year, month) <= (int(end.year), int(end.month)):
        months.append((year, month))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return months


def resolve_monthly_derivative_expiries(trading_dates: Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
    """Return monthly VN30 futures expiry proxies aligned to available trading dates.

    Vietnam index futures normally expire on the third Thursday. If that date is
    not in the local cache, use the nearest previous trading date in the same
    month, which handles holidays without inventing non-trading observations.
    """

    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize().sort_values().unique()
    if len(dates) == 0:
        return pd.DatetimeIndex([])

    expiries: List[pd.Timestamp] = []
    month_start = dates.min().replace(day=1)
    month_end_raw = dates.max() + pd.DateOffset(months=1)
    month_end = month_end_raw.replace(day=monthrange(int(month_end_raw.year), int(month_end_raw.month))[1])
    for year, month in _month_iter(month_start, month_end):
        theoretical = pd.Timestamp(_third_thursday(year, month))
        same_month = dates[(dates.year == year) & (dates.month == month)]
        if same_month.empty:
            continue
        eligible = same_month[same_month <= theoretical]
        expiries.append(eligible.max() if not eligible.empty else same_month.min())
    return pd.DatetimeIndex(sorted(set(expiries)))


def _next_theoretical_expiry_on_or_after(current_date: pd.Timestamp) -> pd.Timestamp:
    current = pd.Timestamp(current_date).normalize()
    for offset in range(0, 14):
        month = int(current.month) + offset
        year = int(current.year) + ((month - 1) // 12)
        month = ((month - 1) % 12) + 1
        expiry = pd.Timestamp(_third_thursday(year, month))
        if expiry >= current:
            return expiry
    raise RuntimeError(f"Cannot resolve next derivative expiry for {current.date()}")


def build_derivative_expiry_features(trading_dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize().sort_values().unique()
    expiries = resolve_monthly_derivative_expiries(dates)
    positions = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    expiry_positions = [int(positions.loc[expiry]) for expiry in expiries if expiry in positions.index]

    rows: List[Dict[str, object]] = []
    for pos, current_date in enumerate(dates):
        previous_positions = [expiry_pos for expiry_pos in expiry_positions if expiry_pos <= pos]
        next_positions = [expiry_pos for expiry_pos in expiry_positions if expiry_pos >= pos]
        previous_pos = max(previous_positions) if previous_positions else np.nan
        next_pos = min(next_positions) if next_positions else np.nan
        days_from = float(pos - previous_pos) if np.isfinite(previous_pos) else np.nan
        days_to = float(next_pos - pos) if np.isfinite(next_pos) else np.nan
        if not np.isfinite(days_to):
            next_expiry = _next_theoretical_expiry_on_or_after(current_date)
            days_to = float(max(len(pd.bdate_range(current_date, next_expiry)) - 1, 0))
        cycle_denominator = days_from + days_to if np.isfinite(days_from) and np.isfinite(days_to) else np.nan
        rows.append(
            {
                "Date": current_date,
                "DerivExpiryDay": float(days_to == 0.0),
                "DerivPreExpiry3Sessions": float(np.isfinite(days_to) and 0.0 <= days_to <= 3.0),
                "DerivPostExpiry3Sessions": float(np.isfinite(days_from) and 0.0 <= days_from <= 3.0),
                "DerivExpiryWeek": float(
                    (np.isfinite(days_to) and 0.0 <= days_to <= 2.0)
                    or (np.isfinite(days_from) and 0.0 <= days_from <= 2.0)
                ),
                "DerivDaysToExpiry": days_to,
                "DerivDaysFromExpiry": days_from,
                "DerivExpiryCyclePos": float(days_from / cycle_denominator) if cycle_denominator else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _series_return_pct(current: pd.Series, anchor: pd.Series) -> pd.Series:
    return ((current.astype(float) / anchor.astype(float).replace(0.0, np.nan)) - 1.0) * 100.0


def _range_position(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    denominator = (rolling_max - rolling_min).replace(0.0, np.nan)
    return (series - rolling_min) / denominator


def _load_available_close_matrix(history_dir: Path) -> pd.DataFrame:
    frames: List[pd.Series] = []
    for path in sorted(history_dir.glob("*_daily.csv")):
        ticker = _normalise_ticker(path.name.removesuffix("_daily.csv"))
        if not ticker or ticker in INDEX_TICKERS:
            continue
        frame = _load_optional_daily_ohlcv(ticker, history_dir)
        if frame.empty:
            continue
        frames.append(frame["Close"].rename(ticker))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def build_ex_vin_market_features(history_dir: Path, ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    ticker = _normalise_ticker(ticker)
    close_matrix = _load_available_close_matrix(history_dir)
    if close_matrix.empty:
        return pd.DataFrame(columns=["Date"])

    index_df = _load_daily_ohlcv("VNINDEX", history_dir)
    ticker_df = _load_daily_ohlcv(ticker, history_dir)
    vin_columns = [column for column in close_matrix.columns if column in VIN_FAMILY_TICKERS]
    ex_vin_columns = [column for column in close_matrix.columns if column not in VIN_FAMILY_TICKERS]
    if not ex_vin_columns:
        raise RuntimeError("Cannot build ex-Vin proxy: no non-Vin equity histories are available.")

    returns = close_matrix.pct_change(fill_method=None) * 100.0
    ex_vin_ret1 = returns[ex_vin_columns].mean(axis=1)
    vin_ret1 = returns[vin_columns].mean(axis=1) if vin_columns else pd.Series(np.nan, index=returns.index)
    ex_vin_index = (1.0 + (ex_vin_ret1 / 100.0)).cumprod() * 100.0
    vin_index = (1.0 + (vin_ret1 / 100.0)).cumprod() * 100.0

    index_close = index_df["Close"].reindex(ex_vin_index.index).ffill()
    ticker_close = ticker_df["Close"].reindex(ex_vin_index.index).ffill()
    index_ret1 = index_close.pct_change(fill_method=None) * 100.0
    ticker_ret1 = ticker_close.pct_change(fill_method=None) * 100.0

    features = pd.DataFrame(index=ex_vin_index.index)
    features["ExVinRet1Pct"] = ex_vin_ret1
    features["ExVinRet5Pct"] = ex_vin_ret1.rolling(5).sum()
    features["ExVinRet20Pct"] = ex_vin_ret1.rolling(20).sum()
    features["ExVinRangePos20"] = _range_position(ex_vin_index, 20)
    features["ExVinRangePos60"] = _range_position(ex_vin_index, 60)
    features["ExVinVolatility10"] = ex_vin_ret1.rolling(10).std()
    features["VinBasketRet1Pct"] = vin_ret1
    features["VinBasketRet5Pct"] = vin_ret1.rolling(5).sum()
    features["VinBasketRet20Pct"] = vin_ret1.rolling(20).sum()
    features["VinBasketRangePos20"] = _range_position(vin_index, 20)
    features["VNIndexRet1MinusExVinRet1Pct"] = index_ret1 - ex_vin_ret1
    features["VNIndexRet5MinusExVinRet5Pct"] = index_ret1.rolling(5).sum() - features["ExVinRet5Pct"]
    features["VNIndexRet20MinusExVinRet20Pct"] = index_ret1.rolling(20).sum() - features["ExVinRet20Pct"]
    features["TickerRet1MinusExVinRet1Pct"] = ticker_ret1 - ex_vin_ret1
    features["TickerRet5MinusExVinRet5Pct"] = ticker_ret1.rolling(5).sum() - features["ExVinRet5Pct"]
    features["TickerCorr20ExVin"] = ticker_ret1.rolling(20).corr(ex_vin_ret1)
    features["TickerBeta20ExVin"] = ticker_ret1.rolling(20).cov(ex_vin_ret1) / ex_vin_ret1.rolling(20).var()
    features["Date"] = features.index
    return features.reset_index(drop=True)


def add_index_expiry_features(sample_df: pd.DataFrame, history_dir: Path, ticker: str = DEFAULT_TICKER) -> tuple[pd.DataFrame, List[str]]:
    out = sample_df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    expiry_features = build_derivative_expiry_features(out["Date"].drop_duplicates())
    market_features = build_ex_vin_market_features(history_dir, ticker=ticker)
    added_columns = [
        column
        for column in list(expiry_features.columns) + list(market_features.columns)
        if column != "Date"
    ]
    feature_by_date = expiry_features.merge(market_features, on="Date", how="left")
    out = out.merge(feature_by_date, on="Date", how="left")

    lag_columns: List[str] = []
    for column in added_columns:
        if column not in out.columns:
            continue
        for lag in (1, 2, 3):
            lag_column = f"{column}_Lag{lag}"
            out[lag_column] = out.groupby(["Ticker", "Horizon"], sort=False)[column].shift(lag)
            lag_columns.append(lag_column)
    return out.replace([np.inf, -np.inf], np.nan), added_columns + lag_columns


def _fit_predict_target(
    model_factory: Callable[[], Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> np.ndarray:
    model = model_factory()
    model.fit(train_df[list(feature_columns)], train_df[target_column].astype(float))
    return model.predict(test_df[list(feature_columns)])


def _score_history(frame: pd.DataFrame) -> Dict[str, float]:
    close_mae = float(mean_absolute_error(frame["ActualCloseRetPct"], frame["PredCloseRetPct"]))
    range_mae = float(mean_absolute_error(frame["ActualRangePct"], frame["PredRangePct"]))
    cum_high_mae = float(mean_absolute_error(frame["ActualCumHighRetPct"], frame["PredCumHighRetPct"]))
    cum_low_mae = float(mean_absolute_error(frame["ActualCumLowRetPct"], frame["PredCumLowRetPct"]))
    cum_upside_miss = float((frame["ActualCumHighRetPct"] - frame["PredCumHighRetPct"]).clip(lower=0.0).mean())
    cum_downside_miss = float((frame["PredCumLowRetPct"] - frame["ActualCumLowRetPct"]).clip(lower=0.0).mean())
    direction_hit = float(
        (
            np.sign(frame["ActualCloseRetPct"].astype(float))
            == np.sign(frame["PredCloseRetPct"].astype(float))
        ).mean()
        * 100.0
    )
    selection_score = float(
        close_mae
        + (0.35 * range_mae)
        + (0.90 * cum_upside_miss)
        + (0.60 * cum_downside_miss)
        + (0.40 * cum_high_mae)
        + (0.20 * cum_low_mae)
        - (0.01 * direction_hit)
    )
    return {
        "EvalRows": int(frame.shape[0]),
        "CloseMAEPct": close_mae,
        "RangeMAEPct": range_mae,
        "CumHighMAEPct": cum_high_mae,
        "CumLowMAEPct": cum_low_mae,
        "CumUpsideMissMAEPct": cum_upside_miss,
        "CumDownsideMissMAEPct": cum_downside_miss,
        "CloseDirHitPct": direction_hit,
        "SelectionScore": selection_score,
    }


def walk_forward_feature_experiment(
    sample_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    min_train_dates: int,
    retrain_every: int,
    model_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factories = build_experiment_model_factories()
    model_factories = {name: factories[name] for name in model_names if name in factories}
    if not model_factories:
        raise ValueError(f"No supported models selected: {', '.join(model_names)}")
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

    for feature_set_name, feature_columns in feature_sets.items():
        for model_name, factory in model_factories.items():
            for (ticker, horizon), scoped_df in sample_df.groupby(["Ticker", "Horizon"], sort=False):
                labeled = scoped_df[scoped_df["TargetOpenRetPct"].notna()].copy()
                unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
                if len(unique_dates) <= min_train_dates:
                    continue
                eval_dates = unique_dates[min_train_dates:]
                latest_date = scoped_df["Date"].max()

                block_predictions: List[pd.DataFrame] = []
                for start in range(0, len(eval_dates), retrain_every):
                    block_dates = eval_dates[start : start + retrain_every]
                    train_df = labeled[labeled["Date"] < block_dates[0]].copy()
                    block_df = labeled[labeled["Date"].isin(block_dates)].copy()
                    if train_df.empty or block_df.empty:
                        continue
                    out = block_df[
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
                            "ActualCumHigh",
                            "ActualCumLow",
                        ]
                        + TARGET_COLUMNS
                    ].copy()
                    out["FeatureSet"] = feature_set_name
                    out["Model"] = model_name
                    for target_column in TARGET_COLUMNS:
                        out[f"Pred{target_column}"] = _fit_predict_target(
                            factory,
                            train_df,
                            block_df,
                            feature_columns,
                            target_column,
                        )
                    out = pd.concat([out, _reconstruct_ohlc_frame(out)], axis=1)
                    out = _attach_actual_return_columns(out)
                    block_predictions.append(out)
                if block_predictions:
                    history_frames.append(pd.concat(block_predictions, ignore_index=True))

                train_all = labeled[labeled["Date"] < latest_date].copy()
                current_row = scoped_df[scoped_df["Date"] == latest_date].copy()
                if train_all.empty or current_row.empty:
                    continue
                forecast = current_row[["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"]].copy()
                forecast["FeatureSet"] = feature_set_name
                forecast["Model"] = model_name
                for target_column in TARGET_COLUMNS:
                    forecast[f"Pred{target_column}"] = _fit_predict_target(
                        factory,
                        train_all,
                        current_row,
                        feature_columns,
                        target_column,
                    )
                forecast = pd.concat([forecast, _reconstruct_ohlc_frame(forecast)], axis=1)
                current_frames.append(forecast)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    metrics_rows: List[Dict[str, object]] = []
    if not history_df.empty:
        for (ticker, horizon, feature_set, model), group in history_df.groupby(
            ["Ticker", "Horizon", "FeatureSet", "Model"],
            sort=False,
        ):
            metrics_rows.append(
                {
                    "Ticker": ticker,
                    "Horizon": int(horizon),
                    "FeatureSet": feature_set,
                    "Model": model,
                    **_score_history(group),
                }
            )
    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["Ticker", "Horizon", "SelectionScore", "CumUpsideMissMAEPct", "CloseMAEPct"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    return history_df, current_df, metrics_df


def select_best_current_forecasts(current_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty or metrics_df.empty:
        return pd.DataFrame()
    best_keys = metrics_df.sort_values("SelectionScore").groupby(["Ticker", "Horizon"], as_index=False).first()
    merged = current_df.merge(
        best_keys[["Ticker", "Horizon", "FeatureSet", "Model"]],
        on=["Ticker", "Horizon", "FeatureSet", "Model"],
        how="inner",
    )
    metric_columns = [
        "Ticker",
        "Horizon",
        "FeatureSet",
        "Model",
        "EvalRows",
        "CloseMAEPct",
        "RangeMAEPct",
        "CumHighMAEPct",
        "CumLowMAEPct",
        "CumUpsideMissMAEPct",
        "CumDownsideMissMAEPct",
        "CloseDirHitPct",
        "SelectionScore",
    ]
    return merged.merge(metrics_df[metric_columns], on=["Ticker", "Horizon", "FeatureSet", "Model"], how="left")


def run_experiment(
    *,
    ticker: str,
    history_dir: Path,
    output_dir: Path,
    horizons: Sequence[int],
    min_train_dates: int,
    retrain_every: int,
    model_names: Sequence[str],
) -> Dict[str, object]:
    ticker = _normalise_ticker(ticker)
    max_horizon = max(int(horizon) for horizon in horizons)
    sample = build_ticker_ohlc_sample(ticker, history_dir, max_horizon=max_horizon)
    sample = sample[sample["Horizon"].isin([int(horizon) for horizon in horizons])].copy()
    enhanced_sample, added_columns = add_index_expiry_features(sample, history_dir, ticker=ticker)
    baseline_columns = list(FEATURE_COLUMNS)
    enhanced_columns = baseline_columns + added_columns
    feature_sets = {
        "baseline_ohlc": baseline_columns,
        "index_expiry_exvin": enhanced_columns,
    }
    history_df, current_df, metrics_df = walk_forward_feature_experiment(
        enhanced_sample,
        feature_sets,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
        model_names=model_names,
    )
    best_current = select_best_current_forecasts(current_df, metrics_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_dir / OUTPUT_HISTORY, index=False)
    best_current.to_csv(output_dir / OUTPUT_CURRENT, index=False)
    current_df.to_csv(output_dir / OUTPUT_CURRENT_CANDIDATES, index=False)
    metrics_df.to_csv(output_dir / OUTPUT_METRICS, index=False)
    latest_features = enhanced_sample[enhanced_sample["Date"].eq(enhanced_sample["Date"].max())][
        ["Date", "Ticker", "Horizon", "BaseClose"] + added_columns
    ].copy()
    latest_features.to_csv(output_dir / OUTPUT_FEATURES, index=False)

    summary = {
        "ticker": ticker,
        "latest_date": str(enhanced_sample["Date"].max().date()),
        "horizons": [int(horizon) for horizon in horizons],
        "feature_sets": list(feature_sets.keys()),
        "model_candidates": list(model_names),
        "added_feature_count": len(added_columns),
        "added_features": added_columns,
        "history_rows": int(history_df.shape[0]),
        "current_rows": int(best_current.shape[0]),
        "current_candidate_rows": int(current_df.shape[0]),
        "metrics_rows": int(metrics_df.shape[0]),
        "best_current": best_current[
            [
                "Ticker",
                "Horizon",
                "FeatureSet",
                "Model",
                "BaseClose",
                "PredCumHigh",
                "PredClose",
                "PredCumHighRetPct",
                "PredCloseRetPct",
                "CumHighMAEPct",
                "CloseMAEPct",
                "CloseDirHitPct",
                "SelectionScore",
            ]
        ].to_dict(orient="records")
        if not best_current.empty
        else [],
    }
    (output_dir / OUTPUT_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VIC OHLC forecasts with VNINDEX, ex-Vin proxy, and derivative-expiry features."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--horizons",
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZONS),
        help="Comma-separated forecast horizons, e.g. 1,3,5,10.",
    )
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--retrain-every", type=int, default=DEFAULT_RETRAIN_EVERY)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names: random_forest,hist_gbm,ridge.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values.*",
        category=UserWarning,
    )
    args = parse_args()
    horizons = [int(raw.strip()) for raw in str(args.horizons).split(",") if raw.strip()]
    model_names = [raw.strip() for raw in str(args.models).split(",") if raw.strip()]
    summary = run_experiment(
        ticker=args.ticker,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        horizons=horizons,
        min_train_dates=args.min_train_dates,
        retrain_every=args.retrain_every,
        model_names=model_names,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
