from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_CASE_TICKERS = ["HPG", "FPT", "SSI", "VCB", "NKG", "GAS", "PLX", "REE"]
DEFAULT_MAX_HORIZON = 10
LAGS = (1, 2, 3)
TARGET_COLUMNS = [
    "TargetOpenRetPct",
    "TargetCloseRetPct",
    "TargetUpperWickPct",
    "TargetLowerWickPct",
]


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _resolve_column(frame: pd.DataFrame, column: str) -> str:
    for candidate in (column, column.lower(), column.upper(), column.capitalize()):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Missing required column '{column}'")


def _load_daily_ohlcv(ticker: str, history_dir: Path) -> pd.DataFrame:
    path = history_dir / f"{_normalise_ticker(ticker)}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise RuntimeError(f"Empty history file: {path}")
    date_col = "date_vn" if "date_vn" in raw.columns else _resolve_column(raw, "t")
    if date_col == "date_vn":
        index = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        index = pd.to_datetime(pd.to_numeric(raw[date_col], errors="coerce"), unit="s", errors="coerce")
    df = pd.DataFrame(
        {
            "Open": pd.to_numeric(raw[_resolve_column(raw, "open")], errors="coerce").to_numpy(),
            "High": pd.to_numeric(raw[_resolve_column(raw, "high")], errors="coerce").to_numpy(),
            "Low": pd.to_numeric(raw[_resolve_column(raw, "low")], errors="coerce").to_numpy(),
            "Close": pd.to_numeric(raw[_resolve_column(raw, "close")], errors="coerce").to_numpy(),
            "Volume": pd.to_numeric(raw[_resolve_column(raw, "volume")], errors="coerce").to_numpy(),
        },
        index=index,
    )
    df.index.name = "Date"
    return df[~df.index.isna()].sort_index()


def _load_optional_daily_ohlcv(ticker: str, history_dir: Path) -> pd.DataFrame:
    path = history_dir / f"{_normalise_ticker(ticker)}_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    return _load_daily_ohlcv(ticker, history_dir)


def _range_position(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    denominator = (rolling_max - rolling_min).replace(0.0, np.nan)
    return (series - rolling_min) / denominator


def _series_ratio_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned_numerator = numerator.astype(float)
    aligned_denominator = denominator.astype(float).replace(0.0, np.nan)
    return (aligned_numerator / aligned_denominator) * 100.0


def _series_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned_numerator = numerator.astype(float)
    aligned_denominator = denominator.astype(float).replace(0.0, np.nan)
    return aligned_numerator / aligned_denominator


def _series_return_pct(current: pd.Series, anchor: pd.Series) -> pd.Series:
    aligned_current = current.astype(float)
    aligned_anchor = anchor.astype(float).replace(0.0, np.nan)
    return ((aligned_current / aligned_anchor) - 1.0) * 100.0


def _adaptive_abs_threshold(
    series: pd.Series,
    *,
    window: int,
    quantile: float,
    floor: float,
    std_multiplier: float = 1.0,
) -> pd.Series:
    absolute = series.astype(float).abs()
    quantile_threshold = absolute.rolling(window, min_periods=window).quantile(quantile).shift(1)
    std_threshold = absolute.rolling(window, min_periods=window).std().shift(1) * float(std_multiplier)
    threshold = pd.concat([quantile_threshold, std_threshold], axis=1).max(axis=1)
    return threshold.clip(lower=float(floor))


def _three_state_signal(series: pd.Series, threshold: pd.Series) -> pd.Series:
    values = series.astype(float)
    limits = threshold.astype(float)
    state = pd.Series(0.0, index=values.index, dtype=float)
    valid = limits.notna()
    state.loc[valid & (values >= limits)] = 1.0
    state.loc[valid & (values <= -limits)] = -1.0
    return state


def _direction_streak_state(series: pd.Series, *, min_streak: int = 2) -> pd.Series:
    values = series.astype(float)
    positive = values.gt(0.0)
    negative = values.lt(0.0)
    positive_streak = positive.groupby((~positive).cumsum()).cumsum()
    negative_streak = negative.groupby((~negative).cumsum()).cumsum()
    state = pd.Series(0.0, index=values.index, dtype=float)
    state.loc[positive_streak >= int(min_streak)] = 1.0
    state.loc[negative_streak >= int(min_streak)] = -1.0
    return state


def _limit_proxy_state(series: pd.Series, *, limit_pct: float = 6.75) -> pd.Series:
    values = series.astype(float)
    state = pd.Series(0.0, index=values.index, dtype=float)
    state.loc[values >= float(limit_pct)] = 1.0
    state.loc[values <= -float(limit_pct)] = -1.0
    return state


def _build_weekly_context_features(
    open_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    volume_series: pd.Series,
) -> Dict[str, pd.Series]:
    dt_index = pd.DatetimeIndex(close_series.index)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    week_period = dt_index.to_period("W-FRI")
    week_keys = pd.Series(week_period, index=close_series.index)

    weekly = pd.DataFrame(
        {
            "Close": close_series.groupby(week_period).last(),
            "High": high_series.groupby(week_period).max(),
            "Low": low_series.groupby(week_period).min(),
            "Volume": volume_series.groupby(week_period).sum(),
        }
    ).sort_index()

    prev_week_close = week_keys.map(weekly["Close"].shift(1)).astype(float)
    prev_week_high = week_keys.map(weekly["High"].shift(1)).astype(float)
    prev_week_low = week_keys.map(weekly["Low"].shift(1)).astype(float)
    prev_week_volume = week_keys.map(weekly["Volume"].shift(1)).astype(float)

    prev_week_ret = week_keys.map(_series_return_pct(weekly["Close"], weekly["Close"].shift(1)).shift(1)).astype(float)
    prev_week_range = week_keys.map(_series_ratio_pct(weekly["High"] - weekly["Low"], weekly["Close"].shift(1)).shift(1)).astype(float)
    prev_week_vol_ratio4 = week_keys.map(_series_ratio(weekly["Volume"], weekly["Volume"].rolling(4).mean()).shift(1)).astype(float)

    week_to_date_high = high_series.groupby(week_period).cummax()
    week_to_date_low = low_series.groupby(week_period).cummin()
    week_to_date_volume = volume_series.groupby(week_period).cumsum()

    return {
        "WeekToDateRetPct": _series_return_pct(close_series, prev_week_close),
        "WeekToDateRangePct": _series_ratio_pct(week_to_date_high - week_to_date_low, prev_week_close),
        "WeekToDateVolumePctPrevWeek": _series_ratio_pct(week_to_date_volume, prev_week_volume),
        "DistPrevWeekHighPct": _series_return_pct(close_series, prev_week_high),
        "DistPrevWeekLowPct": _series_return_pct(close_series, prev_week_low),
        "PrevWeekRetPct": prev_week_ret,
        "PrevWeekRangePct": prev_week_range,
        "PrevWeekVolRatio4": prev_week_vol_ratio4,
    }


def _build_feature_columns() -> List[str]:
    columns = [
        "TickerGapPct",
        "TickerBodyPct",
        "TickerRangePct",
        "TickerUpperWickPct",
        "TickerLowerWickPct",
        "TickerRet1Pct",
        "TickerRet5Pct",
        "TickerRet20Pct",
        "TickerDistSMA20Pct",
        "TickerDistSMA50Pct",
        "TickerRangePos20",
        "TickerRangePos60",
        "TickerVolRatio20",
        "TickerVolatility10",
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
        "TickerWeekToDateRetPct",
        "TickerWeekToDateRangePct",
        "TickerWeekToDateVolumePctPrevWeek",
        "TickerDistPrevWeekHighPct",
        "TickerDistPrevWeekLowPct",
        "TickerPrevWeekRetPct",
        "TickerPrevWeekRangePct",
        "TickerPrevWeekVolRatio4",
        "Rel5Pct",
        "Rel20Pct",
        "Rel5PctVsVN30",
        "RelWeekToDateRetPct",
        "RelPrevWeekRetPct",
        "Corr20",
        "Beta20",
        "IndexGapPct",
        "IndexBodyPct",
        "IndexRangePct",
        "IndexRet1Pct",
        "IndexRet5Pct",
        "IndexRet20Pct",
        "IndexDistSMA20Pct",
        "IndexDistSMA50Pct",
        "IndexRangePos20",
        "IndexRangePos60",
        "IndexVolRatio20",
        "IndexVolatility10",
        "IndexColorStreakState",
        "IndexWeekToDateRetPct",
        "IndexWeekToDateRangePct",
        "IndexWeekToDateVolumePctPrevWeek",
        "IndexDistPrevWeekHighPct",
        "IndexDistPrevWeekLowPct",
        "IndexPrevWeekRetPct",
        "IndexPrevWeekRangePct",
        "IndexPrevWeekVolRatio4",
        "VN30Ret1Pct",
        "VN30Ret5Pct",
        "VN30Ret20Pct",
        "VN30DistSMA20Pct",
        "VN30RangePos20",
        "VN30VolRatio20",
        "VN30ColorStreakState",
        "Corr20VN30",
        "Beta20VN30",
    ]
    for base_name in (
        "TickerRet1Pct",
        "TickerRangePct",
        "TickerVolRatio20",
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
        "Rel5Pct",
        "Rel5PctVsVN30",
        "IndexRet1Pct",
        "IndexRangePct",
        "IndexColorStreakState",
        "VN30Ret1Pct",
        "VN30ColorStreakState",
    ):
        for lag in LAGS:
            columns.append(f"{base_name}_Lag{lag}")
    return columns


FEATURE_COLUMNS = _build_feature_columns()


def build_ticker_ohlc_sample(
    ticker: str,
    history_dir: Path,
    max_horizon: int = DEFAULT_MAX_HORIZON,
) -> pd.DataFrame:
    ticker = _normalise_ticker(ticker)
    ticker_df = _load_daily_ohlcv(ticker, history_dir).add_prefix("Ticker")
    index_df = _load_daily_ohlcv("VNINDEX", history_dir).add_prefix("Index")
    vn30_df = _load_optional_daily_ohlcv("VN30", history_dir).add_prefix("VN30")
    merged = ticker_df.join(index_df, how="inner")
    if not vn30_df.empty:
        merged = merged.join(vn30_df, how="left")
    if merged.empty:
        raise RuntimeError(f"No overlapping dates for {ticker} and VNINDEX")

    ticker_close = merged["TickerClose"]
    ticker_open = merged["TickerOpen"]
    ticker_high = merged["TickerHigh"]
    ticker_low = merged["TickerLow"]
    ticker_volume = merged["TickerVolume"]
    index_close = merged["IndexClose"]
    index_open = merged["IndexOpen"]
    index_high = merged["IndexHigh"]
    index_low = merged["IndexLow"]
    index_volume = merged["IndexVolume"]

    ticker_prev_close = ticker_close.shift(1)
    index_prev_close = index_close.shift(1)
    ticker_sma20 = ticker_close.rolling(20).mean()
    ticker_ret1 = ticker_close.pct_change(1, fill_method=None) * 100.0
    ticker_ret5 = ticker_close.pct_change(5, fill_method=None) * 100.0
    ticker_ret20 = ticker_close.pct_change(20, fill_method=None) * 100.0
    ticker_dist_sma20 = ((ticker_close / ticker_sma20) - 1.0) * 100.0
    ticker_dist_sma50 = ((ticker_close / ticker_close.rolling(50).mean()) - 1.0) * 100.0
    ticker_range_pct = ((ticker_high - ticker_low) / ticker_prev_close) * 100.0
    ticker_body_pct = ((ticker_close - ticker_open) / ticker_prev_close) * 100.0
    ticker_gap_pct = ((ticker_open / ticker_prev_close) - 1.0) * 100.0
    ticker_upper_wick_pct = ((ticker_high - pd.concat([ticker_open, ticker_close], axis=1).max(axis=1)) / ticker_prev_close) * 100.0
    ticker_lower_wick_pct = ((pd.concat([ticker_open, ticker_close], axis=1).min(axis=1) - ticker_low) / ticker_prev_close) * 100.0
    ticker_vol_ratio20 = ticker_volume / ticker_volume.rolling(20).mean()
    ticker_volatility10 = ticker_close.pct_change(fill_method=None).rolling(10).std() * 100.0
    ticker_color_streak_state = _direction_streak_state(ticker_ret1, min_streak=2)
    ticker_limit_proxy_state = _limit_proxy_state(ticker_ret1, limit_pct=6.75)
    ticker_range_pos20 = _range_position(ticker_close, 20)
    ticker_range_pos60 = _range_position(ticker_close, 60)
    ticker_ret3 = ticker_close.pct_change(3, fill_method=None) * 100.0
    ticker_shock_threshold = _adaptive_abs_threshold(ticker_ret1, window=40, quantile=0.8, floor=3.0, std_multiplier=1.75)
    ticker_impulse_threshold = _adaptive_abs_threshold(ticker_ret3, window=60, quantile=0.8, floor=5.0, std_multiplier=1.5)
    signed_range_pct = ticker_range_pct.abs() * np.sign(ticker_body_pct.fillna(0.0))
    ticker_range_state_threshold = _adaptive_abs_threshold(ticker_range_pct, window=40, quantile=0.75, floor=3.5, std_multiplier=1.25)
    ticker_shock_state_1d = _three_state_signal(ticker_ret1, ticker_shock_threshold)
    ticker_impulse_state_3d = _three_state_signal(ticker_ret3, ticker_impulse_threshold)
    ticker_wide_range_state = _three_state_signal(signed_range_pct, ticker_range_state_threshold)
    ticker_trend_threshold = _adaptive_abs_threshold(ticker_ret5, window=60, quantile=0.75, floor=4.0, std_multiplier=1.25)
    ticker_trend_regime_state = pd.Series(0.0, index=merged.index, dtype=float)
    ticker_trend_regime_state.loc[
        (ticker_ret5 >= ticker_trend_threshold) & (ticker_range_pos20 >= 0.7)
    ] = 1.0
    ticker_trend_regime_state.loc[
        (ticker_ret5 <= -ticker_trend_threshold) & (ticker_range_pos20 <= 0.3)
    ] = -1.0
    compression_vol_threshold = ticker_volatility10.rolling(60, min_periods=60).quantile(0.35).shift(1).clip(lower=0.8)
    compression_range_threshold = ticker_range_pct.rolling(60, min_periods=60).quantile(0.35).shift(1).clip(lower=1.2)
    compression_mask = (
        (ticker_volatility10 <= compression_vol_threshold)
        & (ticker_range_pct <= compression_range_threshold)
    )
    ticker_compression_state = pd.Series(0.0, index=merged.index, dtype=float)
    ticker_compression_state.loc[
        compression_mask & (ticker_close >= ticker_sma20) & (ticker_range_pos20 >= 0.45)
    ] = 1.0
    ticker_compression_state.loc[
        compression_mask & (ticker_close < ticker_sma20) & (ticker_range_pos20 <= 0.55)
    ] = -1.0
    close_above_sma20 = ticker_close >= ticker_sma20
    ticker_reclaim_state = pd.Series(0.0, index=merged.index, dtype=float)
    ticker_reclaim_state.loc[
        close_above_sma20 & (~close_above_sma20.shift(1, fill_value=False)) & (ticker_ret1 > 0.0)
    ] = 1.0
    ticker_reclaim_state.loc[
        (~close_above_sma20) & close_above_sma20.shift(1, fill_value=False) & (ticker_ret1 < 0.0)
    ] = -1.0
    ticker_weekly = _build_weekly_context_features(
        ticker_open,
        ticker_high,
        ticker_low,
        ticker_close,
        ticker_volume,
    )

    index_ret1 = index_close.pct_change(1, fill_method=None) * 100.0
    index_ret5 = index_close.pct_change(5, fill_method=None) * 100.0
    index_ret20 = index_close.pct_change(20, fill_method=None) * 100.0
    index_dist_sma20 = ((index_close / index_close.rolling(20).mean()) - 1.0) * 100.0
    index_dist_sma50 = ((index_close / index_close.rolling(50).mean()) - 1.0) * 100.0
    index_range_pct = ((index_high - index_low) / index_prev_close) * 100.0
    index_body_pct = ((index_close - index_open) / index_prev_close) * 100.0
    index_gap_pct = ((index_open / index_prev_close) - 1.0) * 100.0
    index_vol_ratio20 = index_volume / index_volume.rolling(20).mean()
    index_volatility10 = index_close.pct_change(fill_method=None).rolling(10).std() * 100.0
    index_range_pos20 = _range_position(index_close, 20)
    index_range_pos60 = _range_position(index_close, 60)
    index_color_streak_state = _direction_streak_state(index_ret1, min_streak=2)
    index_weekly = _build_weekly_context_features(
        index_open,
        index_high,
        index_low,
        index_close,
        index_volume,
    )

    vn30_close = merged["VN30Close"] if "VN30Close" in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
    vn30_open = merged["VN30Open"] if "VN30Open" in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
    vn30_high = merged["VN30High"] if "VN30High" in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
    vn30_low = merged["VN30Low"] if "VN30Low" in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
    vn30_volume = merged["VN30Volume"] if "VN30Volume" in merged.columns else pd.Series(np.nan, index=merged.index, dtype=float)
    vn30_prev_close = vn30_close.shift(1)
    vn30_ret1 = vn30_close.pct_change(1, fill_method=None) * 100.0
    vn30_ret5 = vn30_close.pct_change(5, fill_method=None) * 100.0
    vn30_ret20 = vn30_close.pct_change(20, fill_method=None) * 100.0
    vn30_dist_sma20 = ((vn30_close / vn30_close.rolling(20).mean()) - 1.0) * 100.0
    vn30_range_pos20 = _range_position(vn30_close, 20)
    vn30_vol_ratio20 = vn30_volume / vn30_volume.rolling(20).mean()
    vn30_color_streak_state = _direction_streak_state(vn30_ret1, min_streak=2)

    rel5_pct = ticker_ret5 - index_ret5
    rel20_pct = ticker_ret20 - index_ret20
    rel5_pct_vs_vn30 = ticker_ret5 - vn30_ret5
    rel_week_to_date_pct = ticker_weekly["WeekToDateRetPct"] - index_weekly["WeekToDateRetPct"]
    rel_prev_week_ret_pct = ticker_weekly["PrevWeekRetPct"] - index_weekly["PrevWeekRetPct"]
    ticker_ret_frac = ticker_close.pct_change(fill_method=None)
    index_ret_frac = index_close.pct_change(fill_method=None)
    vn30_ret_frac = vn30_close.pct_change(fill_method=None)
    corr20 = ticker_ret_frac.rolling(20).corr(index_ret_frac)
    beta20 = ticker_ret_frac.rolling(20).cov(index_ret_frac) / index_ret_frac.rolling(20).var()
    corr20_vn30 = ticker_ret_frac.rolling(20).corr(vn30_ret_frac)
    beta20_vn30 = ticker_ret_frac.rolling(20).cov(vn30_ret_frac) / vn30_ret_frac.rolling(20).var()
    rel_rotation_threshold = _adaptive_abs_threshold(rel5_pct, window=60, quantile=0.75, floor=1.0, std_multiplier=1.25)
    ticker_relative_rotation_state = pd.Series(0.0, index=merged.index, dtype=float)
    index_linked_mask = corr20 >= 0.45
    ticker_relative_rotation_state.loc[index_linked_mask & (rel5_pct >= rel_rotation_threshold)] = 1.0
    ticker_relative_rotation_state.loc[index_linked_mask & (rel5_pct <= -rel_rotation_threshold)] = -1.0
    close_in_range = _series_ratio(ticker_close - ticker_low, ticker_high - ticker_low)
    exhaustion_dist_threshold = _adaptive_abs_threshold(ticker_dist_sma20, window=80, quantile=0.8, floor=6.0, std_multiplier=1.0)
    upper_wick_threshold = _adaptive_abs_threshold(ticker_upper_wick_pct, window=60, quantile=0.75, floor=1.5, std_multiplier=1.0)
    lower_wick_threshold = _adaptive_abs_threshold(ticker_lower_wick_pct, window=60, quantile=0.75, floor=1.5, std_multiplier=1.0)
    ticker_exhaustion_state = pd.Series(0.0, index=merged.index, dtype=float)
    ticker_exhaustion_state.loc[
        (ticker_dist_sma20 >= exhaustion_dist_threshold)
        & ((close_in_range <= 0.35) | (ticker_upper_wick_pct >= upper_wick_threshold))
    ] = 1.0
    ticker_exhaustion_state.loc[
        (ticker_dist_sma20 <= -exhaustion_dist_threshold)
        & ((close_in_range >= 0.65) | (ticker_lower_wick_pct >= lower_wick_threshold))
    ] = -1.0

    feature_map = {
        "TickerGapPct": ticker_gap_pct,
        "TickerBodyPct": ticker_body_pct,
        "TickerRangePct": ticker_range_pct,
        "TickerUpperWickPct": ticker_upper_wick_pct,
        "TickerLowerWickPct": ticker_lower_wick_pct,
        "TickerRet1Pct": ticker_ret1,
        "TickerRet5Pct": ticker_ret5,
        "TickerRet20Pct": ticker_ret20,
        "TickerDistSMA20Pct": ticker_dist_sma20,
        "TickerDistSMA50Pct": ticker_dist_sma50,
        "TickerRangePos20": ticker_range_pos20,
        "TickerRangePos60": ticker_range_pos60,
        "TickerVolRatio20": ticker_vol_ratio20,
        "TickerVolatility10": ticker_volatility10,
        "TickerColorStreakState": ticker_color_streak_state,
        "TickerLimitProxyState": ticker_limit_proxy_state,
        "TickerShockState1D": ticker_shock_state_1d,
        "TickerImpulseState3D": ticker_impulse_state_3d,
        "TickerWideRangeState": ticker_wide_range_state,
        "TickerTrendRegimeState": ticker_trend_regime_state,
        "TickerCompressionState": ticker_compression_state,
        "TickerReclaimState": ticker_reclaim_state,
        "TickerRelativeRotationState": ticker_relative_rotation_state,
        "TickerExhaustionState": ticker_exhaustion_state,
        "TickerWeekToDateRetPct": ticker_weekly["WeekToDateRetPct"],
        "TickerWeekToDateRangePct": ticker_weekly["WeekToDateRangePct"],
        "TickerWeekToDateVolumePctPrevWeek": ticker_weekly["WeekToDateVolumePctPrevWeek"],
        "TickerDistPrevWeekHighPct": ticker_weekly["DistPrevWeekHighPct"],
        "TickerDistPrevWeekLowPct": ticker_weekly["DistPrevWeekLowPct"],
        "TickerPrevWeekRetPct": ticker_weekly["PrevWeekRetPct"],
        "TickerPrevWeekRangePct": ticker_weekly["PrevWeekRangePct"],
        "TickerPrevWeekVolRatio4": ticker_weekly["PrevWeekVolRatio4"],
        "Rel5Pct": rel5_pct,
        "Rel20Pct": rel20_pct,
        "Rel5PctVsVN30": rel5_pct_vs_vn30,
        "RelWeekToDateRetPct": rel_week_to_date_pct,
        "RelPrevWeekRetPct": rel_prev_week_ret_pct,
        "Corr20": corr20,
        "Beta20": beta20,
        "IndexGapPct": index_gap_pct,
        "IndexBodyPct": index_body_pct,
        "IndexRangePct": index_range_pct,
        "IndexRet1Pct": index_ret1,
        "IndexRet5Pct": index_ret5,
        "IndexRet20Pct": index_ret20,
        "IndexDistSMA20Pct": index_dist_sma20,
        "IndexDistSMA50Pct": index_dist_sma50,
        "IndexRangePos20": index_range_pos20,
        "IndexRangePos60": index_range_pos60,
        "IndexVolRatio20": index_vol_ratio20,
        "IndexVolatility10": index_volatility10,
        "IndexColorStreakState": index_color_streak_state,
        "IndexWeekToDateRetPct": index_weekly["WeekToDateRetPct"],
        "IndexWeekToDateRangePct": index_weekly["WeekToDateRangePct"],
        "IndexWeekToDateVolumePctPrevWeek": index_weekly["WeekToDateVolumePctPrevWeek"],
        "IndexDistPrevWeekHighPct": index_weekly["DistPrevWeekHighPct"],
        "IndexDistPrevWeekLowPct": index_weekly["DistPrevWeekLowPct"],
        "IndexPrevWeekRetPct": index_weekly["PrevWeekRetPct"],
        "IndexPrevWeekRangePct": index_weekly["PrevWeekRangePct"],
        "IndexPrevWeekVolRatio4": index_weekly["PrevWeekVolRatio4"],
        "VN30Ret1Pct": vn30_ret1,
        "VN30Ret5Pct": vn30_ret5,
        "VN30Ret20Pct": vn30_ret20,
        "VN30DistSMA20Pct": vn30_dist_sma20,
        "VN30RangePos20": vn30_range_pos20,
        "VN30VolRatio20": vn30_vol_ratio20,
        "VN30ColorStreakState": vn30_color_streak_state,
        "Corr20VN30": corr20_vn30,
        "Beta20VN30": beta20_vn30,
    }
    for base_name in (
        "TickerRet1Pct",
        "TickerRangePct",
        "TickerVolRatio20",
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
        "Rel5Pct",
        "Rel5PctVsVN30",
        "IndexRet1Pct",
        "IndexRangePct",
        "IndexColorStreakState",
        "VN30Ret1Pct",
        "VN30ColorStreakState",
    ):
        series = feature_map[base_name]
        for lag in LAGS:
            feature_map[f"{base_name}_Lag{lag}"] = series.shift(lag)

    base_sample = pd.DataFrame(feature_map, index=merged.index)
    base_sample["Ticker"] = ticker
    base_sample["Date"] = merged.index
    base_sample["BaseClose"] = ticker_close

    date_series = pd.Series(merged.index, index=merged.index)
    horizon_frames: List[pd.DataFrame] = []
    for horizon in range(1, max_horizon + 1):
        next_open = ticker_open.shift(-horizon)
        next_high = ticker_high.shift(-horizon)
        next_low = ticker_low.shift(-horizon)
        next_close = ticker_close.shift(-horizon)
        next_anchor_high = pd.concat([next_open, next_close], axis=1).max(axis=1)
        next_anchor_low = pd.concat([next_open, next_close], axis=1).min(axis=1)

        frame = base_sample.copy()
        frame["Horizon"] = horizon
        frame["ForecastWindow"] = f"T+{horizon}"
        frame["ForecastDate"] = date_series.shift(-horizon)
        frame["ActualOpen"] = next_open
        frame["ActualHigh"] = next_high
        frame["ActualLow"] = next_low
        frame["ActualClose"] = next_close
        frame["TargetOpenRetPct"] = ((next_open / ticker_close) - 1.0) * 100.0
        frame["TargetCloseRetPct"] = ((next_close / ticker_close) - 1.0) * 100.0
        frame["TargetUpperWickPct"] = (((next_high - next_anchor_high) / ticker_close) * 100.0).clip(lower=0.0)
        frame["TargetLowerWickPct"] = (((next_anchor_low - next_low) / ticker_close) * 100.0).clip(lower=0.0)
        horizon_frames.append(frame)

    sample = pd.concat(horizon_frames, ignore_index=False)
    sample = sample.replace([np.inf, -np.inf], np.nan)
    sample.index.name = None
    ordered_columns = ["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"] + FEATURE_COLUMNS + [
        "ActualOpen",
        "ActualHigh",
        "ActualLow",
        "ActualClose",
    ] + TARGET_COLUMNS
    return sample[ordered_columns].sort_values(["Date", "Ticker", "Horizon"]).reset_index(drop=True)


def build_multi_ticker_sample(
    tickers: Sequence[str],
    history_dir: Path,
    max_horizon: int = DEFAULT_MAX_HORIZON,
) -> pd.DataFrame:
    frames = [build_ticker_ohlc_sample(ticker, history_dir, max_horizon=max_horizon) for ticker in tickers]
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
        "random_forest": lambda: make_numeric_pipeline(
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            )
        ),
        "hist_gbm": lambda: make_numeric_pipeline(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=250,
                random_state=42,
            )
        ),
    }


def _reconstruct_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base_close = frame["BaseClose"].astype(float)
    open_ret = frame["PredTargetOpenRetPct"].astype(float)
    close_ret = frame["PredTargetCloseRetPct"].astype(float)
    upper_wick = frame["PredTargetUpperWickPct"].astype(float).clip(lower=0.0)
    lower_wick = frame["PredTargetLowerWickPct"].astype(float).clip(lower=0.0)

    pred_open = base_close * (1.0 + (open_ret / 100.0))
    pred_close = base_close * (1.0 + (close_ret / 100.0))
    anchor_high = np.maximum(pred_open, pred_close)
    anchor_low = np.minimum(pred_open, pred_close)
    pred_high = anchor_high + (base_close * upper_wick / 100.0)
    pred_low = anchor_low - (base_close * lower_wick / 100.0)

    return pd.DataFrame(
        {
            "PredOpen": pred_open,
            "PredHigh": pred_high,
            "PredLow": pred_low,
            "PredClose": pred_close,
            "PredOpenRetPct": ((pred_open / base_close) - 1.0) * 100.0,
            "PredHighRetPct": ((pred_high / base_close) - 1.0) * 100.0,
            "PredLowRetPct": ((pred_low / base_close) - 1.0) * 100.0,
            "PredCloseRetPct": ((pred_close / base_close) - 1.0) * 100.0,
            "PredRangePct": ((pred_high - pred_low) / base_close) * 100.0,
        },
        index=frame.index,
    )


def _attach_actual_return_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base_close = out["BaseClose"].astype(float)
    out["ActualOpenRetPct"] = ((out["ActualOpen"] / base_close) - 1.0) * 100.0
    out["ActualHighRetPct"] = ((out["ActualHigh"] / base_close) - 1.0) * 100.0
    out["ActualLowRetPct"] = ((out["ActualLow"] / base_close) - 1.0) * 100.0
    out["ActualCloseRetPct"] = ((out["ActualClose"] / base_close) - 1.0) * 100.0
    out["ActualRangePct"] = ((out["ActualHigh"] - out["ActualLow"]) / base_close) * 100.0
    return out


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


def walk_forward_ohlc_predictions(
    sample_df: pd.DataFrame,
    min_train_dates: int,
    retrain_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_factories = build_model_factories()
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

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
                if not block_dates:
                    continue
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
                    ]
                    + TARGET_COLUMNS
                ].copy()
                out["Model"] = model_name
                for target_column in TARGET_COLUMNS:
                    out[f"Pred{target_column}"] = _fit_predict_target(
                        factory,
                        train_df,
                        block_df,
                        FEATURE_COLUMNS,
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
            forecast["Model"] = model_name
            for target_column in TARGET_COLUMNS:
                forecast[f"Pred{target_column}"] = _fit_predict_target(
                    factory,
                    train_all,
                    current_row,
                    FEATURE_COLUMNS,
                    target_column,
                )
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


def summarise_ohlc_models(prediction_history: pd.DataFrame, current_forecasts: pd.DataFrame) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()

    has_horizon = "Horizon" in prediction_history.columns
    group_columns: Sequence[str] | str = ["Model", "Horizon"] if has_horizon else "Model"
    rows: List[Dict[str, object]] = []
    for group_key, group in prediction_history.groupby(group_columns, sort=False):
        if has_horizon:
            model_name, horizon_value = group_key
            current_group = current_forecasts[
                (current_forecasts["Model"] == model_name) & (current_forecasts["Horizon"] == horizon_value)
            ].copy()
        else:
            model_name = group_key
            horizon_value = None
            current_group = current_forecasts[current_forecasts["Model"] == model_name].copy()

        row: Dict[str, object] = {
            "Model": model_name,
            "EvalRows": int(group.shape[0]),
            "TickerCount": int(group["Ticker"].nunique()),
            "OpenMAEPct": float(mean_absolute_error(group["ActualOpenRetPct"], group["PredOpenRetPct"])),
            "HighMAEPct": float(mean_absolute_error(group["ActualHighRetPct"], group["PredHighRetPct"])),
            "LowMAEPct": float(mean_absolute_error(group["ActualLowRetPct"], group["PredLowRetPct"])),
            "CloseMAEPct": float(mean_absolute_error(group["ActualCloseRetPct"], group["PredCloseRetPct"])),
            "RangeMAEPct": float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"])),
            "MeanOHLCMAEPct": float(
                np.mean(
                    [
                        mean_absolute_error(group["ActualOpenRetPct"], group["PredOpenRetPct"]),
                        mean_absolute_error(group["ActualHighRetPct"], group["PredHighRetPct"]),
                        mean_absolute_error(group["ActualLowRetPct"], group["PredLowRetPct"]),
                        mean_absolute_error(group["ActualCloseRetPct"], group["PredCloseRetPct"]),
                    ]
                )
            ),
            "CloseDirHitPct": float(
                (
                    np.sign(group["ActualCloseRetPct"].astype(float))
                    == np.sign(group["PredCloseRetPct"].astype(float))
                ).mean()
                * 100.0
            ),
            "CurrentTickerCount": int(current_group["Ticker"].nunique()),
            "CurrentAvgPredCloseRetPct": float(current_group["PredCloseRetPct"].mean()) if not current_group.empty else float("nan"),
            "CurrentBullishPct": float((current_group["PredCloseRetPct"] > 0).mean() * 100.0) if not current_group.empty else float("nan"),
        }
        if horizon_value is not None:
            row["Horizon"] = int(horizon_value)
            row["ForecastWindow"] = f"T+{int(horizon_value)}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if has_horizon:
        return summary_df.sort_values(
            ["Horizon", "MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    return summary_df.sort_values(["MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"], ascending=[True, True, False]).reset_index(drop=True)


def choose_best_ohlc_model(summary_df: pd.DataFrame, horizon: int | None = None) -> str:
    if summary_df.empty:
        return ""

    scoped = summary_df.copy()
    if "Horizon" in scoped.columns:
        if horizon is not None:
            scoped = scoped[scoped["Horizon"] == int(horizon)].copy()
            if scoped.empty:
                return ""
            ordered = scoped.sort_values(
                ["MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"],
                ascending=[True, True, False],
            )
            return str(ordered.iloc[0]["Model"])

        aggregated = (
            scoped.groupby("Model", sort=False)[["MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"]]
            .mean()
            .reset_index()
        )
        ordered = aggregated.sort_values(
            ["MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"],
            ascending=[True, True, False],
        )
        return str(ordered.iloc[0]["Model"])

    ordered = scoped.sort_values(
        ["MeanOHLCMAEPct", "CloseMAEPct", "CloseDirHitPct"],
        ascending=[True, True, False],
    )
    return str(ordered.iloc[0]["Model"])


def build_case_studies(
    prediction_history: pd.DataFrame,
    current_forecasts: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()
    ticker_set = {_normalise_ticker(ticker) for ticker in tickers}
    has_horizon = "Horizon" in prediction_history.columns
    group_columns: Sequence[str] | str = ["Ticker", "Model", "Horizon"] if has_horizon else ["Ticker", "Model"]
    rows: List[Dict[str, object]] = []
    for group_key, group in prediction_history.groupby(group_columns, sort=False):
        if has_horizon:
            ticker, model_name, horizon_value = group_key
        else:
            ticker, model_name = group_key
            horizon_value = None
        if ticker not in ticker_set:
            continue

        current_mask = (current_forecasts["Ticker"] == ticker) & (current_forecasts["Model"] == model_name)
        if horizon_value is not None:
            current_mask = current_mask & (current_forecasts["Horizon"] == horizon_value)
        current_row = current_forecasts[current_mask]
        current_payload = current_row.iloc[0].to_dict() if not current_row.empty else {}

        row: Dict[str, object] = {
            "Ticker": ticker,
            "Model": model_name,
            "EvalRows": int(group.shape[0]),
            "MeanOHLCMAEPct": float(
                np.mean(
                    [
                        mean_absolute_error(group["ActualOpenRetPct"], group["PredOpenRetPct"]),
                        mean_absolute_error(group["ActualHighRetPct"], group["PredHighRetPct"]),
                        mean_absolute_error(group["ActualLowRetPct"], group["PredLowRetPct"]),
                        mean_absolute_error(group["ActualCloseRetPct"], group["PredCloseRetPct"]),
                    ]
                )
            ),
            "CloseMAEPct": float(mean_absolute_error(group["ActualCloseRetPct"], group["PredCloseRetPct"])),
            "CloseDirHitPct": float(
                (
                    np.sign(group["ActualCloseRetPct"].astype(float))
                    == np.sign(group["PredCloseRetPct"].astype(float))
                ).mean()
                * 100.0
            ),
            "LastTrainDate": group["Date"].max(),
            "ForecastBaseDate": current_payload.get("Date"),
            "ForecastDate": current_payload.get("ForecastDate"),
            "ForecastOpen": float(current_payload.get("PredOpen", float("nan"))),
            "ForecastHigh": float(current_payload.get("PredHigh", float("nan"))),
            "ForecastLow": float(current_payload.get("PredLow", float("nan"))),
            "ForecastClose": float(current_payload.get("PredClose", float("nan"))),
            "ForecastCloseRetPct": float(current_payload.get("PredCloseRetPct", float("nan"))),
            "ForecastRangePct": float(current_payload.get("PredRangePct", float("nan"))),
            "ForecastCandleBias": current_payload.get("ForecastCandleBias", ""),
        }
        if horizon_value is not None:
            row["Horizon"] = int(horizon_value)
            row["ForecastWindow"] = f"T+{int(horizon_value)}"
        rows.append(row)

    sort_columns = ["Ticker", "Horizon", "MeanOHLCMAEPct", "CloseMAEPct"] if has_horizon else ["Ticker", "MeanOHLCMAEPct", "CloseMAEPct"]
    return pd.DataFrame(rows).sort_values(sort_columns).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay direct multi-horizon OHLC regression models using ticker history plus VNINDEX context."
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Directory containing *_daily.csv files (default: repo out/data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for analysis artifacts (default: repo out/analysis).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_CASE_TICKERS,
        help="Tickers to evaluate. Each ticker is modeled from its own history plus VNINDEX.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=140,
        help="Minimum number of labeled daily rows before walk-forward evaluation starts.",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=10,
        help="Retrain cadence in trading days for walk-forward blocks.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=DEFAULT_MAX_HORIZON,
        help="Direct forecast horizon in trading sessions. T+1..T+N are all trained separately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [_normalise_ticker(ticker) for ticker in args.tickers]
    sample_df = build_multi_ticker_sample(tickers, args.history_dir, max_horizon=int(args.max_horizon))
    prediction_history, current_forecasts = walk_forward_ohlc_predictions(
        sample_df,
        min_train_dates=args.min_train_dates,
        retrain_every=args.retrain_every,
    )
    summary_df = summarise_ohlc_models(prediction_history, current_forecasts)
    best_model = choose_best_ohlc_model(summary_df)
    best_model_by_horizon: Dict[str, str] = {}
    if not summary_df.empty and "Horizon" in summary_df.columns:
        for horizon in sorted(summary_df["Horizon"].dropna().astype(int).unique()):
            best_model_by_horizon[f"T+{int(horizon)}"] = choose_best_ohlc_model(summary_df, horizon=int(horizon))
    case_studies = build_case_studies(prediction_history, current_forecasts, tickers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ohlc_model_summary.csv"
    history_path = args.output_dir / "ohlc_prediction_history.csv"
    current_path = args.output_dir / "ohlc_current_forecasts.csv"
    case_path = args.output_dir / "ohlc_case_studies.csv"
    meta_path = args.output_dir / "ohlc_summary.json"

    summary_df.to_csv(summary_path, index=False)
    prediction_history.to_csv(history_path, index=False)
    current_forecasts.to_csv(current_path, index=False)
    case_studies.to_csv(case_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "BestModelOverall": best_model,
                "BestModelByHorizon": best_model_by_horizon,
                "Tickers": tickers,
                "MaxHorizon": int(args.max_horizon),
                "LatestBaseDate": str(sample_df["Date"].max().date()) if not sample_df.empty else "",
                "HistoryRows": int(prediction_history.shape[0]),
                "CurrentForecastRows": int(current_forecasts.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {current_path}")
    print(f"Wrote {case_path}")
    print(f"Wrote {meta_path}")
    if best_model:
        print(f"Best model overall: {best_model}")
    if best_model_by_horizon:
        for window, model_name in best_model_by_horizon.items():
            print(f"Best model {window}: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
