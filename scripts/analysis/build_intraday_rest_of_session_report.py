from __future__ import annotations

import argparse
from datetime import time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import _normalise_ticker
from scripts.data_fetching.fetch_ticker_data import ensure_intraday_cache
from scripts.data_fetching.vndirect_price_depth import refresh_depth_for_intraday_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
VN_TZ = timezone(timedelta(hours=7))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_HISTORY_CALENDAR_DAYS = 420
DEFAULT_MIN_TRAIN_DATES = 90
DEFAULT_HOLDOUT_DATES = 25
DEFAULT_INTRADAY_HISTORY_DIR = REPO_ROOT / "out" / "data" / "intraday_1m"
DEFAULT_RESOLUTION = "1"
OUTPUT_FILE_NAME = "ml_intraday_rest_of_session.csv"
METRICS_FILE_NAME = "ml_intraday_rest_of_session_metrics.csv"
BACKTEST_FILE_NAME = "ml_intraday_rest_of_session_backtest.csv"
CALIBRATION_PRIOR_ROWS = 4.0
CALIBRATION_PRIOR_PROB_PCT = 50.0

MARKET_OPEN = time(9, 0)
MORNING_BUCKET_START = time(9, 45)
AM_EARLY_END = time(10, 30)
LUNCH_BREAK_START = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
PM_EARLY_END = time(13, 45)
PM_LATE_END = time(14, 25)
MARKET_CLOSE = time(14, 30)
TOTAL_TRADING_MINUTES = 240.0

DEPTH_LEVELS = (1, 2, 3)
DEPTH_COLUMN_ALIASES = {
    "BestBid": ("best_bid", "bid", "bid1", "Bid1", "ValidBid1"),
    "BestAsk": ("best_ask", "ask", "ask1", "offer1", "Ask1", "Offer1", "ValidAsk1"),
    "BidVolume1": ("bid_volume_1", "bid1_volume", "bidVolume1", "bid_vol1", "bidVol1", "BidVolume1", "BidVol1"),
    "BidVolume2": ("bid_volume_2", "bid2_volume", "bidVolume2", "bid_vol2", "bidVol2", "BidVolume2", "BidVol2"),
    "BidVolume3": ("bid_volume_3", "bid3_volume", "bidVolume3", "bid_vol3", "bidVol3", "BidVolume3", "BidVol3"),
    "AskVolume1": ("ask_volume_1", "ask1_volume", "askVolume1", "ask_vol1", "askVol1", "AskVolume1", "AskVol1"),
    "AskVolume2": ("ask_volume_2", "ask2_volume", "askVolume2", "ask_vol2", "askVol2", "AskVolume2", "AskVol2"),
    "AskVolume3": ("ask_volume_3", "ask3_volume", "askVolume3", "ask_vol3", "askVol3", "AskVolume3", "AskVol3"),
}
REQUIRED_DEPTH_COLUMNS = ["BestBid", "BestAsk", "BidVolume1", "AskVolume1"]
DEPTH_FEATURE_COLUMNS = [
    "TickerDepthAvailable",
    "TickerBidAskSpreadPct",
    "TickerTopBidVolume",
    "TickerTopAskVolume",
    "TickerDepthImbalance",
    "TickerDepthImbalanceChange15m",
    "TickerDepthPressurePctADV20",
]

FEATURE_COLUMNS = [
    "BucketCode",
    "TickerSnapshotRetFromPrevClosePct",
    "TickerGapPct",
    "TickerOpenToSnapshotRetPct",
    "TickerLast5mRetPct",
    "TickerLast15mRetPct",
    "TickerLast30mRetPct",
    "TickerLast60mRetPct",
    "TickerRange30mPct",
    "TickerRange60mPct",
    "TickerPosIn30mRange",
    "TickerPosIn60mRange",
    "TickerSessionRangePct",
    "TickerPosInSessionRange",
    "TickerVWAPDeviationPct",
    "TickerSessionVolumePctADV20",
    "TickerCloseToSessionHighPct",
    "TickerCloseToSessionLowPct",
    "TickerMinutesFromOpen",
    "TickerMinutesToClose",
    "TickerSessionProgressPct",
    "TickerAfternoonOpenToSnapshotRetPct",
    "TickerAfternoonVolumePctADV20",
    *DEPTH_FEATURE_COLUMNS,
    "IndexSnapshotRetFromPrevClosePct",
    "IndexGapPct",
    "IndexOpenToSnapshotRetPct",
    "IndexLast5mRetPct",
    "IndexLast15mRetPct",
    "IndexLast30mRetPct",
    "IndexLast60mRetPct",
    "IndexRange30mPct",
    "IndexRange60mPct",
    "IndexPosIn30mRange",
    "IndexPosIn60mRange",
    "IndexSessionRangePct",
    "IndexPosInSessionRange",
    "IndexVWAPDeviationPct",
    "IndexSessionVolumePctADV20",
    "IndexMinutesFromOpen",
    "IndexMinutesToClose",
    "IndexSessionProgressPct",
    "IndexAfternoonOpenToSnapshotRetPct",
    "IndexAfternoonVolumePctADV20",
    "RelSnapshotRetFromPrevClosePct",
    "RelOpenToSnapshotPct",
    "RelLast15mPct",
    "RelLast60mPct",
    "RelVWAPDeviationPct",
    "RelSessionVolumePctADV20",
    "RelAfternoonOpenToSnapshotPct",
]
TARGET_COLUMNS = [
    "TargetLowRetPct",
    "TargetCloseRetPct",
    "TargetHighRetPct",
]
REQUIRED_OUTPUT_COLUMNS = [
    "SnapshotDate",
    "SnapshotTimeBucket",
    "Ticker",
    "PrevClose",
    "Base",
    "SnapshotRetFromPrevClosePct",
    "SnapshotRedFromPrevClose",
    "Low",
    "Mid",
    "High",
    "PredLowRetPct",
    "PredMidRetPct",
    "PredHighRetPct",
    "Model",
    "EvalRows",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
    "PredCloseUpFromSnapshot",
    "PredCloseUpFromSnapshotProbPct",
    "PredCloseUpFromSnapshotCalibrationRows",
    "PredRecoverToPrevClose",
    "PredRecoverToPrevCloseProbPct",
    "PredRecoverToPrevCloseCalibrationRows",
    "PredFinalCloseVsPrevClosePct",
    "RecoverySetup",
    "RecoveryCalibrationConflict",
    "SelectionScore",
]
BACKTEST_OUTPUT_COLUMNS = [
    "SnapshotDate",
    "SnapshotTimeBucket",
    "Ticker",
    "Model",
    "PrevClose",
    "Base",
    "SnapshotRetFromPrevClosePct",
    "SnapshotRedFromPrevClose",
    "TargetCloseVsPrevClosePct",
    "PredFinalCloseVsPrevClosePct",
    "TargetCloseRetPct",
    "PredTargetCloseRetPct",
    "CloseAbsErrPct",
    "CloseDirHit",
    "ActualRecoverToPrevClose",
    "PredRecoverToPrevClose",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
    "SelectionScore",
]
BUCKET_CODES = {
    "AM_EARLY": 1.0,
    "AM_LATE": 2.0,
    "LUNCH_BREAK": 3.0,
    "PM_EARLY": 4.0,
    "PM_LATE": 5.0,
}


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _first_present_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        column = lookup.get(str(alias).lower())
        if column is not None:
            return column
    return None


def _to_vn_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(VN_TZ)
    return ts.tz_convert(VN_TZ)


def classify_snapshot_bucket(engine_run_at: object) -> str | None:
    current_time = _to_vn_timestamp(engine_run_at).timetz().replace(tzinfo=None)
    if MORNING_BUCKET_START <= current_time < AM_EARLY_END:
        return "AM_EARLY"
    if AM_EARLY_END <= current_time < LUNCH_BREAK_START:
        return "AM_LATE"
    if LUNCH_BREAK_START <= current_time < AFTERNOON_OPEN:
        return "LUNCH_BREAK"
    if AFTERNOON_OPEN <= current_time < PM_EARLY_END:
        return "PM_EARLY"
    if PM_EARLY_END <= current_time < PM_LATE_END:
        return "PM_LATE"
    return None


def _current_engine_timestamp(universe_csv: Path) -> pd.Timestamp:
    frame = pd.read_csv(universe_csv)
    _require_columns(frame, ["Ticker", "EngineRunAt"], str(universe_csv))
    if frame.empty:
        raise RuntimeError(f"{universe_csv} is empty")
    return _to_vn_timestamp(frame["EngineRunAt"].dropna().astype(str).iloc[0])


def _load_universe_tickers(universe_csv: Path) -> List[str]:
    frame = pd.read_csv(universe_csv)
    _require_columns(frame, ["Ticker", "EngineRunAt"], str(universe_csv))
    if frame.empty:
        raise RuntimeError(f"{universe_csv} is empty")
    tickers: List[str] = []
    for raw in frame["Ticker"].dropna().tolist():
        ticker = _normalise_ticker(raw)
        if not ticker or ticker == "VNINDEX" or ticker in tickers:
            continue
        tickers.append(ticker)
    if not tickers:
        raise RuntimeError(f"{universe_csv} did not provide any non-index ticker")
    return tickers


def refresh_intraday_cache(
    tickers: Sequence[str],
    history_dir: Path,
    *,
    history_calendar_days: int,
    resolution: str,
) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ["VNINDEX", *tickers]:
        ensure_intraday_cache(
            _normalise_ticker(ticker),
            outdir=str(history_dir),
            min_days=int(history_calendar_days),
            resolution=str(resolution),
        )
    refresh_depth_for_intraday_cache(
        tickers,
        history_dir,
        resolution=str(resolution),
        depth_dir=REPO_ROOT / "out" / "data" / "depth_snapshots",
    )


def _intraday_cache_path(history_dir: Path, ticker: str, resolution: str) -> Path:
    return history_dir / f"{_normalise_ticker(ticker)}_{str(resolution).strip()}m.csv"


def _normalise_depth_columns(raw_frame: pd.DataFrame, parsed_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = parsed_frame.copy()
    for canonical, aliases in DEPTH_COLUMN_ALIASES.items():
        source_column = _first_present_column(raw_frame, aliases)
        if source_column is None:
            enriched[canonical] = np.nan
        else:
            enriched[canonical] = pd.to_numeric(raw_frame[source_column], errors="coerce")
    return enriched


def validate_market_depth_available(frame: pd.DataFrame, label: str) -> None:
    usable_mask = pd.Series([True] * len(frame), index=frame.index)
    missing_or_empty = []
    for column in REQUIRED_DEPTH_COLUMNS:
        if column not in frame.columns or pd.to_numeric(frame[column], errors="coerce").dropna().empty:
            missing_or_empty.append(column)
            usable_mask = pd.Series([False] * len(frame), index=frame.index)
        else:
            usable_mask &= pd.to_numeric(frame[column], errors="coerce").notna()
    if missing_or_empty:
        raise RuntimeError(
            f"{label} intraday cache is missing usable market depth columns: {', '.join(missing_or_empty)}. "
            "The active intraday model requires 1-minute OHLCV plus order-book depth "
            "(best bid/ask and bid/ask volume) and will not produce a standard forecast without it."
        )
    latest_row = frame.sort_values("Timestamp").tail(1)
    if latest_row.empty or not bool(usable_mask.loc[latest_row.index].iloc[0]):
        raise RuntimeError(f"{label} latest intraday row does not have usable market depth.")


def load_intraday_cache_frame(history_dir: Path, ticker: str, resolution: str) -> pd.DataFrame:
    path = _intraday_cache_path(history_dir, ticker, resolution)
    if not path.exists():
        raise FileNotFoundError(f"Missing intraday cache file: {path}")
    raw_frame = pd.read_csv(path)
    _require_columns(raw_frame, ["t", "open", "high", "low", "close", "volume"], str(path))
    ts = pd.to_datetime(pd.to_numeric(raw_frame["t"], errors="coerce"), unit="s", utc=True, errors="coerce")
    frame = pd.DataFrame(
        {
            "Timestamp": ts.dt.tz_convert(VN_TZ),
            "Open": pd.to_numeric(raw_frame["open"], errors="coerce"),
            "High": pd.to_numeric(raw_frame["high"], errors="coerce"),
            "Low": pd.to_numeric(raw_frame["low"], errors="coerce"),
            "Close": pd.to_numeric(raw_frame["close"], errors="coerce"),
            "Volume": pd.to_numeric(raw_frame["volume"], errors="coerce").fillna(0.0),
        }
    )
    frame = _normalise_depth_columns(raw_frame, frame).dropna(subset=["Timestamp", "Close"])
    if frame.empty:
        raise RuntimeError(f"Intraday cache file is empty after parsing: {path}")
    frame = frame.sort_values("Timestamp").reset_index(drop=True)
    frame["TradeDate"] = frame["Timestamp"].dt.tz_convert(VN_TZ).dt.normalize()
    frame["TradeTime"] = frame["Timestamp"].dt.tz_convert(VN_TZ).dt.time
    return frame


def _trailing_window_minutes(frame: pd.DataFrame, minutes_back: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    current_ts = frame["Timestamp"].iloc[-1]
    cutoff = current_ts - pd.Timedelta(minutes=int(minutes_back))
    return frame[frame["Timestamp"] >= cutoff].copy()


def _ret_from_trailing_minutes(frame: pd.DataFrame, minutes_back: int) -> float:
    window = _trailing_window_minutes(frame, minutes_back)
    if window.shape[0] < 2:
        return 0.0
    current = float(window["Close"].iloc[-1])
    anchor = float(window["Close"].iloc[0])
    if anchor == 0:
        return 0.0
    return ((current / anchor) - 1.0) * 100.0


def _safe_ratio_pct(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or float(denominator) == 0.0:
        return 0.0
    return (float(numerator) / float(denominator)) * 100.0


def _safe_return_pct(current: float, anchor: float) -> float:
    if pd.isna(anchor) or float(anchor) == 0.0:
        return 0.0
    return ((float(current) / float(anchor)) - 1.0) * 100.0


def _safe_pos_in_range(current: float, low: float, high: float) -> float:
    width = float(high) - float(low)
    if width == 0.0:
        return 0.5
    return (float(current) - float(low)) / width


def _safe_depth_imbalance(bid_volume: float, ask_volume: float) -> float:
    total = float(bid_volume) + float(ask_volume)
    if pd.isna(total) or total == 0.0:
        return 0.0
    return (float(bid_volume) - float(ask_volume)) / total


def _depth_volume_sum(row: pd.Series, side: str) -> float:
    values = [
        pd.to_numeric(pd.Series([row.get(f"{side}Volume{level}", np.nan)]), errors="coerce").iloc[0]
        for level in DEPTH_LEVELS
    ]
    return float(np.nansum(values))


def _depth_imbalance_for_row(row: pd.Series) -> float:
    return _safe_depth_imbalance(_depth_volume_sum(row, "Bid"), _depth_volume_sum(row, "Ask"))


def _mean_upside_miss_mae(actual_high: pd.Series, predicted_high: pd.Series) -> float:
    return float((actual_high.astype(float) - predicted_high.astype(float)).clip(lower=0.0).mean())


def _mean_downside_miss_mae(actual_low: pd.Series, predicted_low: pd.Series) -> float:
    return float((predicted_low.astype(float) - actual_low.astype(float)).clip(lower=0.0).mean())


def _range_pct_from_window(window: pd.DataFrame, anchor_price: float) -> float:
    if window.empty:
        return 0.0
    window_high = float(window["High"].max())
    window_low = float(window["Low"].min())
    return _safe_ratio_pct(window_high - window_low, anchor_price)


def _pos_in_window_range(window: pd.DataFrame, current_price: float) -> float:
    if window.empty:
        return 0.5
    window_high = float(window["High"].max())
    window_low = float(window["Low"].min())
    return _safe_pos_in_range(current_price, window_low, window_high)


def _trading_minutes_from_open(snapshot_time: time) -> float:
    current_minutes = snapshot_time.hour * 60 + snapshot_time.minute
    open_minutes = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    lunch_minutes = LUNCH_BREAK_START.hour * 60 + LUNCH_BREAK_START.minute
    afternoon_open_minutes = AFTERNOON_OPEN.hour * 60 + AFTERNOON_OPEN.minute
    if current_minutes < lunch_minutes:
        return float(current_minutes - open_minutes)
    if current_minutes < afternoon_open_minutes:
        return float(lunch_minutes - open_minutes)
    return float((lunch_minutes - open_minutes) + (current_minutes - afternoon_open_minutes))


def _is_training_snapshot(snapshot_ts: pd.Timestamp, is_latest_date: bool, position: int, last_position: int) -> bool:
    if is_latest_date:
        return position == last_position
    if snapshot_ts.minute % 15 != 0:
        return False
    return classify_snapshot_bucket(snapshot_ts) is not None


def _prepare_current_snapshot_row(
    current_row: pd.DataFrame,
    *,
    engine_run_at: pd.Timestamp,
    current_bucket: str,
) -> pd.DataFrame:
    if current_row.empty:
        return current_row
    prepared = current_row.copy()
    current_time = engine_run_at.timetz().replace(tzinfo=None)
    minutes_from_open = _trading_minutes_from_open(current_time)
    minutes_to_close = max(0.0, TOTAL_TRADING_MINUTES - minutes_from_open)
    session_progress_pct = (minutes_from_open / TOTAL_TRADING_MINUTES) * 100.0
    prepared["SnapshotTs"] = engine_run_at
    prepared["SnapshotDate"] = engine_run_at.strftime("%Y-%m-%d")
    prepared["SnapshotTimeBucket"] = current_bucket
    prepared["BucketCode"] = float(BUCKET_CODES[current_bucket])
    for prefix in ("Ticker", "Index"):
        prepared[f"{prefix}BucketCode"] = float(BUCKET_CODES[current_bucket])
        prepared[f"{prefix}MinutesFromOpen"] = minutes_from_open
        prepared[f"{prefix}MinutesToClose"] = minutes_to_close
        prepared[f"{prefix}SessionProgressPct"] = session_progress_pct
    return prepared


def summarise_intraday_snapshots(
    frame: pd.DataFrame,
    *,
    prefix: str,
    include_partial_latest: bool,
) -> pd.DataFrame:
    _require_columns(frame, ["Timestamp", "TradeDate", "TradeTime", "Open", "High", "Low", "Close", "Volume"], prefix)
    session_summary = (
        frame.groupby("TradeDate", sort=True)
        .agg(DayClose=("Close", "last"), DayVolume=("Volume", "sum"))
        .reset_index()
        .sort_values("TradeDate")
        .reset_index(drop=True)
    )
    session_summary["PrevClose"] = session_summary["DayClose"].shift(1)
    session_summary["ADV20"] = session_summary["DayVolume"].rolling(20).mean().shift(1)
    daily_lookup = session_summary.set_index("TradeDate")

    rows: List[Dict[str, object]] = []
    latest_trade_date = frame["TradeDate"].max()
    for trade_date, session_rows in frame.groupby("TradeDate", sort=True):
        session_rows = session_rows.sort_values("Timestamp").reset_index(drop=True)
        if trade_date not in daily_lookup.index:
            continue
        prev_close = float(daily_lookup.at[trade_date, "PrevClose"]) if pd.notna(daily_lookup.at[trade_date, "PrevClose"]) else float("nan")
        adv20 = float(daily_lookup.at[trade_date, "ADV20"]) if pd.notna(daily_lookup.at[trade_date, "ADV20"]) else float("nan")
        if pd.isna(prev_close):
            continue

        is_latest_date = trade_date == latest_trade_date
        for position, snapshot in session_rows.iterrows():
            bucket = classify_snapshot_bucket(snapshot["Timestamp"])
            if bucket is None:
                continue
            if is_latest_date and not include_partial_latest:
                continue
            if not _is_training_snapshot(snapshot["Timestamp"], is_latest_date, int(position), int(session_rows.index[-1])):
                continue

            so_far = session_rows.iloc[: position + 1].copy()
            future_rows = session_rows.iloc[position + 1 :].copy()
            if future_rows.empty and not (include_partial_latest and is_latest_date and int(position) == int(session_rows.index[-1])):
                continue

            prices = so_far["Close"].astype(float)
            volumes = pd.to_numeric(so_far["Volume"], errors="coerce").fillna(0.0)
            snapshot_close = float(prices.iloc[-1])
            session_high = float(so_far["High"].max())
            session_low = float(so_far["Low"].min())
            trailing_15m = _trailing_window_minutes(so_far, 15)
            trailing_30m = _trailing_window_minutes(so_far, 30)
            trailing_60m = _trailing_window_minutes(so_far, 60)
            session_volume = float(volumes.sum())
            if session_volume > 0:
                session_vwap = float((prices * volumes).sum() / session_volume)
            else:
                session_vwap = snapshot_close
            top_bid_volume = _depth_volume_sum(snapshot, "Bid")
            top_ask_volume = _depth_volume_sum(snapshot, "Ask")
            depth_available = float(
                not pd.isna(snapshot.get("BestBid", np.nan))
                and not pd.isna(snapshot.get("BestAsk", np.nan))
                and not pd.isna(snapshot.get("BidVolume1", np.nan))
                and not pd.isna(snapshot.get("AskVolume1", np.nan))
            )
            depth_imbalance = _safe_depth_imbalance(top_bid_volume, top_ask_volume)
            if trailing_15m.empty:
                depth_imbalance_change_15m = 0.0
            else:
                depth_imbalance_change_15m = depth_imbalance - _depth_imbalance_for_row(trailing_15m.iloc[0])

            afternoon_rows_so_far = so_far[so_far["TradeTime"] >= AFTERNOON_OPEN]
            if afternoon_rows_so_far.empty:
                afternoon_open = float("nan")
                afternoon_volume = 0.0
            else:
                first_afternoon = afternoon_rows_so_far.iloc[0]
                afternoon_open = float(
                    first_afternoon["Open"] if not pd.isna(first_afternoon["Open"]) else first_afternoon["Close"]
                )
                afternoon_volume = float(pd.to_numeric(afternoon_rows_so_far["Volume"], errors="coerce").fillna(0.0).sum())

            current_time = snapshot["TradeTime"]
            minutes_from_open = _trading_minutes_from_open(current_time)
            minutes_to_close = max(0.0, TOTAL_TRADING_MINUTES - minutes_from_open)
            session_progress_pct = (minutes_from_open / TOTAL_TRADING_MINUTES) * 100.0

            row = {
                "TradeDate": pd.Timestamp(trade_date),
                "SnapshotTs": snapshot["Timestamp"],
                "SnapshotDate": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
                "SnapshotTimeBucket": bucket,
                f"{prefix}BucketCode": float(BUCKET_CODES[bucket]),
                f"{prefix}Open": float(
                    so_far.iloc[0]["Open"] if not pd.isna(so_far.iloc[0]["Open"]) else so_far.iloc[0]["Close"]
                ),
                f"{prefix}PrevClose": prev_close,
                f"{prefix}SnapshotClose": snapshot_close,
                f"{prefix}SessionHigh": session_high,
                f"{prefix}SessionLow": session_low,
                f"{prefix}SessionVWAP": session_vwap,
                f"{prefix}SessionVolume": session_volume,
                f"{prefix}SnapshotRetFromPrevClosePct": _safe_return_pct(snapshot_close, prev_close),
                f"{prefix}GapPct": _safe_return_pct(float(so_far.iloc[0]["Open"]), prev_close),
                f"{prefix}OpenToSnapshotRetPct": _safe_return_pct(snapshot_close, float(so_far.iloc[0]["Open"])),
                f"{prefix}Last5mRetPct": _ret_from_trailing_minutes(so_far, 5),
                f"{prefix}Last15mRetPct": _ret_from_trailing_minutes(so_far, 15),
                f"{prefix}Last30mRetPct": _ret_from_trailing_minutes(so_far, 30),
                f"{prefix}Last60mRetPct": _ret_from_trailing_minutes(so_far, 60),
                f"{prefix}Range30mPct": _range_pct_from_window(trailing_30m, prev_close),
                f"{prefix}Range60mPct": _range_pct_from_window(trailing_60m, prev_close),
                f"{prefix}PosIn30mRange": _pos_in_window_range(trailing_30m, snapshot_close),
                f"{prefix}PosIn60mRange": _pos_in_window_range(trailing_60m, snapshot_close),
                f"{prefix}SessionRangePct": _safe_ratio_pct(session_high - session_low, prev_close),
                f"{prefix}PosInSessionRange": _safe_pos_in_range(snapshot_close, session_low, session_high),
                f"{prefix}VWAPDeviationPct": _safe_return_pct(snapshot_close, session_vwap),
                f"{prefix}SessionVolumePctADV20": _safe_ratio_pct(session_volume, adv20),
                f"{prefix}CloseToSessionHighPct": _safe_return_pct(snapshot_close, session_high),
                f"{prefix}CloseToSessionLowPct": _safe_return_pct(snapshot_close, session_low),
                f"{prefix}MinutesFromOpen": minutes_from_open,
                f"{prefix}MinutesToClose": minutes_to_close,
                f"{prefix}SessionProgressPct": session_progress_pct,
                f"{prefix}AfternoonOpenToSnapshotRetPct": _safe_return_pct(snapshot_close, afternoon_open),
                f"{prefix}AfternoonVolumePctADV20": _safe_ratio_pct(afternoon_volume, adv20),
                f"{prefix}DepthAvailable": depth_available,
                f"{prefix}BidAskSpreadPct": _safe_ratio_pct(
                    float(snapshot.get("BestAsk", np.nan)) - float(snapshot.get("BestBid", np.nan)),
                    snapshot_close,
                ),
                f"{prefix}TopBidVolume": top_bid_volume,
                f"{prefix}TopAskVolume": top_ask_volume,
                f"{prefix}DepthImbalance": depth_imbalance,
                f"{prefix}DepthImbalanceChange15m": depth_imbalance_change_15m,
                f"{prefix}DepthPressurePctADV20": _safe_ratio_pct(top_bid_volume + top_ask_volume, adv20),
            }

            if future_rows.empty:
                row[f"{prefix}TargetCloseRetPct"] = float("nan")
                row[f"{prefix}TargetHighRetPct"] = float("nan")
                row[f"{prefix}TargetLowRetPct"] = float("nan")
                row[f"{prefix}TargetCloseVsPrevClosePct"] = float("nan")
            else:
                day_close = float(session_rows.iloc[-1]["Close"])
                row[f"{prefix}TargetCloseRetPct"] = _safe_return_pct(day_close, snapshot_close)
                row[f"{prefix}TargetHighRetPct"] = _safe_return_pct(float(future_rows["High"].max()), snapshot_close)
                row[f"{prefix}TargetLowRetPct"] = _safe_return_pct(float(future_rows["Low"].min()), snapshot_close)
                row[f"{prefix}TargetCloseVsPrevClosePct"] = _safe_return_pct(day_close, prev_close)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["TradeDate", "SnapshotTs"]).reset_index(drop=True)


def build_intraday_rest_of_session_sample(
    ticker: str,
    history_dir: Path,
    resolution: str,
    *,
    require_depth: bool = False,
) -> pd.DataFrame:
    ticker_frame = load_intraday_cache_frame(history_dir, ticker, resolution)
    index_frame = load_intraday_cache_frame(history_dir, "VNINDEX", resolution)
    if require_depth:
        validate_market_depth_available(ticker_frame, _normalise_ticker(ticker))
    ticker_daily = summarise_intraday_snapshots(ticker_frame, prefix="Ticker", include_partial_latest=True)
    index_daily = summarise_intraday_snapshots(index_frame, prefix="Index", include_partial_latest=True)
    if ticker_daily.empty or index_daily.empty:
        raise RuntimeError(f"Insufficient intraday rest-of-session samples for {ticker}")

    merged = ticker_daily.merge(
        index_daily,
        on=["TradeDate", "SnapshotTs", "SnapshotDate", "SnapshotTimeBucket"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError(f"No overlapping intraday snapshot rows for {ticker} and VNINDEX")

    merged["Ticker"] = _normalise_ticker(ticker)
    merged["PrevClose"] = merged["TickerPrevClose"].astype(float)
    merged["Base"] = merged["TickerSnapshotClose"].astype(float)
    merged["BucketCode"] = merged["TickerBucketCode"].astype(float)
    merged["SnapshotRetFromPrevClosePct"] = merged["TickerSnapshotRetFromPrevClosePct"].astype(float)
    merged["SnapshotRedFromPrevClose"] = merged["SnapshotRetFromPrevClosePct"] < 0.0
    merged["TargetCloseVsPrevClosePct"] = merged["TickerTargetCloseVsPrevClosePct"].astype(float)
    merged["RelSnapshotRetFromPrevClosePct"] = (
        merged["TickerSnapshotRetFromPrevClosePct"] - merged["IndexSnapshotRetFromPrevClosePct"]
    )
    merged["RelOpenToSnapshotPct"] = merged["TickerOpenToSnapshotRetPct"] - merged["IndexOpenToSnapshotRetPct"]
    merged["RelLast15mPct"] = merged["TickerLast15mRetPct"] - merged["IndexLast15mRetPct"]
    merged["RelLast60mPct"] = merged["TickerLast60mRetPct"] - merged["IndexLast60mRetPct"]
    merged["RelVWAPDeviationPct"] = merged["TickerVWAPDeviationPct"] - merged["IndexVWAPDeviationPct"]
    merged["RelSessionVolumePctADV20"] = (
        merged["TickerSessionVolumePctADV20"] - merged["IndexSessionVolumePctADV20"]
    )
    merged["RelAfternoonOpenToSnapshotPct"] = (
        merged["TickerAfternoonOpenToSnapshotRetPct"] - merged["IndexAfternoonOpenToSnapshotRetPct"]
    )

    ordered_columns = [
        "TradeDate",
        "SnapshotTs",
        "SnapshotDate",
        "SnapshotTimeBucket",
        "Ticker",
        "PrevClose",
        "Base",
        "SnapshotRetFromPrevClosePct",
        "SnapshotRedFromPrevClose",
        *FEATURE_COLUMNS,
        "TargetCloseVsPrevClosePct",
        "TargetLowRetPct",
        "TargetCloseRetPct",
        "TargetHighRetPct",
    ]
    sample = merged.rename(
        columns={
            "TickerTargetLowRetPct": "TargetLowRetPct",
            "TickerTargetCloseRetPct": "TargetCloseRetPct",
            "TickerTargetHighRetPct": "TargetHighRetPct",
            "TickerMinutesFromOpen": "TickerMinutesFromOpen",
            "TickerMinutesToClose": "TickerMinutesToClose",
        }
    ).replace([np.inf, -np.inf], np.nan)
    sample[FEATURE_COLUMNS] = sample[FEATURE_COLUMNS].astype(float).fillna(0.0)
    sample[TARGET_COLUMNS] = sample[TARGET_COLUMNS].astype(float)
    return sample[ordered_columns].sort_values(["TradeDate", "SnapshotTs"]).reset_index(drop=True)


def build_multi_ticker_rest_of_session_sample(
    tickers: Sequence[str],
    history_dir: Path,
    resolution: str,
    *,
    require_depth: bool = False,
) -> pd.DataFrame:
    frames = [
        build_intraday_rest_of_session_sample(ticker, history_dir, resolution, require_depth=require_depth)
        for ticker in tickers
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        "ridge": lambda: make_numeric_pipeline(Ridge(alpha=1.0)),
        "random_forest": lambda: make_numeric_pipeline(
            RandomForestRegressor(
                n_estimators=48,
                max_depth=8,
                max_features="sqrt",
                min_samples_leaf=4,
                n_jobs=4,
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
        "mlp_deep": lambda: make_numeric_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=240,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            )
        ),
    }


def generate_intraday_rest_of_session_predictions(
    sample_df: pd.DataFrame,
    *,
    engine_run_at: pd.Timestamp,
    current_bucket: str,
    min_train_dates: int,
    holdout_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_factories = build_live_model_factories()
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

    for ticker, scoped in sample_df.groupby("Ticker", sort=False):
        scoped = scoped.sort_values(["TradeDate", "SnapshotTs"]).reset_index(drop=True)
        latest_trade_date = scoped["TradeDate"].max()
        latest_rows = scoped[scoped["TradeDate"] == latest_trade_date].copy()
        current_row = latest_rows[latest_rows["SnapshotTimeBucket"] == current_bucket].tail(1).copy()
        if current_row.empty:
            current_row = latest_rows.tail(1).copy()
        current_row = _prepare_current_snapshot_row(
            current_row,
            engine_run_at=engine_run_at,
            current_bucket=current_bucket,
        )
        labeled = scoped[scoped["TargetCloseRetPct"].notna()].copy()
        labeled_dates = pd.Series(labeled["TradeDate"].drop_duplicates()).sort_values().reset_index(drop=True)
        if labeled_dates.shape[0] < int(min_train_dates) + int(holdout_dates):
            continue

        holdout_date_values = set(labeled_dates.tail(int(holdout_dates)).tolist())
        train_df = labeled[~labeled["TradeDate"].isin(holdout_date_values)].copy()
        holdout_df = labeled[labeled["TradeDate"].isin(holdout_date_values)].copy()
        if train_df.empty or holdout_df.empty:
            continue

        for model_name, factory in model_factories.items():
            history_frame = holdout_df[
                [
                    "TradeDate",
                    "SnapshotDate",
                    "SnapshotTimeBucket",
                    "Ticker",
                    "PrevClose",
                    "Base",
                    "SnapshotRetFromPrevClosePct",
                    "SnapshotRedFromPrevClose",
                    "TargetCloseVsPrevClosePct",
                    *TARGET_COLUMNS,
                ]
            ].copy()
            history_frame["Model"] = model_name
            for target_column in TARGET_COLUMNS:
                history_frame[f"Pred{target_column}"] = _fit_predict_target(factory, train_df, holdout_df, target_column)
            history_frame["ActualRangePct"] = (
                history_frame["TargetHighRetPct"].astype(float) - history_frame["TargetLowRetPct"].astype(float)
            )
            history_frame["PredRangePct"] = (
                history_frame["PredTargetHighRetPct"].astype(float) - history_frame["PredTargetLowRetPct"].astype(float)
            )
            history_frames.append(history_frame)

            current_frame = current_row[
                [
                    "TradeDate",
                    "SnapshotDate",
                    "SnapshotTimeBucket",
                    "Ticker",
                    "PrevClose",
                    "Base",
                    "SnapshotRetFromPrevClosePct",
                    "SnapshotRedFromPrevClose",
                ]
            ].copy()
            current_frame["Model"] = model_name
            for target_column in TARGET_COLUMNS:
                current_frame[f"Pred{target_column}"] = _fit_predict_target(factory, labeled, current_row, target_column)
            current_frames.append(current_frame)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    return history_df, current_df


def summarise_intraday_rest_of_session_metrics(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    _require_columns(
        history_df,
        [
            "Ticker",
            "SnapshotTimeBucket",
            "Model",
            "TargetCloseRetPct",
            "PredTargetCloseRetPct",
            "TargetHighRetPct",
            "PredTargetHighRetPct",
            "TargetLowRetPct",
            "PredTargetLowRetPct",
            "ActualRangePct",
            "PredRangePct",
        ],
        "Intraday rest-of-session prediction history",
    )
    rows: List[Dict[str, object]] = []
    for (ticker, snapshot_bucket, model_name), group in history_df.groupby(
        ["Ticker", "SnapshotTimeBucket", "Model"],
        sort=False,
    ):
        close_mae = float(mean_absolute_error(group["TargetCloseRetPct"], group["PredTargetCloseRetPct"]))
        range_mae = float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"]))
        upside_miss_mae = _mean_upside_miss_mae(group["TargetHighRetPct"], group["PredTargetHighRetPct"])
        downside_miss_mae = _mean_downside_miss_mae(group["TargetLowRetPct"], group["PredTargetLowRetPct"])
        close_dir = float(
            (
                np.sign(group["TargetCloseRetPct"].astype(float))
                == np.sign(group["PredTargetCloseRetPct"].astype(float))
            ).mean()
            * 100.0
        )
        selection_score = float(
            close_mae
            + (0.35 * range_mae)
            + (0.90 * upside_miss_mae)
            + (0.60 * downside_miss_mae)
            - (0.01 * close_dir)
        )
        rows.append(
            {
                "Ticker": ticker,
                "SnapshotTimeBucket": snapshot_bucket,
                "Model": model_name,
                "EvalRows": int(group.shape[0]),
                "CloseMAEPct": close_mae,
                "RangeMAEPct": range_mae,
                "UpsideMissMAEPct": upside_miss_mae,
                "DownsideMissMAEPct": downside_miss_mae,
                "CloseDirHitPct": close_dir,
                "SelectionScore": selection_score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "Ticker",
            "SnapshotTimeBucket",
            "SelectionScore",
            "UpsideMissMAEPct",
            "DownsideMissMAEPct",
            "CloseMAEPct",
            "RangeMAEPct",
            "CloseDirHitPct",
        ],
        ascending=[True, True, True, True, True, True, True, False],
    ).reset_index(drop=True)


def _mean_bool_pct(values: pd.Series) -> float:
    scoped = values.dropna()
    if scoped.empty:
        return float("nan")
    return float(scoped.astype(bool).mean() * 100.0)


def _conditional_probability(
    frame: pd.DataFrame,
    *,
    predicted_positive: bool,
    predicted_column: str,
    actual_column: str,
) -> tuple[float, int]:
    if frame.empty or predicted_column not in frame.columns or actual_column not in frame.columns:
        return float("nan"), 0
    prediction_values = pd.to_numeric(frame[predicted_column], errors="coerce")
    actual_values = pd.to_numeric(frame[actual_column], errors="coerce")
    comparable = frame[prediction_values.notna() & actual_values.notna()].copy()
    if comparable.empty:
        return float("nan"), 0
    comparable_pred = pd.to_numeric(comparable[predicted_column], errors="coerce")
    if predicted_positive:
        analog = comparable[comparable_pred >= 0.0]
    else:
        analog = comparable[comparable_pred < 0.0]
    if analog.empty:
        analog = comparable
    actual_positive = (pd.to_numeric(analog[actual_column], errors="coerce") >= 0.0).dropna()
    if actual_positive.empty:
        return float("nan"), 0
    positive_count = float(actual_positive.astype(bool).sum())
    row_count = int(actual_positive.shape[0])
    prior_positive = CALIBRATION_PRIOR_ROWS * (CALIBRATION_PRIOR_PROB_PCT / 100.0)
    probability = ((positive_count + prior_positive) / (float(row_count) + CALIBRATION_PRIOR_ROWS)) * 100.0
    return float(probability), row_count


def _recovery_setup(snapshot_red: bool, pred_up: bool, pred_recover: bool) -> str:
    if not snapshot_red:
        return "NOT_RED_SNAPSHOT"
    if pred_recover:
        return "RED_RECOVER_TO_PREV_CLOSE"
    if pred_up:
        return "RED_BOUNCE_FROM_SNAPSHOT"
    return "RED_RECOVERY_NOT_CONFIRMED"


def select_current_intraday_rest_of_session_forecasts(history_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = summarise_intraday_rest_of_session_metrics(history_df)
    if metrics_df.empty or current_df.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    _require_columns(
        current_df,
        [
            "SnapshotDate",
            "SnapshotTimeBucket",
            "Ticker",
            "PrevClose",
            "Base",
            "SnapshotRetFromPrevClosePct",
            "SnapshotRedFromPrevClose",
            "Model",
            "PredTargetLowRetPct",
            "PredTargetCloseRetPct",
            "PredTargetHighRetPct",
        ],
        "Intraday current forecasts",
    )
    best_metrics = metrics_df.drop_duplicates(subset=["Ticker", "SnapshotTimeBucket"], keep="first")
    merged = best_metrics.merge(
        current_df,
        on=["Ticker", "SnapshotTimeBucket", "Model"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    low = merged["Base"].astype(float) * (1.0 + (merged["PredTargetLowRetPct"].astype(float) / 100.0))
    mid = merged["Base"].astype(float) * (1.0 + (merged["PredTargetCloseRetPct"].astype(float) / 100.0))
    high = merged["Base"].astype(float) * (1.0 + (merged["PredTargetHighRetPct"].astype(float) / 100.0))
    prev_close = merged["PrevClose"].astype(float)
    pred_final_close_vs_prev_close_pct = ((mid / prev_close) - 1.0) * 100.0
    pred_close_up_from_snapshot = merged["PredTargetCloseRetPct"].astype(float) >= 0.0
    pred_recover_to_prev_close = pred_final_close_vs_prev_close_pct >= 0.0

    close_up_probabilities: List[float] = []
    close_up_calibration_rows: List[int] = []
    recover_probabilities: List[float] = []
    recover_calibration_rows: List[int] = []
    recovery_setups: List[str] = []
    recovery_calibration_conflicts: List[bool] = []
    for _, row in merged.iterrows():
        scoped_history = history_df[
            (history_df["Ticker"] == row["Ticker"])
            & (history_df["SnapshotTimeBucket"] == row["SnapshotTimeBucket"])
            & (history_df["Model"] == row["Model"])
        ].copy()
        if "PrevClose" in scoped_history.columns:
            scoped_history["PredFinalCloseVsPrevClosePct"] = (
                (
                    scoped_history["Base"].astype(float)
                    * (1.0 + (scoped_history["PredTargetCloseRetPct"].astype(float) / 100.0))
                )
                / scoped_history["PrevClose"].astype(float)
                - 1.0
            ) * 100.0

        current_pred_up = float(row["PredTargetCloseRetPct"]) >= 0.0
        current_snapshot_red = bool(row["SnapshotRedFromPrevClose"])
        current_pred_recover = (
            (
                float(row["Base"]) * (1.0 + (float(row["PredTargetCloseRetPct"]) / 100.0))
            )
            / float(row["PrevClose"])
            - 1.0
        ) >= 0.0
        close_up_probability, close_up_rows = _conditional_probability(
            scoped_history,
            predicted_positive=current_pred_up,
            predicted_column="PredTargetCloseRetPct",
            actual_column="TargetCloseRetPct",
        )
        close_up_probabilities.append(close_up_probability)
        close_up_calibration_rows.append(close_up_rows)

        recover_history = scoped_history
        if current_snapshot_red and "SnapshotRedFromPrevClose" in scoped_history.columns:
            recover_history = scoped_history[scoped_history["SnapshotRedFromPrevClose"].astype(bool)].copy()
        recover_probability, recover_rows = _conditional_probability(
            recover_history,
            predicted_positive=current_pred_recover,
            predicted_column="PredFinalCloseVsPrevClosePct",
            actual_column="TargetCloseVsPrevClosePct",
        )
        recover_probabilities.append(recover_probability)
        recover_calibration_rows.append(recover_rows)
        recovery_setups.append(
            _recovery_setup(
                snapshot_red=current_snapshot_red,
                pred_up=current_pred_up,
                pred_recover=current_pred_recover,
            )
        )
        recovery_calibration_conflicts.append(
            (not pd.isna(close_up_probability) and current_pred_up != (close_up_probability >= 50.0))
            or (not pd.isna(recover_probability) and current_pred_recover != (recover_probability >= 50.0))
        )

    report_df = pd.DataFrame(
        {
            "SnapshotDate": merged["SnapshotDate"],
            "SnapshotTimeBucket": merged["SnapshotTimeBucket"],
            "Ticker": merged["Ticker"],
            "PrevClose": prev_close,
            "Base": merged["Base"].astype(float),
            "SnapshotRetFromPrevClosePct": merged["SnapshotRetFromPrevClosePct"].astype(float),
            "SnapshotRedFromPrevClose": merged["SnapshotRedFromPrevClose"].astype(bool),
            "Low": low.astype(float),
            "Mid": mid.astype(float),
            "High": high.astype(float),
            "PredLowRetPct": merged["PredTargetLowRetPct"].astype(float),
            "PredMidRetPct": merged["PredTargetCloseRetPct"].astype(float),
            "PredHighRetPct": merged["PredTargetHighRetPct"].astype(float),
            "Model": merged["Model"],
            "EvalRows": merged["EvalRows"].astype(int),
            "CloseMAEPct": merged["CloseMAEPct"].astype(float),
            "RangeMAEPct": merged["RangeMAEPct"].astype(float),
            "CloseDirHitPct": merged["CloseDirHitPct"].astype(float),
            "PredCloseUpFromSnapshot": pred_close_up_from_snapshot.astype(bool),
            "PredCloseUpFromSnapshotProbPct": close_up_probabilities,
            "PredCloseUpFromSnapshotCalibrationRows": close_up_calibration_rows,
            "PredRecoverToPrevClose": pred_recover_to_prev_close.astype(bool),
            "PredRecoverToPrevCloseProbPct": recover_probabilities,
            "PredRecoverToPrevCloseCalibrationRows": recover_calibration_rows,
            "PredFinalCloseVsPrevClosePct": pred_final_close_vs_prev_close_pct.astype(float),
            "RecoverySetup": recovery_setups,
            "RecoveryCalibrationConflict": recovery_calibration_conflicts,
            "SelectionScore": merged["SelectionScore"].astype(float),
        }
    )
    _require_columns(report_df, REQUIRED_OUTPUT_COLUMNS, "Intraday rest-of-session report")
    return report_df.sort_values(["Ticker"]).reset_index(drop=True)


def select_intraday_rest_of_session_backtest(history_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = summarise_intraday_rest_of_session_metrics(history_df)
    if metrics_df.empty or history_df.empty:
        return pd.DataFrame(columns=BACKTEST_OUTPUT_COLUMNS)

    best_metrics = metrics_df.drop_duplicates(subset=["Ticker", "SnapshotTimeBucket"], keep="first")
    merged = history_df.merge(
        best_metrics[
            [
                "Ticker",
                "SnapshotTimeBucket",
                "Model",
                "CloseMAEPct",
                "RangeMAEPct",
                "CloseDirHitPct",
                "SelectionScore",
            ]
        ],
        on=["Ticker", "SnapshotTimeBucket", "Model"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=BACKTEST_OUTPUT_COLUMNS)

    pred_final_close_vs_prev_close_pct = (
        (
            merged["Base"].astype(float)
            * (1.0 + (merged["PredTargetCloseRetPct"].astype(float) / 100.0))
        )
        / merged["PrevClose"].astype(float)
        - 1.0
    ) * 100.0
    close_dir_hit = (
        np.sign(merged["TargetCloseRetPct"].astype(float))
        == np.sign(merged["PredTargetCloseRetPct"].astype(float))
    )
    report_df = pd.DataFrame(
        {
            "SnapshotDate": merged["SnapshotDate"],
            "SnapshotTimeBucket": merged["SnapshotTimeBucket"],
            "Ticker": merged["Ticker"],
            "Model": merged["Model"],
            "PrevClose": merged["PrevClose"].astype(float),
            "Base": merged["Base"].astype(float),
            "SnapshotRetFromPrevClosePct": merged["SnapshotRetFromPrevClosePct"].astype(float),
            "SnapshotRedFromPrevClose": merged["SnapshotRedFromPrevClose"].astype(bool),
            "TargetCloseVsPrevClosePct": merged["TargetCloseVsPrevClosePct"].astype(float),
            "PredFinalCloseVsPrevClosePct": pred_final_close_vs_prev_close_pct.astype(float),
            "TargetCloseRetPct": merged["TargetCloseRetPct"].astype(float),
            "PredTargetCloseRetPct": merged["PredTargetCloseRetPct"].astype(float),
            "CloseAbsErrPct": (
                merged["TargetCloseRetPct"].astype(float)
                - merged["PredTargetCloseRetPct"].astype(float)
            ).abs(),
            "CloseDirHit": close_dir_hit.astype(bool),
            "ActualRecoverToPrevClose": merged["TargetCloseVsPrevClosePct"].astype(float) >= 0.0,
            "PredRecoverToPrevClose": pred_final_close_vs_prev_close_pct.astype(float) >= 0.0,
            "CloseMAEPct": merged["CloseMAEPct"].astype(float),
            "RangeMAEPct": merged["RangeMAEPct"].astype(float),
            "CloseDirHitPct": merged["CloseDirHitPct"].astype(float),
            "SelectionScore": merged["SelectionScore"].astype(float),
        }
    )
    _require_columns(report_df, BACKTEST_OUTPUT_COLUMNS, "Intraday rest-of-session backtest")
    return report_df.sort_values(["SnapshotDate", "SnapshotTimeBucket", "Ticker"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an optional intraday forecast file for the remainder of the current session."
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=REPO_ROOT / "out" / "universe.csv",
        help="Universe CSV used to resolve the working ticker set and current EngineRunAt.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_INTRADAY_HISTORY_DIR,
        help="Directory containing cached 1-minute intraday OHLCV plus market-depth files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the optional intraday live artifact.",
    )
    parser.add_argument(
        "--history-calendar-days",
        type=int,
        default=DEFAULT_HISTORY_CALENDAR_DAYS,
        help="Calendar-day lookback for the cached intraday history.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=DEFAULT_MIN_TRAIN_DATES,
        help="Minimum historical labeled intraday snapshots before the ticker-specific model may train.",
    )
    parser.add_argument(
        "--holdout-dates",
        type=int,
        default=DEFAULT_HOLDOUT_DATES,
        help="Number of latest labeled intraday snapshots used as holdout when ranking models.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION,
        help="Intraday resolution used for the live model (default: 1-minute bars).",
    )
    parser.add_argument(
        "--allow-missing-depth",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch: allow intraday modeling without order-book depth. "
            "The standard active intraday forecast requires market depth."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output_dir / OUTPUT_FILE_NAME
    metrics_path = args.output_dir / METRICS_FILE_NAME
    backtest_path = args.output_dir / BACKTEST_FILE_NAME
    engine_run_at = _current_engine_timestamp(args.universe_csv)
    current_bucket = classify_snapshot_bucket(engine_run_at)
    if current_bucket is None:
        output_path.unlink(missing_ok=True)
        print(
            f"Skipped {OUTPUT_FILE_NAME}: EngineRunAt={engine_run_at.isoformat()} is outside the supported intraday windows."
        )
        return 0

    tickers = _load_universe_tickers(args.universe_csv)
    refresh_intraday_cache(
        tickers,
        args.history_dir,
        history_calendar_days=int(args.history_calendar_days),
        resolution=str(args.resolution),
    )
    sample_df = build_multi_ticker_rest_of_session_sample(
        tickers,
        args.history_dir,
        str(args.resolution),
        require_depth=not bool(args.allow_missing_depth),
    )
    history_df, current_df = generate_intraday_rest_of_session_predictions(
        sample_df,
        engine_run_at=engine_run_at,
        current_bucket=current_bucket,
        min_train_dates=int(args.min_train_dates),
        holdout_dates=int(args.holdout_dates),
    )
    metrics_df = summarise_intraday_rest_of_session_metrics(history_df)
    backtest_df = select_intraday_rest_of_session_backtest(history_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_df.empty:
        metrics_path.unlink(missing_ok=True)
    else:
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote {metrics_path}")
    if backtest_df.empty:
        backtest_path.unlink(missing_ok=True)
    else:
        backtest_df.to_csv(backtest_path, index=False)
        print(f"Wrote {backtest_path}")

    report_df = select_current_intraday_rest_of_session_forecasts(history_df, current_df)
    if report_df.empty:
        output_path.unlink(missing_ok=True)
        print(
            "Skipped "
            f"{OUTPUT_FILE_NAME}: the current universe did not yield enough historical intraday samples."
        )
        return 0

    report_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
