from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.data_fetching.cafef_flows import CafeFFlowCache, ensure_foreign_flow_df, ensure_proprietary_flow_df, summarize_flow_metrics
from scripts.data_fetching.fetch_ticker_data import ensure_intraday_cache, ensure_ohlc_cache
from scripts.data_fetching.macro_factor_cache import load_macro_factor_matrix, refresh_macro_factor_cache
from scripts.data_fetching.vietstock_bctt_api import (
    VietstockBcttCache,
    build_quarterly_feature_frame_from_tables,
    load_or_fetch_bctt_feature_frame,
)
from scripts.data_fetching.vietstock_overview_api import build_fundamental_frame
from scripts.data_fetching.vndirect_price_depth import refresh_depth_for_intraday_cache


DEFAULT_CONFIG = Path("config/data_hub.yaml")
DEFAULT_OUTPUT_DIR = Path("data-hub/latest")
VN_TZ = "Asia/Ho_Chi_Minh"
INDEX_TICKERS = {"VNINDEX", "VN30", "VN100"}


API_CATALOG = [
    {
        "Source": "VNDIRECT dchart",
        "Kind": "ohlcv",
        "Endpoint": "https://dchart-api.vndirect.com.vn/dchart/history",
        "NumericData": "daily/intraday open, high, low, close, volume",
        "DefaultOutput": "daily/*.csv, intraday/*.csv, latest_metrics.csv",
        "News": "no",
    },
    {
        "Source": "VNDIRECT priceboard",
        "Kind": "order_book_depth",
        "Endpoint": "https://price-streaming-api-free.vndirect.com.vn/v2/stocks/snapshot",
        "NumericData": "10-level bid/ask depth, match/current price, accumulated volume/value, foreign buy/sell/room, floor/ceiling",
        "DefaultOutput": "depth/latest_depth.csv, latest_metrics.csv",
        "News": "no",
    },
    {
        "Source": "Vietstock overview",
        "Kind": "fundamental_snapshot",
        "Endpoint": "https://finance.vietstock.vn/{ticker}-ctcp.htm",
        "NumericData": "forward PE, PB, derived ROE",
        "DefaultOutput": "fundamentals/vietstock_overview.csv, latest_metrics.csv",
        "News": "no",
    },
    {
        "Source": "Vietstock BCTT",
        "Kind": "financial_statement_cache",
        "Endpoint": "finance.vietstock.vn BCTT tables via cached Playwright collector",
        "NumericData": "EPS4Q, BVPS, PB, margins, ROE/ROA, debt ratios, revenue/profit growth",
        "DefaultOutput": "available through BCTT cache/features when collected",
        "News": "no",
    },
    {
        "Source": "CafeF flows",
        "Kind": "foreign_proprietary_flow",
        "Endpoint": "CafeF Ajax GDKhoiNgoai/GDTuDoanh",
        "NumericData": "foreign/proprietary net shares and net value over 1/5/20 sessions",
        "DefaultOutput": "flows/cafef_flows.csv, latest_metrics.csv",
        "News": "no",
    },
    {
        "Source": "FRED/Stooq macro cache",
        "Kind": "macro_market_numeric",
        "Endpoint": "configured in config/macro_factors.yaml",
        "NumericData": "oil, gold, USD, VIX, US yields, global equity index closes",
        "DefaultOutput": "macro/latest_macro.csv, macro/macro_matrix_tail.csv",
        "News": "no",
    },
    {
        "Source": "Vietstock/VNDIRECT/Investing components",
        "Kind": "market_membership",
        "Endpoint": "board/component pages and VNDIRECT stocks API",
        "NumericData": "membership flags for VN30/VN100/HOSE universe",
        "DefaultOutput": "universe.csv when collected upstream",
        "News": "no",
    },
]


CALCULATION_CATALOG = [
    {
        "Group": "trend_momentum",
        "Columns": "returns, SMA/EMA distance, RSI, 52-week position",
        "Inputs": "daily OHLCV",
        "DefaultOutput": "daily/*.csv, latest_metrics.csv",
    },
    {
        "Group": "risk_volatility",
        "Columns": "ATR, realized volatility, downside volatility, drawdown, close location",
        "Inputs": "daily OHLCV",
        "DefaultOutput": "daily/*.csv, latest_metrics.csv",
    },
    {
        "Group": "liquidity",
        "Columns": "traded value, ADV, average value, volume/value ratios",
        "Inputs": "daily/intraday OHLCV",
        "DefaultOutput": "daily/*.csv, latest_metrics.csv",
    },
    {
        "Group": "relative_strength",
        "Columns": "excess return versus VNINDEX/VN30, rolling beta/correlation",
        "Inputs": "ticker daily returns plus index daily returns",
        "DefaultOutput": "daily/*.csv, latest_metrics.csv",
    },
    {
        "Group": "market_breadth",
        "Columns": "advancers/decliners, up-value share, above moving-average share, new highs/lows",
        "Inputs": "configured ticker daily metrics",
        "DefaultOutput": "market/breadth_daily.csv",
    },
    {
        "Group": "intraday_microstructure",
        "Columns": "minute return, cumulative VWAP, volume rate, latest-day profile",
        "Inputs": "1-minute OHLCV plus optional depth snapshot",
        "DefaultOutput": "intraday/*.csv, intraday/minute_profile/*.csv, intraday/summary_by_ticker.csv",
    },
    {
        "Group": "cross_section",
        "Columns": "latest return/liquidity/volume-spike ranks across the configured universe",
        "Inputs": "latest ticker metrics",
        "DefaultOutput": "market/cross_section_latest.csv",
    },
]


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _is_index_ticker(ticker: str) -> bool:
    return _normalise_ticker(ticker) in INDEX_TICKERS


def load_config(path: Path) -> Dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("tickers", ["VIC", "VNINDEX", "VN30"])
    raw.setdefault("paths", {})
    raw.setdefault("refresh", {})
    raw.setdefault("windows", {})
    return raw


def _path(config: Mapping[str, object], name: str, default: str) -> Path:
    return Path(((config.get("paths") or {}) if isinstance(config.get("paths"), dict) else {}).get(name, default))


def _window(config: Mapping[str, object], name: str, default: int) -> int:
    windows = (config.get("windows") or {}) if isinstance(config.get("windows"), dict) else {}
    return int(windows.get(name, default))


def _refresh_enabled(config: Mapping[str, object], name: str, refresh_requested: bool) -> bool:
    if not refresh_requested:
        return False
    refresh = (config.get("refresh") or {}) if isinstance(config.get("refresh"), dict) else {}
    return bool(refresh.get(name, False))


def _source_status(
    source: str,
    *,
    attempted: bool,
    status: str,
    ticker_count: int = 0,
    row_count: int = 0,
    detail: str = "",
    output: str = "",
) -> Dict[str, object]:
    return {
        "Source": source,
        "Attempted": bool(attempted),
        "Status": status,
        "TickerCount": int(ticker_count),
        "RowCount": int(row_count),
        "Detail": detail,
        "Output": output,
    }


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_daily(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["Ticker"] = ticker
    out["Date"] = pd.to_datetime(out.get("date_vn"), errors="coerce")
    if out["Date"].isna().all() and "t" in out.columns:
        out["Date"] = pd.to_datetime(pd.to_numeric(out["t"], errors="coerce"), unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ).dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    return out.dropna(subset=["Date", "close"]).sort_values("Date").reset_index(drop=True)


def enrich_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy().sort_values("Date").reset_index(drop=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    prev_close = close.shift(1)
    out["ret_1d_pct"] = close.pct_change(1, fill_method=None) * 100.0
    out["ret_5d_pct"] = close.pct_change(5, fill_method=None) * 100.0
    out["ret_20d_pct"] = close.pct_change(20, fill_method=None) * 100.0
    out["ret_60d_pct"] = close.pct_change(60, fill_method=None) * 100.0
    out["ret_120d_pct"] = close.pct_change(120, fill_method=None) * 100.0
    out["ret_252d_pct"] = close.pct_change(252, fill_method=None) * 100.0
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_200"] = close.rolling(200).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["dist_sma20_pct"] = ((close / out["sma_20"]) - 1.0) * 100.0
    out["dist_sma50_pct"] = ((close / out["sma_50"]) - 1.0) * 100.0
    out["dist_sma200_pct"] = ((close / out["sma_200"]) - 1.0) * 100.0
    out["dist_ema20_pct"] = ((close / out["ema_20"]) - 1.0) * 100.0
    out["range_pct"] = ((high - low) / prev_close.replace(0.0, np.nan)) * 100.0
    out["body_pct"] = ((close - out["open"].astype(float)) / prev_close.replace(0.0, np.nan)) * 100.0
    out["gap_pct"] = ((out["open"].astype(float) / prev_close.replace(0.0, np.nan)) - 1.0) * 100.0
    out["close_location_pct"] = ((close - low) / (high - low).replace(0.0, np.nan)) * 100.0
    out["traded_value"] = close * out["volume"].astype(float)
    out["adv20_shares"] = out["volume"].astype(float).rolling(20).mean()
    out["adv60_shares"] = out["volume"].astype(float).rolling(60).mean()
    out["avg_value_20"] = out["traded_value"].rolling(20).mean()
    out["avg_value_60"] = out["traded_value"].rolling(60).mean()
    out["volume_ratio_20"] = out["volume"].astype(float) / out["adv20_shares"].replace(0.0, np.nan)
    out["value_ratio_20"] = out["traded_value"] / out["avg_value_20"].replace(0.0, np.nan)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr14_pct"] = (out["atr14"] / close.replace(0.0, np.nan)) * 100.0
    daily_ret = close.pct_change(1, fill_method=None)
    out["realized_vol_20d_pct"] = daily_ret.rolling(20).std() * np.sqrt(252.0) * 100.0
    out["realized_vol_60d_pct"] = daily_ret.rolling(60).std() * np.sqrt(252.0) * 100.0
    out["downside_vol_20d_pct"] = daily_ret.clip(upper=0.0).rolling(20).std() * np.sqrt(252.0) * 100.0
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    rolling_high = high.rolling(252, min_periods=20).max()
    rolling_low = low.rolling(252, min_periods=20).min()
    out["high_52w"] = rolling_high
    out["low_52w"] = rolling_low
    out["pos_52w_pct"] = ((close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)) * 100.0
    out["drawdown_60d_pct"] = ((close / close.rolling(60, min_periods=20).max()) - 1.0) * 100.0
    out["drawdown_252d_pct"] = ((close / close.rolling(252, min_periods=20).max()) - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def latest_daily_metrics(frame: pd.DataFrame) -> Dict[str, object]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {
        "LatestDate": row.get("Date").date().isoformat() if pd.notna(row.get("Date")) else "",
        "LastClose": row.get("close"),
        "DailyOpen": row.get("open"),
        "DailyHigh": row.get("high"),
        "DailyLow": row.get("low"),
        "DailyVolume": row.get("volume"),
        "Ret1dPct": row.get("ret_1d_pct"),
        "Ret5dPct": row.get("ret_5d_pct"),
        "Ret20dPct": row.get("ret_20d_pct"),
        "Ret60dPct": row.get("ret_60d_pct"),
        "Ret120dPct": row.get("ret_120d_pct"),
        "Ret252dPct": row.get("ret_252d_pct"),
        "SMA20": row.get("sma_20"),
        "SMA50": row.get("sma_50"),
        "SMA200": row.get("sma_200"),
        "EMA20": row.get("ema_20"),
        "EMA50": row.get("ema_50"),
        "DistSMA20Pct": row.get("dist_sma20_pct"),
        "DistSMA50Pct": row.get("dist_sma50_pct"),
        "DistSMA200Pct": row.get("dist_sma200_pct"),
        "DistEMA20Pct": row.get("dist_ema20_pct"),
        "RSI14": row.get("rsi14"),
        "ATR14": row.get("atr14"),
        "ATR14Pct": row.get("atr14_pct"),
        "RealizedVol20dPct": row.get("realized_vol_20d_pct"),
        "RealizedVol60dPct": row.get("realized_vol_60d_pct"),
        "DownsideVol20dPct": row.get("downside_vol_20d_pct"),
        "Drawdown60dPct": row.get("drawdown_60d_pct"),
        "Drawdown252dPct": row.get("drawdown_252d_pct"),
        "DailyGapPct": row.get("gap_pct"),
        "DailyCloseLocationPct": row.get("close_location_pct"),
        "DailyTradedValue": row.get("traded_value"),
        "ADV20Shares": row.get("adv20_shares"),
        "ADV60Shares": row.get("adv60_shares"),
        "AvgValue20": row.get("avg_value_20"),
        "AvgValue60": row.get("avg_value_60"),
        "VolumeRatio20": row.get("volume_ratio_20"),
        "ValueRatio20": row.get("value_ratio_20"),
        "High52w": row.get("high_52w"),
        "Low52w": row.get("low_52w"),
        "Pos52wPct": row.get("pos_52w_pct"),
        "ExcessRet20dVsVNINDEXPct": row.get("excess_ret_20d_vs_vnindex_pct"),
        "ExcessRet60dVsVNINDEXPct": row.get("excess_ret_60d_vs_vnindex_pct"),
        "Beta60dVsVNINDEX": row.get("beta_60d_vs_vnindex"),
        "Corr60dVsVNINDEX": row.get("corr_60d_vs_vnindex"),
        "ExcessRet20dVsVN30Pct": row.get("excess_ret_20d_vs_vn30_pct"),
        "ExcessRet60dVsVN30Pct": row.get("excess_ret_60d_vs_vn30_pct"),
        "Beta60dVsVN30": row.get("beta_60d_vs_vn30"),
        "Corr60dVsVN30": row.get("corr_60d_vs_vn30"),
    }


def add_market_relative_metrics(daily_frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    enriched = {ticker: frame.copy() for ticker, frame in daily_frames.items()}
    for index_ticker in ("VNINDEX", "VN30"):
        index_frame = enriched.get(index_ticker)
        if index_frame is None or index_frame.empty or "ret_1d_pct" not in index_frame.columns:
            continue
        suffix = index_ticker.lower()
        index_returns = index_frame[["Date", "ret_1d_pct", "ret_20d_pct", "ret_60d_pct"]].rename(
            columns={
                "ret_1d_pct": f"{suffix}_ret_1d_pct",
                "ret_20d_pct": f"{suffix}_ret_20d_pct",
                "ret_60d_pct": f"{suffix}_ret_60d_pct",
            }
        )
        for ticker, frame in list(enriched.items()):
            if frame.empty:
                continue
            merged = frame.merge(index_returns, on="Date", how="left")
            merged[f"excess_ret_20d_vs_{suffix}_pct"] = merged["ret_20d_pct"] - merged[f"{suffix}_ret_20d_pct"]
            merged[f"excess_ret_60d_vs_{suffix}_pct"] = merged["ret_60d_pct"] - merged[f"{suffix}_ret_60d_pct"]
            ticker_ret = merged["ret_1d_pct"] / 100.0
            index_ret = merged[f"{suffix}_ret_1d_pct"] / 100.0
            rolling_cov = ticker_ret.rolling(60).cov(index_ret)
            rolling_var = index_ret.rolling(60).var()
            merged[f"beta_60d_vs_{suffix}"] = rolling_cov / rolling_var.replace(0.0, np.nan)
            merged[f"corr_60d_vs_{suffix}"] = ticker_ret.rolling(60).corr(index_ret)
            enriched[ticker] = merged.drop(columns=[f"{suffix}_ret_1d_pct", f"{suffix}_ret_20d_pct", f"{suffix}_ret_60d_pct"])
    return enriched


def _read_intraday(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty or "t" not in raw.columns:
        return pd.DataFrame()
    out = raw.copy()
    out["Ticker"] = ticker
    out["Timestamp"] = pd.to_datetime(pd.to_numeric(out["t"], errors="coerce"), unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    out["TradeDate"] = out["Timestamp"].dt.date.astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    return out.dropna(subset=["Timestamp", "close"]).sort_values("Timestamp").reset_index(drop=True)


def enrich_intraday_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy().sort_values("Timestamp").reset_index(drop=True)
    out["MinuteOfDay"] = out["Timestamp"].dt.hour * 60 + out["Timestamp"].dt.minute
    out["MinuteIndexInDay"] = out.groupby("TradeDate").cumcount() + 1
    out["ret_1m_pct"] = out.groupby("TradeDate")["close"].pct_change(1, fill_method=None) * 100.0
    out["volume_per_minute"] = out["volume"].fillna(0.0).astype(float)
    out["traded_value_1m"] = out["close"].astype(float) * out["volume_per_minute"]
    out["cum_volume"] = out.groupby("TradeDate")["volume_per_minute"].cumsum()
    out["cum_traded_value"] = out.groupby("TradeDate")["traded_value_1m"].cumsum()
    out["cum_vwap"] = out["cum_traded_value"] / out["cum_volume"].replace(0.0, np.nan)
    out["close_vs_cum_vwap_pct"] = ((out["close"].astype(float) / out["cum_vwap"]) - 1.0) * 100.0
    day_open = out.groupby("TradeDate")["open"].transform("first").astype(float)
    out["ret_from_day_open_pct"] = ((out["close"].astype(float) / day_open.replace(0.0, np.nan)) - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def intraday_summary(frame: pd.DataFrame, adv20: float | None = None) -> Dict[str, object]:
    if frame.empty:
        return {}
    latest_date = frame["TradeDate"].iloc[-1]
    day = frame[frame["TradeDate"].eq(latest_date)].copy()
    if day.empty:
        return {}
    close = day["close"].astype(float)
    volume = day["volume"].fillna(0.0).astype(float)
    vwap = float((close * volume).sum() / volume.sum()) if float(volume.sum()) > 0 else float(close.iloc[-1])
    latest = day.iloc[-1]
    session_open = float(day["open"].dropna().iloc[0]) if day["open"].notna().any() else float(close.iloc[0])
    session_high = float(day["high"].max())
    session_low = float(day["low"].min())
    out = {
        "IntradayDate": latest_date,
        "IntradayLastTimestamp": str(latest.get("Timestamp")),
        "IntradayOpen": session_open,
        "IntradayHigh": session_high,
        "IntradayLow": session_low,
        "IntradayLast": float(close.iloc[-1]),
        "IntradayVolume": float(volume.sum()),
        "IntradayVWAP": vwap,
        "IntradayRetFromOpenPct": ((float(close.iloc[-1]) / session_open) - 1.0) * 100.0 if session_open else np.nan,
        "IntradayRangePct": ((session_high - session_low) / session_open) * 100.0 if session_open else np.nan,
        "IntradayCloseVsVWAPPct": ((float(close.iloc[-1]) / vwap) - 1.0) * 100.0 if vwap else np.nan,
        "IntradayAvgVolumePerMinute": float(volume.sum()) / float(day.shape[0]) if day.shape[0] else np.nan,
        "IntradayTradedValue": float((close * volume).sum()),
    }
    for minutes in (5, 15, 30, 60):
        tail = day.tail(minutes)
        if tail.shape[0] >= 2:
            anchor = float(tail["close"].iloc[0])
            out[f"IntradayRet{minutes}mPct"] = ((float(tail["close"].iloc[-1]) / anchor) - 1.0) * 100.0 if anchor else np.nan
            out[f"IntradayVolume{minutes}m"] = float(tail["volume"].fillna(0.0).astype(float).sum())
    if adv20 and adv20 > 0:
        out["IntradayVolumePctADV20"] = (float(volume.sum()) / float(adv20)) * 100.0
    first_15 = day.head(15)
    first_30 = day.head(30)
    if not first_15.empty:
        out["IntradayFirst15mVolume"] = float(first_15["volume"].fillna(0.0).astype(float).sum())
        out["IntradayFirst15mRangePct"] = ((float(first_15["high"].max()) - float(first_15["low"].min())) / session_open) * 100.0 if session_open else np.nan
    if not first_30.empty:
        out["IntradayFirst30mVolume"] = float(first_30["volume"].fillna(0.0).astype(float).sum())
        out["IntradayFirst30mRangePct"] = ((float(first_30["high"].max()) - float(first_30["low"].min())) / session_open) * 100.0 if session_open else np.nan
    return out


def load_latest_depth(depth_dir: Path, tickers: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        path = depth_dir / f"{ticker}_depth.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame.tail(1).copy())
    if not frames:
        return pd.DataFrame()
    depth = pd.concat(frames, ignore_index=True)
    numeric_cols = [col for col in depth.columns if col not in {"Ticker", "FetchedAt", "PriceboardTime", "FloorCode"}]
    for col in numeric_cols:
        depth[col] = pd.to_numeric(depth[col], errors="coerce")
    depth["BidAskSpreadPct"] = ((depth["BestAsk1"] - depth["BestBid1"]) / depth["CurrentPrice"].replace(0.0, np.nan)) * 100.0
    bid_cols = [col for col in [f"BidVolume{level}" for level in range(1, 11)] if col in depth.columns]
    ask_cols = [col for col in [f"AskVolume{level}" for level in range(1, 11)] if col in depth.columns]
    bid_vol = depth[bid_cols].sum(axis=1)
    ask_vol = depth[ask_cols].sum(axis=1)
    depth["BidDepthVolume10"] = bid_vol
    depth["AskDepthVolume10"] = ask_vol
    depth["DepthImbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol).replace(0.0, np.nan)
    return depth


def load_cached_overview(tickers: Sequence[str], cache_dir: Path) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        path = cache_dir / f"{ticker}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = payload.get("fields") or {}
        rows.append({"Ticker": ticker, "OverviewFetchedAt": payload.get("fetched_at"), **fields})
    return pd.DataFrame(rows)


def load_cached_bctt_latest(tickers: Sequence[str], cache_dir: Path) -> pd.DataFrame:
    rows = []
    cache = VietstockBcttCache(cache_dir, max_age_hours=0)
    for ticker in tickers:
        record = cache.load(ticker)
        if record is None:
            continue
        quarterly = build_quarterly_feature_frame_from_tables(ticker, record.tables)
        if quarterly.empty:
            continue
        latest = quarterly.sort_values("PeriodEnd").tail(1).copy()
        latest["BCTTFetchedAt"] = record.fetched_at.isoformat()
        rows.append(latest)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_flow_metrics(tickers: Sequence[str], cache_dir: Path) -> pd.DataFrame:
    rows = []
    cache = CafeFFlowCache(cache_dir, max_age_hours=0)
    for ticker in tickers:
        foreign_path = cache.path_for("foreign", ticker)
        prop_path = cache.path_for("proprietary", ticker)
        if not foreign_path.exists() and not prop_path.exists():
            continue
        foreign = pd.read_csv(foreign_path) if foreign_path.exists() else pd.DataFrame()
        prop = pd.read_csv(prop_path) if prop_path.exists() else pd.DataFrame()
        rows.append(summarize_flow_metrics(ticker, foreign, prop))
    return pd.DataFrame(rows)


def load_macro_outputs(cache_dir: Path, output_dir: Path) -> Dict[str, str]:
    matrix = load_macro_factor_matrix(cache_dir)
    if matrix.empty:
        return {}
    macro_dir = output_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    tail = matrix.tail(260).rename_axis("Date").reset_index()
    tail.to_csv(macro_dir / "macro_matrix_tail.csv", index=False)
    latest = tail.tail(1).melt(id_vars=["Date"], var_name="Factor", value_name="Value")
    latest.to_csv(macro_dir / "latest_macro.csv", index=False)
    return {
        "macro_matrix_tail": "macro/macro_matrix_tail.csv",
        "latest_macro": "macro/latest_macro.csv",
    }


def build_breadth_frame(daily_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    equity_frames = {
        ticker: frame
        for ticker, frame in daily_frames.items()
        if not _is_index_ticker(ticker) and not frame.empty
    }
    if not equity_frames:
        return pd.DataFrame()

    def safe_median(series: pd.Series) -> float:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return float(numeric.median()) if not numeric.empty else np.nan

    combined = pd.concat(
        [
            frame.assign(Ticker=ticker)[
                [
                    "Date",
                    "Ticker",
                    "ret_1d_pct",
                    "ret_5d_pct",
                    "ret_20d_pct",
                    "close",
                    "volume",
                    "traded_value",
                    "sma_20",
                    "sma_50",
                    "high_52w",
                    "low_52w",
                    "volume_ratio_20",
                ]
            ]
            for ticker, frame in equity_frames.items()
        ],
        ignore_index=True,
    )
    for date, group in combined.groupby("Date", sort=True):
        ret = pd.to_numeric(group["ret_1d_pct"], errors="coerce")
        value = pd.to_numeric(group["traded_value"], errors="coerce").fillna(0.0)
        up_mask = ret > 0
        down_mask = ret < 0
        close = pd.to_numeric(group["close"], errors="coerce")
        rows.append(
            {
                "Date": pd.Timestamp(date).date().isoformat(),
                "TickerCount": int(group["Ticker"].nunique()),
                "Advancers": int(up_mask.sum()),
                "Decliners": int(down_mask.sum()),
                "Unchanged": int((ret == 0).sum()),
                "AdvancerPct": float(up_mask.mean() * 100.0) if len(group) else np.nan,
                "DeclinerPct": float(down_mask.mean() * 100.0) if len(group) else np.nan,
                "EqualWeightRet1dPct": float(ret.mean()) if ret.notna().any() else np.nan,
                "MedianRet1dPct": float(ret.median()) if ret.notna().any() else np.nan,
                "MedianRet5dPct": safe_median(group["ret_5d_pct"]),
                "MedianRet20dPct": safe_median(group["ret_20d_pct"]),
                "TotalVolume": float(pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).sum()),
                "TotalTradedValue": float(value.sum()),
                "UpTradedValuePct": float(value[up_mask].sum() / value.sum() * 100.0) if float(value.sum()) > 0 else np.nan,
                "AboveSMA20Pct": float((close > pd.to_numeric(group["sma_20"], errors="coerce")).mean() * 100.0),
                "AboveSMA50Pct": float((close > pd.to_numeric(group["sma_50"], errors="coerce")).mean() * 100.0),
                "New52wHighCount": int((close >= pd.to_numeric(group["high_52w"], errors="coerce")).sum()),
                "New52wLowCount": int((close <= pd.to_numeric(group["low_52w"], errors="coerce")).sum()),
                "MedianVolumeRatio20": safe_median(group["volume_ratio_20"]),
            }
        )
    return pd.DataFrame(rows)


def load_sector_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "Sector"])
    frame = pd.read_csv(path)
    if "Ticker" not in frame.columns:
        return pd.DataFrame(columns=["Ticker", "Sector"])
    out = frame.copy()
    out["Ticker"] = out["Ticker"].map(_normalise_ticker)
    if "Sector" not in out.columns:
        out["Sector"] = ""
    return out[["Ticker", "Sector"]].drop_duplicates(subset=["Ticker"], keep="last")


def build_cross_section_latest(latest: pd.DataFrame) -> pd.DataFrame:
    if latest.empty or "Ticker" not in latest.columns:
        return pd.DataFrame()
    out = latest.copy()
    rank_columns = [
        "Ret1dPct",
        "Ret5dPct",
        "Ret20dPct",
        "Ret60dPct",
        "VolumeRatio20",
        "ValueRatio20",
        "DailyTradedValue",
        "IntradayVolumePctADV20",
        "ExcessRet20dVsVNINDEXPct",
        "RealizedVol20dPct",
        "Drawdown60dPct",
    ]
    for column in rank_columns:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
            if not numeric.notna().any():
                continue
            ascending = column in {"RealizedVol20dPct", "Drawdown60dPct"}
            out[f"{column}Rank"] = numeric.rank(ascending=ascending, method="min", na_option="bottom")
            out[f"{column}PctRank"] = numeric.rank(pct=True, ascending=not ascending)
    return out


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def build_sector_latest(latest: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    if latest.empty or sector_map.empty:
        return pd.DataFrame()
    scoped = latest.merge(sector_map, on="Ticker", how="left")
    scoped = scoped[scoped["Sector"].fillna("").astype(str).str.len() > 0].copy()
    if scoped.empty:
        return pd.DataFrame()
    rows = []
    for sector, group in scoped.groupby("Sector", sort=True):
        rows.append(
            {
                "Sector": sector,
                "TickerCount": int(group["Ticker"].nunique()),
                "MedianRet1dPct": _safe_median(_numeric_column(group, "Ret1dPct")),
                "MedianRet20dPct": _safe_median(_numeric_column(group, "Ret20dPct")),
                "TotalTradedValue": float(_numeric_column(group, "DailyTradedValue").fillna(0.0).sum()),
                "MedianVolumeRatio20": _safe_median(_numeric_column(group, "VolumeRatio20")),
                "AboveSMA20Pct": float((_numeric_column(group, "LastClose") > _numeric_column(group, "SMA20")).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def append_bctt_refresh_summary_status(output: Path, statuses: List[Dict[str, object]], files: Dict[str, object]) -> None:
    summary_path = Path("out/data_hub/vietstock_bctt_cache_summary.csv")
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    status_dir = output / "source_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    dest = status_dir / "bctt_refresh_summary.csv"
    summary.to_csv(dest, index=False)
    files["bctt_refresh_summary"] = "source_status/bctt_refresh_summary.csv"
    success_count = int((pd.to_numeric(summary.get("QuarterCount"), errors="coerce").fillna(0) > 0).sum())
    failed = summary[summary.get("Error", "").fillna("").astype(str).str.len() > 0] if "Error" in summary.columns else pd.DataFrame()
    if success_count == len(summary):
        status = "ok"
    elif success_count > 0:
        status = "partial"
    else:
        status = "error"
    detail = ""
    if not failed.empty:
        detail = "; ".join(
            f"{row.Ticker}: {str(row.Error)[:160]}"
            for row in failed.head(10).itertuples(index=False)
            if hasattr(row, "Ticker") and hasattr(row, "Error")
        )
    statuses.append(
        _source_status(
            "Vietstock BCTT refresh summary",
            attempted=True,
            status=status,
            ticker_count=len(summary),
            row_count=success_count,
            detail=detail,
            output="source_status/bctt_refresh_summary.csv",
        )
    )


def append_industry_map_refresh_summary_status(output: Path, statuses: List[Dict[str, object]], files: Dict[str, object]) -> None:
    summary_path = Path("out/data_hub/industry_map_refresh_summary.csv")
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    status_dir = output / "source_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    dest = status_dir / "industry_map_refresh_summary.csv"
    summary.to_csv(dest, index=False)
    files["industry_map_refresh_summary"] = "source_status/industry_map_refresh_summary.csv"
    ok_count = int((summary.get("Status", "").fillna("").astype(str) == "ok").sum()) if "Status" in summary.columns else 0
    failed = summary[summary.get("Status", "").fillna("").astype(str) == "error"] if "Status" in summary.columns else pd.DataFrame()
    if ok_count == len(summary):
        status = "ok"
    elif ok_count > 0:
        status = "partial"
    else:
        status = "error"
    detail = ""
    if not failed.empty:
        detail = "; ".join(
            f"{row.Ticker}: {str(row.Error)[:160]}"
            for row in failed.head(10).itertuples(index=False)
            if hasattr(row, "Ticker") and hasattr(row, "Error")
        )
    statuses.append(
        _source_status(
            "Vietstock sector profiles",
            attempted=True,
            status=status,
            ticker_count=len(summary),
            row_count=ok_count,
            detail=detail,
            output="source_status/industry_map_refresh_summary.csv",
        )
    )


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _available_columns(frame: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [column for column in columns if column in frame.columns]


def build_symbol_latest_bundle(latest: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "Ticker",
        "LatestDate",
        "LastClose",
        "Ret1dPct",
        "Ret5dPct",
        "Ret20dPct",
        "Ret60dPct",
        "DailyVolume",
        "DailyTradedValue",
        "VolumeRatio20",
        "ValueRatio20",
        "RSI14",
        "ATR14Pct",
        "RealizedVol20dPct",
        "Drawdown60dPct",
        "Pos52wPct",
        "ExcessRet20dVsVNINDEXPct",
        "ExcessRet20dVsVN30Pct",
        "IntradayDate",
        "IntradayLastTimestamp",
        "IntradayLast",
        "IntradayRetFromOpenPct",
        "IntradayVolume",
        "IntradayVolumePctADV20",
        "BidAskSpreadPct",
        "DepthImbalance",
        "ForeignNetValue20d",
        "ProprietaryNetValue20d",
        "ForwardPE",
        "PB",
        "ROE",
        "EPS4Q",
        "BVPS",
        "RevenueGrowthYoYPct",
        "NetProfitGrowthYoYPct",
    ]
    columns = _available_columns(latest, preferred_columns)
    return latest[columns].copy() if columns else pd.DataFrame()


def build_market_snapshot_bundle(
    *,
    generated_at: str,
    tickers: Sequence[str],
    breadth: pd.DataFrame,
    sector_latest: pd.DataFrame,
    macro_latest_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
        {"Section": "artifact", "Metric": "GeneratedAt", "Value": generated_at},
        {"Section": "universe", "Metric": "ConfiguredSymbolCount", "Value": len(tickers)},
        {"Section": "universe", "Metric": "EquityTickerCount", "Value": len([ticker for ticker in tickers if not _is_index_ticker(ticker)])},
    ]
    if not breadth.empty:
        latest_breadth = breadth.tail(1).iloc[0]
        for column in [
            "Date",
            "TickerCount",
            "Advancers",
            "Decliners",
            "Unchanged",
            "EqualWeightRet1dPct",
            "MedianRet1dPct",
            "MedianRet20dPct",
            "TotalTradedValue",
            "UpTradedValuePct",
            "AboveSMA20Pct",
            "AboveSMA50Pct",
        ]:
            if column in latest_breadth.index:
                rows.append({"Section": "breadth_latest", "Metric": column, "Value": latest_breadth.get(column)})
    if not sector_latest.empty:
        for row in sector_latest.sort_values("TotalTradedValue", ascending=False).head(10).itertuples(index=False):
            sector = getattr(row, "Sector", "")
            rows.append({"Section": "sector_latest", "Metric": f"{sector}.TickerCount", "Value": getattr(row, "TickerCount", "")})
            rows.append({"Section": "sector_latest", "Metric": f"{sector}.MedianRet1dPct", "Value": getattr(row, "MedianRet1dPct", "")})
            rows.append({"Section": "sector_latest", "Metric": f"{sector}.TotalTradedValue", "Value": getattr(row, "TotalTradedValue", "")})
    if macro_latest_path.exists():
        macro = pd.read_csv(macro_latest_path)
        for row in macro.itertuples(index=False):
            rows.append({"Section": "macro_latest", "Metric": str(getattr(row, "Factor", "")), "Value": getattr(row, "Value", "")})
    return pd.DataFrame(rows)


def infer_column_group(column: str) -> str:
    lower = column.lower()
    if lower == "ticker":
        return "identifier"
    if lower in {"lastclose", "dailyopen", "dailyhigh", "dailylow", "intradaylast", "intradayopen", "intradayhigh", "intradaylow"}:
        return "price_ohlc"
    if "intraday" in lower or "vwap" in lower:
        return "intraday"
    if "foreign" in lower or "proprietary" in lower:
        return "flows"
    if "pe" in lower or lower in {"pb", "roe", "eps4q", "bvps"} or "growth" in lower or "margin" in lower or "bctt" in lower:
        return "fundamentals"
    if "bid" in lower or "ask" in lower or "depth" in lower or "spread" in lower:
        return "depth"
    if "ret" in lower or "sma" in lower or "ema" in lower or "rsi" in lower or "pos52" in lower:
        return "trend_momentum"
    if "volume" in lower or "value" in lower or "adv" in lower:
        return "liquidity"
    if "vol" in lower or "atr" in lower or "drawdown" in lower or "range" in lower:
        return "risk_volatility"
    if "date" in lower or "time" in lower:
        return "time"
    return "other"


def build_column_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        rows.append(
            {
                "Column": column,
                "Group": infer_column_group(column),
                "DType": str(series.dtype),
                "NonNullCount": non_null,
                "NullCount": int(len(series) - non_null),
                "Example": "" if series.dropna().empty else str(series.dropna().iloc[0])[:120],
            }
        )
    return pd.DataFrame(rows)


def build_ticker_catalog(tickers: Sequence[str], output: Path, latest: pd.DataFrame) -> pd.DataFrame:
    latest_tickers = set(latest["Ticker"].astype(str)) if "Ticker" in latest.columns else set()
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "Ticker": ticker,
                "Kind": "index" if _is_index_ticker(ticker) else "equity",
                "InLatestMetrics": ticker in latest_tickers,
                "DailyPath": f"daily/{ticker}.csv" if (output / "daily" / f"{ticker}.csv").exists() else "",
                "IntradayPath": f"intraday/{ticker}.csv" if (output / "intraday" / f"{ticker}.csv").exists() else "",
                "MinuteProfilePath": f"intraday/minute_profile/{ticker}.csv" if (output / "intraday" / "minute_profile" / f"{ticker}.csv").exists() else "",
            }
        )
    return pd.DataFrame(rows)


def build_file_catalog(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output).as_posix()
        if rel == "index/file_catalog.csv":
            continue
        row: Dict[str, object] = {
            "Path": rel,
            "Directory": path.parent.relative_to(output).as_posix() if path.parent != output else ".",
            "Suffix": path.suffix.lower(),
            "Bytes": path.stat().st_size,
            "Rows": "",
            "Columns": "",
            "Ticker": "",
        }
        if path.suffix.lower() == ".csv":
            try:
                header = pd.read_csv(path, nrows=0)
                row["Columns"] = len(header.columns)
                with path.open("r", encoding="utf-8") as f:
                    row["Rows"] = max(0, sum(1 for _ in f) - 1)
            except Exception:
                pass
            if path.parent.name in {"daily", "intraday", "minute_profile"}:
                row["Ticker"] = path.stem
        rows.append(row)
    return pd.DataFrame(rows)


def write_retrieval_outputs(
    *,
    output: Path,
    generated_at: str,
    tickers: Sequence[str],
    latest: pd.DataFrame,
    breadth: pd.DataFrame,
    sector_latest: pd.DataFrame,
    source_status: pd.DataFrame,
    files: Dict[str, object],
) -> Dict[str, object]:
    index_dir = output / "index"
    bundle_dir = output / "bundles"
    index_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    symbol_latest = build_symbol_latest_bundle(latest)
    if not symbol_latest.empty:
        symbol_latest.to_csv(bundle_dir / "symbol_latest.csv", index=False)
        files["bundle_symbol_latest"] = "bundles/symbol_latest.csv"

    market_snapshot = build_market_snapshot_bundle(
        generated_at=generated_at,
        tickers=tickers,
        breadth=breadth,
        sector_latest=sector_latest,
        macro_latest_path=output / "macro" / "latest_macro.csv",
    )
    market_snapshot.to_csv(bundle_dir / "market_snapshot.csv", index=False)
    files["bundle_market_snapshot"] = "bundles/market_snapshot.csv"

    source_status.to_csv(bundle_dir / "source_audit.csv", index=False)
    files["bundle_source_audit"] = "bundles/source_audit.csv"

    build_ticker_catalog(tickers, output, latest).to_csv(index_dir / "ticker_catalog.csv", index=False)
    build_column_catalog(latest).to_csv(index_dir / "column_catalog.csv", index=False)
    files["ticker_catalog"] = "index/ticker_catalog.csv"
    files["column_catalog"] = "index/column_catalog.csv"

    retrieval_map = {
        "generated_at": generated_at,
        "purpose": "Fast retrieval map for ChatGPT, repo connectors, or Google Drive connectors.",
        "start_here": "START_HERE.json",
        "minimal_read_order": [
            "START_HERE.json",
            "bundles/source_audit.csv",
            "bundles/market_snapshot.csv",
            "bundles/symbol_latest.csv",
            "index/ticker_catalog.csv",
            "index/file_catalog.csv",
        ],
        "use_cases": {
            "market_overview": ["bundles/source_audit.csv", "bundles/market_snapshot.csv", "bundles/symbol_latest.csv"],
            "single_ticker": ["index/ticker_catalog.csv", "bundles/symbol_latest.csv", "daily/{ticker}.csv", "intraday/minute_profile/{ticker}.csv"],
            "source_audit": ["bundles/source_audit.csv", "source_status.csv", "source_status/*.csv"],
            "column_lookup": ["index/column_catalog.csv", "latest_metrics.csv"],
            "deep_dive": ["manifest.json", "index/file_catalog.csv", "latest_metrics.csv"],
        },
    }
    _write_json(bundle_dir / "retrieval_map.json", retrieval_map)
    files["bundle_retrieval_map"] = "bundles/retrieval_map.json"

    start_here = {
        "generated_at": generated_at,
        "purpose": "Numeric-only VN market data hub optimized for connector retrieval.",
        "rules": [
            "No news.",
            "No forecast/model output.",
            "No buy/sell recommendation.",
            "Check source audit before trusting a metric.",
            "Use bundles first; open per-ticker files only when drilling down.",
        ],
        "minimal_read_order": retrieval_map["minimal_read_order"],
        "top_level_files": {
            "manifest": "manifest.json",
            "source_audit": "bundles/source_audit.csv",
            "market_snapshot": "bundles/market_snapshot.csv",
            "symbol_latest": "bundles/symbol_latest.csv",
            "ticker_catalog": "index/ticker_catalog.csv",
            "file_catalog": "index/file_catalog.csv",
            "column_catalog": "index/column_catalog.csv",
        },
        "universe": {
            "symbol_count": len(tickers),
            "equity_count": len([ticker for ticker in tickers if not _is_index_ticker(ticker)]),
            "index_count": len([ticker for ticker in tickers if _is_index_ticker(ticker)]),
            "symbols": list(tickers),
        },
    }
    _write_json(output / "START_HERE.json", start_here)
    files["start_here"] = "START_HERE.json"

    build_file_catalog(output).to_csv(index_dir / "file_catalog.csv", index=False)
    files["file_catalog"] = "index/file_catalog.csv"
    return retrieval_map


def refresh_sources(config: Mapping[str, object], tickers: Sequence[str], *, refresh_all: bool) -> List[Dict[str, object]]:
    statuses: List[Dict[str, object]] = []
    daily_dir = _path(config, "daily_cache", "out/data")
    intraday_dir = _path(config, "intraday_cache", "out/data/intraday_1m")
    depth_dir = _path(config, "depth_cache", "out/data/depth_snapshots")
    if _refresh_enabled(config, "daily", refresh_all):
        errors = []
        for ticker in tickers:
            try:
                ensure_ohlc_cache(ticker, outdir=str(daily_dir), min_days=_window(config, "daily_history_days", 900))
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        statuses.append(
            _source_status(
                "VNDIRECT dchart daily",
                attempted=True,
                status="partial" if errors else "ok",
                ticker_count=len(tickers),
                detail="; ".join(errors[:10]),
                output=str(daily_dir),
            )
        )
    else:
        statuses.append(_source_status("VNDIRECT dchart daily", attempted=False, status="skipped_disabled", ticker_count=len(tickers), output=str(daily_dir)))
    if _refresh_enabled(config, "intraday", refresh_all):
        errors = []
        for ticker in tickers:
            try:
                ensure_intraday_cache(ticker, outdir=str(intraday_dir), min_days=_window(config, "intraday_history_days", 30), resolution="1")
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        statuses.append(
            _source_status(
                "VNDIRECT dchart intraday 1m",
                attempted=True,
                status="partial" if errors else "ok",
                ticker_count=len(tickers),
                detail="; ".join(errors[:10]),
                output=str(intraday_dir),
            )
        )
    else:
        statuses.append(_source_status("VNDIRECT dchart intraday 1m", attempted=False, status="skipped_disabled", ticker_count=len(tickers), output=str(intraday_dir)))
    if _refresh_enabled(config, "depth", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        if equity_tickers:
            try:
                depth = refresh_depth_for_intraday_cache(
                    equity_tickers,
                    intraday_dir,
                    resolution="1",
                    depth_dir=depth_dir,
                    retention_days=_window(config, "depth_history_days", 30),
                )
                statuses.append(_source_status("VNDIRECT priceboard depth", attempted=True, status="ok", ticker_count=len(equity_tickers), row_count=len(depth), output=str(depth_dir)))
            except Exception as exc:
                statuses.append(_source_status("VNDIRECT priceboard depth", attempted=True, status="error", ticker_count=len(equity_tickers), detail=str(exc), output=str(depth_dir)))
        else:
            statuses.append(_source_status("VNDIRECT priceboard depth", attempted=True, status="no_equity_tickers", ticker_count=0, output=str(depth_dir)))
    else:
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        statuses.append(_source_status("VNDIRECT priceboard depth", attempted=False, status="skipped_disabled", ticker_count=len(equity_tickers), output=str(depth_dir)))
    if _refresh_enabled(config, "vietstock_overview", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        if equity_tickers:
            try:
                overview = build_fundamental_frame(equity_tickers, _path(config, "vietstock_overview_cache", "out/vietstock_overview"), max_age_hours=24)
                statuses.append(_source_status("Vietstock overview", attempted=True, status="ok", ticker_count=len(equity_tickers), row_count=len(overview), output=str(_path(config, "vietstock_overview_cache", "out/vietstock_overview"))))
            except Exception as exc:
                statuses.append(_source_status("Vietstock overview", attempted=True, status="error", ticker_count=len(equity_tickers), detail=str(exc), output=str(_path(config, "vietstock_overview_cache", "out/vietstock_overview"))))
        else:
            statuses.append(_source_status("Vietstock overview", attempted=True, status="no_equity_tickers", ticker_count=0, output=str(_path(config, "vietstock_overview_cache", "out/vietstock_overview"))))
    else:
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        statuses.append(_source_status("Vietstock overview", attempted=False, status="skipped_disabled", ticker_count=len(equity_tickers), output=str(_path(config, "vietstock_overview_cache", "out/vietstock_overview"))))
    if _refresh_enabled(config, "vietstock_bctt", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        if equity_tickers:
            try:
                bctt = load_or_fetch_bctt_feature_frame(
                    equity_tickers,
                    _path(config, "vietstock_bctt_cache", "out/vietstock_bctt"),
                    max_age_hours=720,
                )
                statuses.append(_source_status("Vietstock BCTT", attempted=True, status="ok", ticker_count=len(equity_tickers), row_count=len(bctt), output=str(_path(config, "vietstock_bctt_cache", "out/vietstock_bctt"))))
            except Exception as exc:
                statuses.append(_source_status("Vietstock BCTT", attempted=True, status="error", ticker_count=len(equity_tickers), detail=str(exc), output=str(_path(config, "vietstock_bctt_cache", "out/vietstock_bctt"))))
        else:
            statuses.append(_source_status("Vietstock BCTT", attempted=True, status="no_equity_tickers", ticker_count=0, output=str(_path(config, "vietstock_bctt_cache", "out/vietstock_bctt"))))
    else:
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        statuses.append(
            _source_status(
                "Vietstock BCTT",
                attempted=False,
                status="skipped_disabled_use_refresh_bctt",
                ticker_count=len(equity_tickers),
                output=str(_path(config, "vietstock_bctt_cache", "out/vietstock_bctt")),
            )
        )
    if _refresh_enabled(config, "cafef_flows", refresh_all):
        cache = CafeFFlowCache(_path(config, "cafef_cache", "out/cafef_flows"), max_age_hours=4)
        errors = []
        ok_count = 0
        for ticker in [item for item in tickers if not _is_index_ticker(item)]:
            try:
                ensure_foreign_flow_df(ticker, cache)
                ensure_proprietary_flow_df(ticker, cache)
                ok_count += 1
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        statuses.append(
            _source_status(
                "CafeF foreign/proprietary flows",
                attempted=True,
                status="partial" if errors else "ok",
                ticker_count=ok_count + len(errors),
                row_count=ok_count,
                detail="; ".join(errors[:10]),
                output=str(_path(config, "cafef_cache", "out/cafef_flows")),
            )
        )
    else:
        equity_tickers = [ticker for ticker in tickers if not _is_index_ticker(ticker)]
        statuses.append(_source_status("CafeF foreign/proprietary flows", attempted=False, status="skipped_disabled", ticker_count=len(equity_tickers), output=str(_path(config, "cafef_cache", "out/cafef_flows"))))
    if _refresh_enabled(config, "macro", refresh_all):
        try:
            summary = refresh_macro_factor_cache(Path("config/macro_factors.yaml"), _path(config, "macro_cache", "out/macro_factors"), max_age_hours=24)
            statuses.append(_source_status("FRED/Stooq macro cache", attempted=True, status="ok", row_count=len(summary), output=str(_path(config, "macro_cache", "out/macro_factors"))))
        except Exception as exc:
            statuses.append(_source_status("FRED/Stooq macro cache", attempted=True, status="error", detail=str(exc), output=str(_path(config, "macro_cache", "out/macro_factors"))))
    else:
        statuses.append(_source_status("FRED/Stooq macro cache", attempted=False, status="skipped_disabled", output=str(_path(config, "macro_cache", "out/macro_factors"))))
    return statuses


def build_data_hub(config_path: Path, output_dir: Path | None = None, *, refresh_all: bool = False) -> Dict[str, object]:
    config = load_config(config_path)
    tickers = [_normalise_ticker(ticker) for ticker in config.get("tickers", []) if _normalise_ticker(ticker)]
    output = output_dir or _path(config, "output_dir", str(DEFAULT_OUTPUT_DIR))
    generated_at = datetime.now(timezone.utc).isoformat()
    daily_dir = _path(config, "daily_cache", "out/data")
    intraday_dir = _path(config, "intraday_cache", "out/data/intraday_1m")
    depth_dir = _path(config, "depth_cache", "out/data/depth_snapshots")
    _clean_dir(output)
    source_status = refresh_sources(config, tickers, refresh_all=refresh_all)

    (output / "daily").mkdir(parents=True, exist_ok=True)
    (output / "intraday").mkdir(parents=True, exist_ok=True)
    (output / "intraday" / "minute_profile").mkdir(parents=True, exist_ok=True)
    latest_rows = []
    files: Dict[str, object] = {}
    daily_frames = {
        ticker: enrich_daily_frame(_read_daily(daily_dir / f"{ticker}_daily.csv", ticker))
        for ticker in tickers
    }
    daily_frames = add_market_relative_metrics(daily_frames)
    intraday_frames = {
        ticker: enrich_intraday_frame(_read_intraday(intraday_dir / f"{ticker}_1m.csv", ticker))
        for ticker in tickers
    }

    intraday_summary_rows = []
    for ticker in tickers:
        daily = daily_frames.get(ticker, pd.DataFrame())
        if not daily.empty:
            recent_daily = daily.tail(_window(config, "recent_daily_rows", 260)).copy()
            recent_daily.to_csv(output / "daily" / f"{ticker}.csv", index=False)
            row = {"Ticker": ticker, **latest_daily_metrics(daily)}
        else:
            row = {"Ticker": ticker}
        intraday = intraday_frames.get(ticker, pd.DataFrame())
        if not intraday.empty:
            intraday.tail(_window(config, "recent_intraday_rows", 390)).to_csv(output / "intraday" / f"{ticker}.csv", index=False)
            latest_date = intraday["TradeDate"].iloc[-1]
            intraday[intraday["TradeDate"].eq(latest_date)].to_csv(output / "intraday" / "minute_profile" / f"{ticker}.csv", index=False)
            intraday_metrics = intraday_summary(intraday, row.get("ADV20Shares"))
            row.update(intraday_metrics)
            intraday_summary_rows.append({"Ticker": ticker, **intraday_metrics})
        latest_rows.append(row)

    latest = pd.DataFrame(latest_rows)
    if intraday_summary_rows:
        pd.DataFrame(intraday_summary_rows).to_csv(output / "intraday" / "summary_by_ticker.csv", index=False)
        files["intraday_summary"] = "intraday/summary_by_ticker.csv"
        files["intraday_minute_profile_dir"] = "intraday/minute_profile/"

    breadth = build_breadth_frame(daily_frames)
    if not breadth.empty:
        (output / "market").mkdir(parents=True, exist_ok=True)
        breadth.to_csv(output / "market" / "breadth_daily.csv", index=False)
        files["breadth_daily"] = "market/breadth_daily.csv"

    depth = load_latest_depth(depth_dir, tickers)
    if not depth.empty:
        (output / "depth").mkdir(parents=True, exist_ok=True)
        depth.to_csv(output / "depth" / "latest_depth.csv", index=False)
        latest = latest.merge(depth, on="Ticker", how="left")
        files["latest_depth"] = "depth/latest_depth.csv"

    overview = load_cached_overview(tickers, _path(config, "vietstock_overview_cache", "out/vietstock_overview"))
    if not overview.empty:
        (output / "fundamentals").mkdir(parents=True, exist_ok=True)
        overview.to_csv(output / "fundamentals" / "vietstock_overview.csv", index=False)
        latest = latest.merge(overview, on="Ticker", how="left")
        files["vietstock_overview"] = "fundamentals/vietstock_overview.csv"

    bctt = load_cached_bctt_latest(tickers, _path(config, "vietstock_bctt_cache", "out/vietstock_bctt"))
    if not bctt.empty:
        (output / "fundamentals").mkdir(parents=True, exist_ok=True)
        bctt.to_csv(output / "fundamentals" / "vietstock_bctt_latest.csv", index=False)
        latest = latest.merge(bctt, on="Ticker", how="left")
        files["vietstock_bctt_latest"] = "fundamentals/vietstock_bctt_latest.csv"

    flows = load_flow_metrics(tickers, _path(config, "cafef_cache", "out/cafef_flows"))
    if not flows.empty:
        (output / "flows").mkdir(parents=True, exist_ok=True)
        flows.to_csv(output / "flows" / "cafef_flows.csv", index=False)
        latest = latest.merge(flows, on="Ticker", how="left")
        files["cafef_flows"] = "flows/cafef_flows.csv"

    macro_files = load_macro_outputs(_path(config, "macro_cache", "out/macro_factors"), output)
    files.update(macro_files)

    sector_map = load_sector_map(_path(config, "sector_map", "data/industry_map.csv"))
    source_status.append(
        _source_status(
            "Market membership / sector map",
            attempted=False,
            status="cache_loaded" if not sector_map.empty else "missing_cache",
            ticker_count=int(sector_map["Ticker"].nunique()) if not sector_map.empty else 0,
            row_count=len(sector_map),
            output=str(_path(config, "sector_map", "data/industry_map.csv")),
        )
    )
    cross_section = build_cross_section_latest(latest)
    if not cross_section.empty:
        (output / "market").mkdir(parents=True, exist_ok=True)
        cross_section.to_csv(output / "market" / "cross_section_latest.csv", index=False)
        files["cross_section_latest"] = "market/cross_section_latest.csv"
    sector_latest = build_sector_latest(latest, sector_map)
    if not sector_latest.empty:
        (output / "market").mkdir(parents=True, exist_ok=True)
        sector_latest.to_csv(output / "market" / "sector_latest.csv", index=False)
        files["sector_latest"] = "market/sector_latest.csv"

    append_industry_map_refresh_summary_status(output, source_status, files)
    append_bctt_refresh_summary_status(output, source_status, files)
    source_status_frame = pd.DataFrame(source_status)
    source_status_frame.to_csv(output / "source_status.csv", index=False)
    (output / "source_status.json").write_text(json.dumps(_json_safe(source_status), ensure_ascii=False, indent=2), encoding="utf-8")
    files["source_status"] = "source_status.csv"
    files["source_status_json"] = "source_status.json"

    latest.to_csv(output / "latest_metrics.csv", index=False)
    pd.DataFrame(API_CATALOG).to_csv(output / "api_catalog.csv", index=False)
    pd.DataFrame(CALCULATION_CATALOG).to_csv(output / "calculation_catalog.csv", index=False)
    pd.DataFrame({"Ticker": tickers}).to_csv(output / "tickers.csv", index=False)
    files.update(
        {
            "latest_metrics": "latest_metrics.csv",
            "api_catalog": "api_catalog.csv",
            "calculation_catalog": "calculation_catalog.csv",
            "tickers": "tickers.csv",
            "daily_dir": "daily/",
            "intraday_dir": "intraday/",
        }
    )
    retrieval_map = write_retrieval_outputs(
        output=output,
        generated_at=generated_at,
        tickers=tickers,
        latest=latest,
        breadth=breadth,
        sector_latest=sector_latest,
        source_status=source_status_frame,
        files=files,
    )

    manifest = {
        "generated_at": generated_at,
        "config": str(config_path),
        "tickers": tickers,
        "purpose": "Numeric-only market data hub for fast ChatGPT browsing. No news, no recommendations, no model forecasts.",
        "read_order_for_chatgpt": [
            "START_HERE.json",
            "bundles/source_audit.csv",
            "bundles/market_snapshot.csv",
            "bundles/symbol_latest.csv",
            "index/ticker_catalog.csv",
            "index/file_catalog.csv",
            "manifest.json",
            "source_status.csv",
            "latest_metrics.csv",
            "api_catalog.csv",
            "calculation_catalog.csv",
            "market/cross_section_latest.csv if present",
            "market/breadth_daily.csv if present",
            "daily/{ticker}.csv",
            "intraday/{ticker}.csv",
            "intraday/minute_profile/{ticker}.csv",
            "depth/latest_depth.csv if present",
            "fundamentals/vietstock_overview.csv if present",
            "fundamentals/vietstock_bctt_latest.csv if present",
            "flows/cafef_flows.csv if present",
            "macro/latest_macro.csv if present",
        ],
        "files": files,
        "retrieval_map": retrieval_map,
        "api_catalog": API_CATALOG,
        "calculation_catalog": CALCULATION_CATALOG,
    }
    (output / "manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    build_file_catalog(output).to_csv(output / "index" / "file_catalog.csv", index=False)
    (output / "README.md").write_text(
        "\n".join(
            [
                "# Numeric Data Hub",
                "",
                "This directory is optimized for ChatGPT browsing.",
                "",
                "- Numeric data only: prices, volume, order book, flows, fundamentals, macro caches.",
                "- No news, no recommendations, no model forecast.",
                "- Start with `START_HERE.json`, then `bundles/source_audit.csv`, `bundles/market_snapshot.csv`, `bundles/symbol_latest.csv`, and `index/ticker_catalog.csv`.",
                "- Use `index/file_catalog.csv` to locate drill-down files without scanning every directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build numeric-only data hub for ChatGPT browsing.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true", help="Refresh every enabled source before building outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_data_hub(args.config, args.output_dir, refresh_all=bool(args.refresh))
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
