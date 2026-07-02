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
        "NumericData": "best bid/ask levels, bid/ask volume, match/current price, foreign room",
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


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


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
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_200"] = close.rolling(200).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["dist_sma20_pct"] = ((close / out["sma_20"]) - 1.0) * 100.0
    out["dist_sma50_pct"] = ((close / out["sma_50"]) - 1.0) * 100.0
    out["range_pct"] = ((high - low) / prev_close.replace(0.0, np.nan)) * 100.0
    out["body_pct"] = ((close - out["open"].astype(float)) / prev_close.replace(0.0, np.nan)) * 100.0
    out["adv20_shares"] = out["volume"].astype(float).rolling(20).mean()
    out["volume_ratio_20"] = out["volume"].astype(float) / out["adv20_shares"].replace(0.0, np.nan)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr14_pct"] = (out["atr14"] / close.replace(0.0, np.nan)) * 100.0
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
        "SMA20": row.get("sma_20"),
        "SMA50": row.get("sma_50"),
        "SMA200": row.get("sma_200"),
        "EMA20": row.get("ema_20"),
        "DistSMA20Pct": row.get("dist_sma20_pct"),
        "DistSMA50Pct": row.get("dist_sma50_pct"),
        "RSI14": row.get("rsi14"),
        "ATR14": row.get("atr14"),
        "ATR14Pct": row.get("atr14_pct"),
        "ADV20Shares": row.get("adv20_shares"),
        "VolumeRatio20": row.get("volume_ratio_20"),
        "High52w": row.get("high_52w"),
        "Low52w": row.get("low_52w"),
        "Pos52wPct": row.get("pos_52w_pct"),
    }


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
    }
    for minutes in (5, 15, 30, 60):
        tail = day.tail(minutes)
        if tail.shape[0] >= 2:
            anchor = float(tail["close"].iloc[0])
            out[f"IntradayRet{minutes}mPct"] = ((float(tail["close"].iloc[-1]) / anchor) - 1.0) * 100.0 if anchor else np.nan
    if adv20 and adv20 > 0:
        out["IntradayVolumePctADV20"] = (float(volume.sum()) / float(adv20)) * 100.0
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
    numeric_cols = [col for col in depth.columns if col not in {"Ticker", "FetchedAt", "PriceboardTime"}]
    for col in numeric_cols:
        depth[col] = pd.to_numeric(depth[col], errors="coerce")
    depth["BidAskSpreadPct"] = ((depth["BestAsk1"] - depth["BestBid1"]) / depth["CurrentPrice"].replace(0.0, np.nan)) * 100.0
    bid_vol = depth[[col for col in ["BidVolume1", "BidVolume2", "BidVolume3"] if col in depth.columns]].sum(axis=1)
    ask_vol = depth[[col for col in ["AskVolume1", "AskVolume2", "AskVolume3"] if col in depth.columns]].sum(axis=1)
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


def refresh_sources(config: Mapping[str, object], tickers: Sequence[str], *, refresh_all: bool) -> None:
    daily_dir = _path(config, "daily_cache", "out/data")
    intraday_dir = _path(config, "intraday_cache", "out/data/intraday_1m")
    depth_dir = _path(config, "depth_cache", "out/data/depth_snapshots")
    if _refresh_enabled(config, "daily", refresh_all):
        for ticker in tickers:
            ensure_ohlc_cache(ticker, outdir=str(daily_dir), min_days=_window(config, "daily_history_days", 900))
    if _refresh_enabled(config, "intraday", refresh_all):
        for ticker in tickers:
            ensure_intraday_cache(ticker, outdir=str(intraday_dir), min_days=_window(config, "intraday_history_days", 30), resolution="1")
    if _refresh_enabled(config, "depth", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not ticker.startswith("VN")]
        if equity_tickers:
            refresh_depth_for_intraday_cache(equity_tickers, intraday_dir, resolution="1", depth_dir=depth_dir)
    if _refresh_enabled(config, "vietstock_overview", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not ticker.startswith("VN")]
        if equity_tickers:
            build_fundamental_frame(equity_tickers, _path(config, "vietstock_overview_cache", "out/vietstock_overview"), max_age_hours=24)
    if _refresh_enabled(config, "vietstock_bctt", refresh_all):
        equity_tickers = [ticker for ticker in tickers if not ticker.startswith("VN")]
        if equity_tickers:
            load_or_fetch_bctt_feature_frame(
                equity_tickers,
                _path(config, "vietstock_bctt_cache", "out/vietstock_bctt"),
                max_age_hours=720,
            )
    if _refresh_enabled(config, "cafef_flows", refresh_all):
        cache = CafeFFlowCache(_path(config, "cafef_cache", "out/cafef_flows"), max_age_hours=4)
        for ticker in [item for item in tickers if not item.startswith("VN")]:
            ensure_foreign_flow_df(ticker, cache)
            ensure_proprietary_flow_df(ticker, cache)
    if _refresh_enabled(config, "macro", refresh_all):
        refresh_macro_factor_cache(Path("config/macro_factors.yaml"), _path(config, "macro_cache", "out/macro_factors"), max_age_hours=24)


def build_data_hub(config_path: Path, output_dir: Path | None = None, *, refresh_all: bool = False) -> Dict[str, object]:
    config = load_config(config_path)
    tickers = [_normalise_ticker(ticker) for ticker in config.get("tickers", []) if _normalise_ticker(ticker)]
    output = output_dir or _path(config, "output_dir", str(DEFAULT_OUTPUT_DIR))
    daily_dir = _path(config, "daily_cache", "out/data")
    intraday_dir = _path(config, "intraday_cache", "out/data/intraday_1m")
    depth_dir = _path(config, "depth_cache", "out/data/depth_snapshots")
    _clean_dir(output)
    refresh_sources(config, tickers, refresh_all=refresh_all)

    (output / "daily").mkdir(parents=True, exist_ok=True)
    (output / "intraday").mkdir(parents=True, exist_ok=True)
    latest_rows = []
    files: Dict[str, object] = {}
    for ticker in tickers:
        daily = enrich_daily_frame(_read_daily(daily_dir / f"{ticker}_daily.csv", ticker))
        if not daily.empty:
            recent_daily = daily.tail(_window(config, "recent_daily_rows", 260)).copy()
            recent_daily.to_csv(output / "daily" / f"{ticker}.csv", index=False)
            row = {"Ticker": ticker, **latest_daily_metrics(daily)}
        else:
            row = {"Ticker": ticker}
        intraday = _read_intraday(intraday_dir / f"{ticker}_1m.csv", ticker)
        if not intraday.empty:
            intraday.tail(_window(config, "recent_intraday_rows", 390)).to_csv(output / "intraday" / f"{ticker}.csv", index=False)
            row.update(intraday_summary(intraday, row.get("ADV20Shares")))
        latest_rows.append(row)

    latest = pd.DataFrame(latest_rows)
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

    latest.to_csv(output / "latest_metrics.csv", index=False)
    pd.DataFrame(API_CATALOG).to_csv(output / "api_catalog.csv", index=False)
    pd.DataFrame({"Ticker": tickers}).to_csv(output / "tickers.csv", index=False)
    files.update(
        {
            "latest_metrics": "latest_metrics.csv",
            "api_catalog": "api_catalog.csv",
            "tickers": "tickers.csv",
            "daily_dir": "daily/",
            "intraday_dir": "intraday/",
        }
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "tickers": tickers,
        "purpose": "Numeric-only market data hub for fast ChatGPT browsing. No news, no recommendations, no model forecasts.",
        "read_order_for_chatgpt": [
            "manifest.json",
            "latest_metrics.csv",
            "api_catalog.csv",
            "daily/{ticker}.csv",
            "intraday/{ticker}.csv",
            "depth/latest_depth.csv if present",
            "fundamentals/vietstock_overview.csv if present",
            "fundamentals/vietstock_bctt_latest.csv if present",
            "flows/cafef_flows.csv if present",
            "macro/latest_macro.csv if present",
        ],
        "files": files,
        "api_catalog": API_CATALOG,
    }
    (output / "manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "\n".join(
            [
                "# Numeric Data Hub",
                "",
                "This directory is optimized for ChatGPT browsing.",
                "",
                "- Numeric data only: prices, volume, order book, flows, fundamentals, macro caches.",
                "- No news, no recommendations, no model forecast.",
                "- Start with `manifest.json`, then `latest_metrics.csv`, then per-ticker files under `daily/` and `intraday/`.",
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
