"""Fetch VNDIRECT priceboard depth snapshots and merge them into intraday caches."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests


VN_TZ = timezone(timedelta(hours=7))
PRICE_STREAMING_URL = "https://price-streaming-api-free.vndirect.com.vn/v2/stocks/snapshot"
DEPTH_LEVELS = range(1, 11)
SFU_STOCK_FIELDS = [
    "code",
    "stockType",
    "floorCode",
    "basicPrice",
    "floorPrice",
    "ceilingPrice",
    *[f"bidPrice{level:02d}" for level in DEPTH_LEVELS],
    *[f"bidQtty{level:02d}" for level in DEPTH_LEVELS],
    *[f"offerPrice{level:02d}" for level in DEPTH_LEVELS],
    *[f"offerQtty{level:02d}" for level in DEPTH_LEVELS],
    "buyForeignQtty",
    "sellForeignQtty",
    "highestPrice",
    "lowestPrice",
    "accumulatedVal",
    "accumulatedVol",
    "matchPrice",
    "matchQtty",
    "currentPrice",
    "currentQtty",
    "totalRoom",
    "currentRoom",
    "inav",
    "underlyingAsset",
    "issuer",
    "exercisePrice",
    "exerciseRatio",
    "expiryDate",
    "time",
    "bv4",
    "sv4",
]


def _normalise_ticker(value: object) -> str:
    return str(value).strip().upper()


def _decode_priceboard_payload(encoded: str) -> List[str]:
    return "".join(chr(ord(char) + (index % 5)) for index, char in enumerate(str(encoded))).split("|")


def _to_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def _decode_stock_snapshot(encoded: str) -> Dict[str, object]:
    parts = _decode_priceboard_payload(encoded)
    if not parts:
        return {}
    payload_type = parts[0]
    values = parts[1:]
    if payload_type != "SFU":
        return {}
    stock_type = values[1] if len(values) > 1 else ""
    if stock_type != "ST":
        return {}
    return dict(zip(SFU_STOCK_FIELDS, values))


def fetch_depth_snapshots(tickers: Iterable[str]) -> pd.DataFrame:
    codes = [_normalise_ticker(ticker) for ticker in tickers if _normalise_ticker(ticker)]
    if not codes:
        return pd.DataFrame()
    response = requests.get(
        PRICE_STREAMING_URL,
        params={"codes": ",".join(codes)},
        timeout=15,
        headers={
            "Accept": "application/json",
            "Referer": "https://priceboard.vndirect.com.vn/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    fetched_at = datetime.now(VN_TZ)
    for encoded in payload:
        decoded = _decode_stock_snapshot(str(encoded))
        code = _normalise_ticker(str(decoded.get("code", "")))
        if not code:
            continue
        row: Dict[str, object] = {
            "Ticker": code,
            "FetchedAt": fetched_at.isoformat(),
            "PriceboardTime": str(decoded.get("time", "")),
            "FloorCode": str(decoded.get("floorCode", "")),
            "BasicPrice": _to_number(decoded.get("basicPrice")),
            "FloorPrice": _to_number(decoded.get("floorPrice")),
            "CeilingPrice": _to_number(decoded.get("ceilingPrice")),
            "HighestPrice": _to_number(decoded.get("highestPrice")),
            "LowestPrice": _to_number(decoded.get("lowestPrice")),
            "AccumulatedValue": _to_number(decoded.get("accumulatedVal")),
            "AccumulatedVolume": _to_number(decoded.get("accumulatedVol")),
            "MatchPrice": _to_number(decoded.get("matchPrice")),
            "MatchVolume": _to_number(decoded.get("matchQtty")),
            "CurrentPrice": _to_number(decoded.get("currentPrice")),
            "CurrentVolume": _to_number(decoded.get("currentQtty")),
            "ForeignBuyVolume": _to_number(decoded.get("buyForeignQtty")),
            "ForeignSellVolume": _to_number(decoded.get("sellForeignQtty")),
            "ForeignNetVolume": _to_number(decoded.get("buyForeignQtty")) - _to_number(decoded.get("sellForeignQtty")),
            "ForeignTotalRoom": _to_number(decoded.get("totalRoom")),
            "ForeignCurrentRoom": _to_number(decoded.get("currentRoom")),
        }
        for level in DEPTH_LEVELS:
            row[f"BestBid{level}"] = _to_number(decoded.get(f"bidPrice{level:02d}"))
            row[f"BestAsk{level}"] = _to_number(decoded.get(f"offerPrice{level:02d}"))
            row[f"BidVolume{level}"] = _to_number(decoded.get(f"bidQtty{level:02d}"))
            row[f"AskVolume{level}"] = _to_number(decoded.get(f"offerQtty{level:02d}"))
        rows.append(row)
    return pd.DataFrame(rows)


def append_depth_snapshot_cache(depth_df: pd.DataFrame, depth_dir: Path, retention_days: int = 30) -> None:
    if depth_df.empty:
        return
    depth_dir.mkdir(parents=True, exist_ok=True)
    for ticker, scoped in depth_df.groupby("Ticker", sort=False):
        path = depth_dir / f"{_normalise_ticker(ticker)}_depth.csv"
        if path.exists():
            existing = pd.read_csv(path)
            merged = pd.concat([existing, scoped], ignore_index=True)
            merged = merged.drop_duplicates(subset=["Ticker", "FetchedAt"], keep="last")
        else:
            merged = scoped.copy()
        fetched_at = pd.to_datetime(merged["FetchedAt"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(retention_days)))
        retained = merged.loc[fetched_at >= cutoff].copy()
        if not retained.empty:
            merged = retained
        merged = merged.sort_values("FetchedAt").reset_index(drop=True)
        merged.to_csv(path, index=False)


def merge_depth_into_intraday_cache(depth_df: pd.DataFrame, history_dir: Path, resolution: str) -> None:
    if depth_df.empty:
        return
    resolution_token = str(resolution).strip()
    for row in depth_df.itertuples(index=False):
        ticker = _normalise_ticker(getattr(row, "Ticker"))
        path = history_dir / f"{ticker}_{resolution_token}m.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "t" not in frame.columns:
            continue
        ts = pd.to_datetime(pd.to_numeric(frame["t"], errors="coerce"), unit="s", utc=True, errors="coerce")
        frame["_Timestamp"] = ts.dt.tz_convert(VN_TZ)
        fetched_at = pd.Timestamp(getattr(row, "FetchedAt")).tz_convert(VN_TZ)
        same_date = frame["_Timestamp"].dt.date == fetched_at.date()
        if same_date.any():
            target_idx = frame.loc[same_date, "_Timestamp"].idxmax()
        else:
            target_idx = frame["_Timestamp"].idxmax()
        frame.loc[target_idx, "best_bid"] = getattr(row, "BestBid1")
        frame.loc[target_idx, "best_ask"] = getattr(row, "BestAsk1")
        for level in range(1, 4):
            frame.loc[target_idx, f"bid_volume_{level}"] = getattr(row, f"BidVolume{level}")
            frame.loc[target_idx, f"ask_volume_{level}"] = getattr(row, f"AskVolume{level}")
        frame = frame.drop(columns=["_Timestamp"])
        frame.to_csv(path, index=False)


def refresh_depth_for_intraday_cache(
    tickers: Iterable[str],
    history_dir: Path,
    *,
    resolution: str,
    depth_dir: Path | None = None,
    retention_days: int = 30,
) -> pd.DataFrame:
    depth_df = fetch_depth_snapshots(tickers)
    if depth_dir is not None:
        append_depth_snapshot_cache(depth_df, depth_dir, retention_days=retention_days)
    merge_depth_into_intraday_cache(depth_df, history_dir, resolution)
    return depth_df
