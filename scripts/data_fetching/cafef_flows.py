#!/usr/bin/env python3
"""Fetch and cache CafeF foreign/proprietary flow tables, then derive metrics."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://cafef.vn"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
FOREIGN_ENDPOINT = (
    "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/GDKhoiNgoai.ashx"
)
PROPRIETARY_ENDPOINT = (
    "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/GDTuDoanh.ashx"
)
HORIZONS = (1, 5, 20)
BILLION_TO_KVND = 1_000_000.0
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.0


class CafeFFlowError(RuntimeError):
    """Raised when CafeF flow data cannot be fetched."""


def _normalized_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def _parse_date(value: str) -> str:
    text = (value or "").strip()
    try:
        return datetime.strptime(text, "%d/%m/%Y").date().isoformat()
    except ValueError:
        return text


def _fetch_json(url: str, params: Dict[str, object], context: str) -> Dict[str, object]:
    headers = {"User-Agent": USER_AGENT, "Referer": BASE_URL}
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            last_exc = exc
            LOGGER.warning(
                "CafeF %s request timed out on attempt %d/%d (timeout=%ss)",
                context,
                attempt,
                MAX_RETRIES,
                REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_exc = exc
            LOGGER.warning(
                "CafeF %s request failed on attempt %d/%d: %s",
                context,
                attempt,
                MAX_RETRIES,
                exc,
            )
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)

    if isinstance(last_exc, requests.Timeout):
        raise CafeFFlowError(
            f"Timed out while fetching CafeF flow data ({context}) after "
            f"{MAX_RETRIES} attempts (timeout={REQUEST_TIMEOUT_SECONDS}s each). "
            "Check network/CafeF availability or set data.cafef_flow_enabled=false in "
            "config/data_engine.yaml to skip CafeF flow metrics."
        ) from last_exc
    raise CafeFFlowError(
        f"Failed to fetch CafeF flow data ({context}) after {MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def _fetch_foreign_flow_df(ticker: str, page_size: int = 200) -> pd.DataFrame:
    params = {
        "Symbol": ticker.upper(),
        "StartDate": "",
        "EndDate": "",
        "PageIndex": 1,
        "PageSize": page_size,
    }
    data = _fetch_json(FOREIGN_ENDPOINT, params, f"foreign/{ticker.upper()}")
    payload = data.get("Data") or {}
    entries = payload.get("Data") or []
    rows: List[Dict[str, object]] = []
    for entry in entries:
        buy_value = entry.get("GtMua") or 0
        sell_value = entry.get("GtBan") or 0
        rows.append(
            {
                "Date": _parse_date(entry.get("Ngay", "")),
                "NetShares": entry.get("KLGDRong"),
                "NetValue_billion": (entry.get("GTDGRong") or 0) / 1_000_000_000,
                "BuyShares": entry.get("KLMua"),
                "BuyValue_billion": buy_value / 1_000_000_000,
                "SellShares": entry.get("KLBan"),
                "SellValue_billion": sell_value / 1_000_000_000,
                "RoomRemainingShares": entry.get("RoomConLai"),
                "ForeignHoldingPct": entry.get("DangSoHuu"),
            }
        )
    return pd.DataFrame(rows)


def _fetch_proprietary_flow_df(ticker: str, page_size: int = 200) -> pd.DataFrame:
    params = {
        "Symbol": ticker.upper(),
        "StartDate": "",
        "EndDate": "",
        "PageIndex": 1,
        "PageSize": page_size,
    }
    data = _fetch_json(PROPRIETARY_ENDPOINT, params, f"proprietary/{ticker.upper()}")
    payload = data.get("Data") or {}
    entries = (payload.get("Data") or {}).get("ListDataTudoanh") or []
    rows: List[Dict[str, object]] = []
    for entry in entries:
        buy_shares = entry.get("KLcpMua") or 0
        sell_shares = entry.get("KlcpBan") or 0
        buy_value = entry.get("GtMua") or 0
        sell_value = entry.get("GtBan") or 0
        rows.append(
            {
                "Date": _parse_date(entry.get("Date", "")),
                "BuyShares": buy_shares,
                "BuyValue_billion": buy_value / 1_000_000_000,
                "SellShares": sell_shares,
                "SellValue_billion": sell_value / 1_000_000_000,
                "NetShares": buy_shares - sell_shares,
                "NetValue_billion": (buy_value - sell_value) / 1_000_000_000,
            }
        )
    return pd.DataFrame(rows)


FlowKind = Literal["foreign", "proprietary"]


@dataclass
class CafeFFlowCache:
    base_dir: Path
    # When max_age_hours <= 0, caches are treated as non-expiring
    # and will only be refreshed if missing.
    max_age_hours: int = 4

    def path_for(self, kind: FlowKind, ticker: str) -> Path:
        return self.base_dir / kind / f"{_normalized_ticker(ticker)}.csv"

    def is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.max_age_hours <= 0:
            return True
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age <= timedelta(hours=self.max_age_hours)


def _ensure_flow(kind: FlowKind, ticker: str, cache: CafeFFlowCache) -> pd.DataFrame:
    path = cache.path_for(kind, ticker)
    if cache.is_fresh(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read CafeF %s cache for %s: %s", kind, ticker, exc)
    if kind == "foreign":
        df = _fetch_foreign_flow_df(ticker)
    else:
        df = _fetch_proprietary_flow_df(ticker)
    if df.empty:
        LOGGER.warning("CafeF returned empty %s dataset for %s", kind, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def ensure_foreign_flow_df(ticker: str, cache: CafeFFlowCache) -> pd.DataFrame:
    return _ensure_flow("foreign", ticker, cache)


def ensure_proprietary_flow_df(ticker: str, cache: CafeFFlowCache) -> pd.DataFrame:
    return _ensure_flow("proprietary", ticker, cache)


def _order_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df
    ordered = df.copy()
    ordered["_parsed_date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.sort_values(["_parsed_date", "Date"], ascending=False).drop(columns=["_parsed_date"])
    return ordered.reset_index(drop=True)


def summarize_flow_metrics(
    ticker: str,
    foreign_df: pd.DataFrame,
    proprietary_df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
) -> Dict[str, float | str]:
    result: Dict[str, float | str] = {"Ticker": _normalized_ticker(ticker)}

    if not foreign_df.empty:
        ordered = _order_by_date(foreign_df)
        latest = ordered.iloc[0]
        result["ForeignFlowDate"] = latest.get("Date")
        result["ForeignRoomRemaining_shares"] = float(latest.get("RoomRemainingShares", float("nan")))
        result["ForeignHoldingPct"] = float(latest.get("ForeignHoldingPct", float("nan")))
        for horizon in horizons:
            head = ordered.head(horizon)
            result[f"NetBuySellForeign_shares_{horizon}d"] = float(
                pd.to_numeric(head.get("NetShares"), errors="coerce").sum()
            )
            values = pd.to_numeric(head.get("NetValue_billion"), errors="coerce") * BILLION_TO_KVND
            result[f"NetBuySellForeign_kVND_{horizon}d"] = float(values.sum())
    else:
        for horizon in horizons:
            result[f"NetBuySellForeign_shares_{horizon}d"] = float("nan")
            result[f"NetBuySellForeign_kVND_{horizon}d"] = float("nan")
        result["ForeignFlowDate"] = ""
        result["ForeignRoomRemaining_shares"] = float("nan")
        result["ForeignHoldingPct"] = float("nan")

    if not proprietary_df.empty:
        ordered = _order_by_date(proprietary_df)
        latest = ordered.iloc[0]
        result["ProprietaryFlowDate"] = latest.get("Date")
        for horizon in horizons:
            head = ordered.head(horizon)
            result[f"NetBuySellProprietary_shares_{horizon}d"] = float(
                pd.to_numeric(head.get("NetShares"), errors="coerce").sum()
            )
            values = pd.to_numeric(head.get("NetValue_billion"), errors="coerce") * BILLION_TO_KVND
            result[f"NetBuySellProprietary_kVND_{horizon}d"] = float(values.sum())
    else:
        result["ProprietaryFlowDate"] = ""
        for horizon in horizons:
            result[f"NetBuySellProprietary_shares_{horizon}d"] = float("nan")
            result[f"NetBuySellProprietary_kVND_{horizon}d"] = float("nan")

    return result


def build_flow_feature_frame(
    tickers: List[str],
    cache_dir: Path,
    max_age_hours: int,
) -> pd.DataFrame:
    clean_tickers = [_normalized_ticker(t) for t in tickers if str(t).strip()]
    if not clean_tickers:
        return pd.DataFrame(columns=["Ticker"])
    cache = CafeFFlowCache(cache_dir, max_age_hours=max_age_hours)
    rows: List[Dict[str, float | str]] = []
    for ticker in clean_tickers:
        foreign_df = ensure_foreign_flow_df(ticker, cache)
        proprietary_df = ensure_proprietary_flow_df(ticker, cache)
        rows.append(summarize_flow_metrics(ticker, foreign_df, proprietary_df))
    return pd.DataFrame(rows)


__all__ = [
    "CafeFFlowError",
    "CafeFFlowCache",
    "build_flow_feature_frame",
    "ensure_foreign_flow_df",
    "ensure_proprietary_flow_df",
    "parse_foreign_flow_html",
    "parse_proprietary_flow_html",
    "summarize_flow_metrics",
]
