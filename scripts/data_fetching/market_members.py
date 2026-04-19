from __future__ import annotations

import json
import logging
import re

import requests


LOGGER = logging.getLogger(__name__)

FALLBACK_VN30_MEMBERS: set[str] = {
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
}

BOARD_SYMBOL_RE = re.compile(r'data-symbol="([A-Z0-9]{3,})"')
INVESTING_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)


def parse_vietstock_board_symbols(html: str) -> set[str]:
    return {symbol.upper() for symbol in BOARD_SYMBOL_RE.findall(html or "")}


def fetch_vietstock_board_symbols(board: str, timeout: int = 15) -> set[str]:
    url = f"https://banggia.vietstock.vn/bang-gia/{board}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    symbols = parse_vietstock_board_symbols(resp.text)
    if not symbols:
        raise RuntimeError(f"No symbols found on Vietstock board page: {url}")
    return symbols


def fetch_vn30_members(timeout: int = 15) -> set[str]:
    try:
        return fetch_vietstock_board_symbols("vn30", timeout=timeout)
    except Exception as exc:  # pragma: no cover - network failure path
        LOGGER.warning(
            "Không lấy được danh sách VN30 động (%s); fallback sang danh sách static.",
            exc,
        )
        return set(FALLBACK_VN30_MEMBERS)


def fetch_hose_members(timeout: int = 15) -> set[str]:
    return fetch_vndirect_stock_codes(query="type:STOCK~floor:HOSE", timeout=timeout)


def extract_vndirect_stock_codes(payload: dict, *, floor: str | None = None) -> set[str]:
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("VNDIRECT stocks payload is empty or malformed")

    codes: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if floor and str(row.get("floor") or "").strip().upper() != floor.upper():
            continue
        code = str(row.get("code") or "").strip().upper()
        if code:
            codes.add(code)
    if not codes:
        raise RuntimeError("No stock codes extracted from VNDIRECT stocks payload")
    return codes


def fetch_vndirect_stock_codes(query: str, timeout: int = 30, size: int = 1000) -> set[str]:
    url = "https://api-finfo.vndirect.com.vn/v4/stocks"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        url,
        params={"q": query, "size": size},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return extract_vndirect_stock_codes(resp.json(), floor="HOSE")


def parse_investing_vn100_members(html: str) -> set[str]:
    match = INVESTING_NEXT_DATA_RE.search(html or "")
    if not match:
        raise RuntimeError("Investing.com page missing __NEXT_DATA__ payload")
    payload = json.loads(match.group("json"))
    collection = (
        payload.get("props", {})
        .get("pageProps", {})
        .get("state", {})
        .get("assetsCollectionStore", {})
        .get("assetsCollection", {})
        .get("_collection")
    )
    if not isinstance(collection, list) or not collection:
        raise RuntimeError("Investing.com payload missing VN100 assets collection")

    symbols: set[str] = set()
    for item in collection:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("ticker")
        if isinstance(symbol, str) and symbol.strip():
            symbols.add(symbol.strip().upper())
    if not symbols:
        raise RuntimeError("No tickers extracted from Investing.com VN100 payload")
    return symbols


def fetch_vn100_members(timeout: int = 30) -> set[str]:
    url = "https://vn.investing.com/indices/vn100-components"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    }
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return parse_investing_vn100_members(resp.text or "")
