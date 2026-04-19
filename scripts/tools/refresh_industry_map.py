from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_fetching.market_members import fetch_hose_members, fetch_vn30_members
from scripts.data_fetching.vietstock_industry import (
    fetch_vietstock_sector_levels,
    select_vietstock_sector,
)


VN_TIME_SERIES_INDEX_RE = re.compile(r"VNINDEX|VN30|VN100", re.IGNORECASE)
INVESTING_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)


class RefreshError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndustryRow:
    ticker: str
    sector: str


def _repo_root() -> Path:
    # scripts/tools/<file>.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _read_sector_lookup(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RefreshError(f"CSV has no header: {path}")
            for col in ("Ticker", "Sector"):
                if col not in reader.fieldnames:
                    raise RefreshError(f"CSV must contain columns Ticker,Sector: {path}")
            out: dict[str, str] = {}
            for row in reader:
                ticker = (row.get("Ticker") or "").strip().upper()
                if not ticker:
                    continue
                out[ticker] = (row.get("Sector") or "").strip()
            return out
    except RefreshError:
        raise
    except Exception as exc:
        raise RefreshError(f"Failed reading CSV: {path}: {exc}") from exc


def _read_rows_from_csv(path: Path, *, include_indices: bool) -> list[IndustryRow]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RefreshError(f"CSV has no header: {path}")
            if "Ticker" not in reader.fieldnames:
                raise RefreshError(f"CSV must contain 'Ticker' column: {path}")
            has_sector = "Sector" in reader.fieldnames
            rows: list[IndustryRow] = []
            for r in reader:
                ticker = (r.get("Ticker") or "").strip().upper()
                if not ticker:
                    continue
                if not include_indices and VN_TIME_SERIES_INDEX_RE.search(ticker):
                    continue
                sector = (r.get("Sector") or "").strip() if has_sector else ""
                rows.append(IndustryRow(ticker=ticker, sector=sector))
            return rows
    except RefreshError:
        raise
    except Exception as exc:
        raise RefreshError(f"Failed reading CSV: {path}: {exc}") from exc


def _read_tickers_from_csv(path: Path) -> list[str]:
    return [row.ticker for row in _read_rows_from_csv(path, include_indices=True)]


def _fetch_investing_vn100_members(*, timeout: int = 30) -> set[str]:
    url = "https://vn.investing.com/indices/vn100-components"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
    except Exception as exc:
        raise RefreshError(f"Failed fetching VN100 constituents from {url}: {exc}") from exc

    html = resp.text or ""
    match = INVESTING_NEXT_DATA_RE.search(html)
    if not match:
        raise RefreshError(
            "Investing.com page missing __NEXT_DATA__; the site may be blocking automated requests."
        )
    try:
        payload = json.loads(match.group("json"))
    except Exception as exc:
        raise RefreshError(f"Failed parsing Investing.com __NEXT_DATA__ JSON: {exc}") from exc

    collection = (
        payload.get("props", {})
        .get("pageProps", {})
        .get("state", {})
        .get("assetsCollectionStore", {})
        .get("assetsCollection", {})
        .get("_collection")
    )
    if not isinstance(collection, list) or not collection:
        raise RefreshError("Investing.com payload missing assets collection for VN100.")

    tickers: set[str] = set()
    for item in collection:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol") or item.get("ticker")
        if not isinstance(sym, str):
            continue
        normalized = sym.strip().upper()
        if normalized:
            tickers.add(normalized)
    if not tickers:
        raise RefreshError("No tickers extracted from Investing.com assets collection.")
    return tickers


def _merge_sectors(
    tickers: Iterable[str], *, sector_lookup: dict[str, str], default_sector: str
) -> list[IndustryRow]:
    rows: list[IndustryRow] = []
    for t in tickers:
        ticker = t.strip().upper()
        if not ticker:
            continue
        sector = sector_lookup.get(ticker, default_sector)
        rows.append(IndustryRow(ticker=ticker, sector=sector))
    return rows


def _apply_nvl_override(sector_lookup: dict[str, str], default_sector: str) -> None:
    """Ensure NVL is included and sectorized deterministically.

    User intent: always include NVL (Novaland) alongside VN100 and set its sector
    by copying a major real-estate ticker's sector (prefer VHM).
    """

    for source in ("VHM", "VIC"):
        if source in sector_lookup and sector_lookup[source].strip():
            sector_lookup["NVL"] = sector_lookup[source].strip()
            return
    sector_lookup["NVL"] = default_sector


def _normalize_tickers(tickers: Iterable[str], *, include_indices: bool) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker:
            continue
        if not include_indices and VN_TIME_SERIES_INDEX_RE.search(ticker):
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


def _has_sector_lookup(sector: str, *, default_sector: str) -> bool:
    value = sector.strip()
    return bool(value) and value != default_sector


def _split_cached_and_missing_tickers(
    tickers: Iterable[str],
    *,
    existing_lookup: dict[str, str],
    include_indices: bool,
    default_sector: str,
    refresh_existing: bool,
) -> tuple[list[IndustryRow], list[str]]:
    cached_rows: list[IndustryRow] = []
    missing: list[str] = []
    for ticker in _normalize_tickers(tickers, include_indices=include_indices):
        cached_sector = existing_lookup.get(ticker, "")
        if not refresh_existing and _has_sector_lookup(cached_sector, default_sector=default_sector):
            cached_rows.append(IndustryRow(ticker=ticker, sector=cached_sector))
            continue
        missing.append(ticker)
    return cached_rows, missing


def _fetch_rows_from_vietstock_profiles(
    tickers: Iterable[str],
    *,
    include_indices: bool,
    sector_level: str,
    timeout: int,
    pause_seconds: float,
) -> list[IndustryRow]:
    rows: list[IndustryRow] = []
    normalized = _normalize_tickers(tickers, include_indices=include_indices)
    with requests.Session() as session:
        for idx, ticker in enumerate(normalized):
            try:
                levels = fetch_vietstock_sector_levels(ticker, session=session, timeout=timeout)
            except Exception as exc:
                raise RefreshError(f"Failed fetching Vietstock sector for {ticker}: {exc}") from exc
            rows.append(IndustryRow(ticker=ticker, sector=select_vietstock_sector(levels, sector_level)))
            if pause_seconds > 0 and idx < len(normalized) - 1:
                time.sleep(pause_seconds)
    return rows


def _validate_rows(rows: list[IndustryRow]) -> None:
    if not rows:
        raise RefreshError("No tickers after filtering; aborting.")
    tickers = [r.ticker for r in rows]
    dupes = sorted({t for t in tickers if tickers.count(t) > 1})
    if dupes:
        raise RefreshError(f"Duplicate tickers detected: {', '.join(dupes[:20])}")


def _write_industry_map(rows: list[IndustryRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ticker", "Sector"], lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({"Ticker": r.ticker, "Sector": r.sector})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh data/industry_map.csv (Ticker,Sector) from CSVs, VN100 membership, or Vietstock company profiles."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-tickers-csv",
        "--from-universe-csv",
        dest="from_tickers_csv",
        metavar="PATH",
        help="Read tickers (and optional Sector) from CSV.",
    )
    src.add_argument(
        "--from-investing-vn100",
        action="store_true",
        help="Fetch VN100 constituents from Investing.com (requires network).",
    )
    src.add_argument(
        "--from-live-vn100-portfolio-profiles",
        action="store_true",
        help="Fetch live VN100 members, merge portfolio/extra tickers, then fetch missing sectors from Vietstock profiles.",
    )
    src.add_argument(
        "--from-vietstock-profiles-csv",
        metavar="PATH",
        help="Read tickers from CSV, then fetch each ticker's sector from Vietstock company profile pages.",
    )
    src.add_argument(
        "--from-vietstock-profiles-hose",
        action="store_true",
        help="Fetch the live HOSE ticker list, then fetch each ticker's sector from Vietstock company profile pages.",
    )
    src.add_argument(
        "--from-vietstock-profiles-vn30",
        action="store_true",
        help="Fetch the live VN30 ticker list, then fetch each ticker's sector from Vietstock company profile pages.",
    )
    parser.add_argument(
        "--merge-sectors-from",
        metavar="PATH",
        help="CSV providing a Sector mapping (Ticker,Sector). Used when source lacks sectors.",
    )
    parser.add_argument(
        "--portfolio-csv",
        metavar="PATH",
        help="Optional portfolio CSV whose Ticker column will be merged into the live VN100 set.",
    )
    parser.add_argument(
        "--extra-ticker",
        action="append",
        default=[],
        help="Extra ticker to force-include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--default-sector",
        default="Không rõ",
        help="Sector value for tickers missing in the merge mapping (default: %(default)s).",
    )
    parser.add_argument(
        "--include-indices",
        action="store_true",
        help="Keep VNINDEX/VN30/VN100 rows if present in source CSV.",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=100,
        help="When fetching VN100, fail if count != this value (default: %(default)s).",
    )
    parser.add_argument(
        "--sector-level",
        choices=("top", "leaf", "path"),
        default="top",
        help="Which Vietstock sector level to write when using company-profile fetching (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds for network fetches (default: %(default)s).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Sleep between Vietstock company-profile requests to throttle manual refresh runs.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="When using Vietstock company-profile fetching, ignore cached sectors already present in the output CSV and fetch them again.",
    )
    parser.add_argument(
        "--output",
        default="data/industry_map.csv",
        help="Output CSV path (relative to repo root by default).",
    )
    args = parser.parse_args(argv)

    out_path = _resolve_path(args.output)
    sector_lookup = _read_sector_lookup(_resolve_path(args.merge_sectors_from)) if args.merge_sectors_from else {}
    existing_output_lookup = _read_sector_lookup(out_path) if out_path.exists() else {}

    if args.from_tickers_csv:
        source_path = _resolve_path(args.from_tickers_csv)
        rows = _read_rows_from_csv(source_path, include_indices=bool(args.include_indices))
        if rows and all(not r.sector for r in rows) and sector_lookup:
            rows = _merge_sectors((r.ticker for r in rows), sector_lookup=sector_lookup, default_sector=args.default_sector)
    elif args.from_live_vn100_portfolio_profiles:
        tickers = set(_fetch_investing_vn100_members(timeout=args.timeout))
        if args.expect_count and len(tickers) != args.expect_count:
            raise RefreshError(f"Expected {args.expect_count} tickers, got {len(tickers)} from Investing.com VN100.")
        if args.portfolio_csv:
            tickers.update(_read_tickers_from_csv(_resolve_path(args.portfolio_csv)))
        tickers.update(str(ticker).strip().upper() for ticker in args.extra_ticker if str(ticker).strip())
        cached_rows, missing_tickers = _split_cached_and_missing_tickers(
            tickers,
            existing_lookup=existing_output_lookup,
            include_indices=bool(args.include_indices),
            default_sector=args.default_sector,
            refresh_existing=bool(args.refresh_existing),
        )
        rows = cached_rows + _fetch_rows_from_vietstock_profiles(
            missing_tickers,
            include_indices=True,
            sector_level=args.sector_level,
            timeout=args.timeout,
            pause_seconds=args.pause_seconds,
        )
    elif args.from_vietstock_profiles_csv:
        source_path = _resolve_path(args.from_vietstock_profiles_csv)
        input_rows = _read_rows_from_csv(source_path, include_indices=bool(args.include_indices))
        cached_rows, missing_tickers = _split_cached_and_missing_tickers(
            (r.ticker for r in input_rows),
            existing_lookup=existing_output_lookup,
            include_indices=bool(args.include_indices),
            default_sector=args.default_sector,
            refresh_existing=bool(args.refresh_existing),
        )
        rows = cached_rows + _fetch_rows_from_vietstock_profiles(
            missing_tickers,
            include_indices=True,
            sector_level=args.sector_level,
            timeout=args.timeout,
            pause_seconds=args.pause_seconds,
        )
    elif args.from_vietstock_profiles_hose:
        tickers = sorted(fetch_hose_members(timeout=args.timeout))
        cached_rows, missing_tickers = _split_cached_and_missing_tickers(
            tickers,
            existing_lookup=existing_output_lookup,
            include_indices=bool(args.include_indices),
            default_sector=args.default_sector,
            refresh_existing=bool(args.refresh_existing),
        )
        rows = cached_rows + _fetch_rows_from_vietstock_profiles(
            missing_tickers,
            include_indices=True,
            sector_level=args.sector_level,
            timeout=args.timeout,
            pause_seconds=args.pause_seconds,
        )
    elif args.from_vietstock_profiles_vn30:
        tickers = sorted(fetch_vn30_members(timeout=args.timeout))
        cached_rows, missing_tickers = _split_cached_and_missing_tickers(
            tickers,
            existing_lookup=existing_output_lookup,
            include_indices=bool(args.include_indices),
            default_sector=args.default_sector,
            refresh_existing=bool(args.refresh_existing),
        )
        rows = cached_rows + _fetch_rows_from_vietstock_profiles(
            missing_tickers,
            include_indices=True,
            sector_level=args.sector_level,
            timeout=args.timeout,
            pause_seconds=args.pause_seconds,
        )
    else:
        tickers = _fetch_investing_vn100_members()
        if args.expect_count and len(tickers) != args.expect_count:
            raise RefreshError(f"Expected {args.expect_count} tickers, got {len(tickers)} from Investing.com VN100.")

        # Always include NVL alongside VN100 (even if Investing's constituent list omits it)
        _apply_nvl_override(sector_lookup, args.default_sector)
        tickers.add("NVL")
        rows = _merge_sectors(sorted(tickers), sector_lookup=sector_lookup, default_sector=args.default_sector)

    rows = sorted(rows, key=lambda r: r.ticker)
    _validate_rows(rows)
    _write_industry_map(rows, out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
