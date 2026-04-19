from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from scripts.data_fetching.vietstock_bctt_api import load_or_fetch_bctt_feature_frame


DEFAULT_INDUSTRY_MAP = Path("data/industry_map.csv")
DEFAULT_PORTFOLIO = Path("data/portfolios/portfolio.csv")
DEFAULT_CACHE_DIR = Path("out/vietstock_bctt")
DEFAULT_SUMMARY_OUT = Path("out/analysis/vietstock_bctt_cache_summary.csv")


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _load_tickers(industry_map_path: Path, portfolio_path: Path, explicit_tickers: Sequence[str]) -> List[str]:
    tickers = {_normalise_ticker(ticker) for ticker in explicit_tickers if _normalise_ticker(ticker)}
    if industry_map_path.exists():
        industry_df = pd.read_csv(industry_map_path)
        if "Ticker" in industry_df.columns:
            tickers.update(_normalise_ticker(ticker) for ticker in industry_df["Ticker"])
    if portfolio_path.exists():
        portfolio_df = pd.read_csv(portfolio_path)
        if "Ticker" in portfolio_df.columns:
            tickers.update(_normalise_ticker(ticker) for ticker in portfolio_df["Ticker"])
    return sorted(ticker for ticker in tickers if ticker and ticker != "VNINDEX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh/cache Vietstock BCTT tables for a ticker universe.")
    parser.add_argument("--industry-map", default=str(DEFAULT_INDUSTRY_MAP), help="Ticker universe CSV.")
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO), help="Optional portfolio CSV to merge into the ticker set.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="BCTT cache directory.")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT), help="Summary CSV output path.")
    parser.add_argument("--ticker", action="append", default=[], help="Optional extra ticker(s) to include.")
    parser.add_argument("--max-age-hours", type=int, default=720, help="Reuse cache newer than this many hours. 0 means trust existing cache forever.")
    parser.add_argument("--headed", action="store_true", help="Run Playwright in headed mode while fetching missing tickers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = _load_tickers(Path(args.industry_map), Path(args.portfolio), args.ticker)
    if not tickers:
        raise SystemExit("No tickers found for BCTT cache refresh.")

    summary_rows = []
    success_count = 0
    for ticker in tickers:
        error = ""
        try:
            quarterly = load_or_fetch_bctt_feature_frame(
                tickers=[ticker],
                cache_dir=Path(args.cache_dir),
                max_age_hours=args.max_age_hours,
                headless=not args.headed,
            )
        except Exception as exc:
            quarterly = pd.DataFrame()
            error = str(exc)

        if quarterly.empty:
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "QuarterCount": 0,
                    "LatestPeriodEnd": "",
                    "LatestAvailableDate": "",
                    "Error": error,
                }
            )
            continue

        success_count += 1
        latest_row = quarterly.sort_values("PeriodEnd").iloc[-1]
        summary_rows.append(
            {
                "Ticker": ticker,
                "QuarterCount": int(len(quarterly)),
                "LatestPeriodEnd": str(pd.to_datetime(latest_row["PeriodEnd"]).date()),
                "LatestAvailableDate": str(pd.to_datetime(latest_row["AvailableDate"]).date()),
                "Error": error,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("Ticker").reset_index(drop=True)
    if success_count == 0:
        raise SystemExit("No BCTT data available after refresh.")

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    payload = {
        "ticker_count": int(len(summary)),
        "success_count": int(success_count),
        "cache_dir": str(Path(args.cache_dir)),
        "summary_csv": str(summary_out),
        "latest_available_date_max": str(summary["LatestAvailableDate"].replace("", pd.NA).dropna().max() if not summary.empty else ""),
    }
    summary_out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
