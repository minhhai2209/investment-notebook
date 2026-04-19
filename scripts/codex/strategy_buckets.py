#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_COLUMNS = [
    "Ticker",
    "StrategyBucket",
    "AllowNewBuy",
    "AllowAvgDown",
    "TargetState",
]


def _resolve_portfolio_csv(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    portfolio_cfg = config.get("portfolio", {}) or {}
    if not isinstance(portfolio_cfg, dict):
        raise ValueError(f"{config_path} key 'portfolio' must be a mapping")
    config_dir = config_path.parent.resolve()
    repo_root = config_dir.parent
    portfolio_dir = Path(portfolio_cfg.get("directory", "data/portfolios"))
    if portfolio_dir.is_absolute():
        return (portfolio_dir / "portfolio.csv").resolve()

    config_candidate = (config_dir / portfolio_dir).resolve()
    try:
        config_candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{portfolio_dir}' escapes repository root {repo_root}"
        ) from exc
    if config_candidate.exists():
        return config_candidate / "portfolio.csv"

    root_candidate = (repo_root / portfolio_dir).resolve()
    try:
        root_candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{portfolio_dir}' escapes repository root {repo_root}"
        ) from exc
    portfolio_dir = root_candidate
    return portfolio_dir / "portfolio.csv"


def _load_portfolio_tickers(portfolio_csv: Path) -> list[str]:
    if not portfolio_csv.exists():
        return []
    try:
        df = pd.read_csv(portfolio_csv, usecols=["Ticker"])
    except ValueError as exc:
        raise ValueError(f"{portfolio_csv} missing required column 'Ticker'") from exc
    tickers: list[str] = []
    seen: set[str] = set()
    for raw in df["Ticker"].dropna().astype(str).tolist():
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _load_active_ticker_filter(config_path: Path) -> list[str]:
    raw_filter = os.environ.get("INDUSTRY_TICKER_FILTER", "").strip()
    if raw_filter:
        return sorted(
            {
                ticker.strip().upper()
                for ticker in raw_filter.split(",")
                if ticker.strip()
            }
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    universe_cfg = config.get("universe", {}) or {}
    if not isinstance(universe_cfg, dict):
        raise ValueError(f"{config_path} key 'universe' must be a mapping")

    for key in ("core_tickers", "tickers"):
        raw_values = universe_cfg.get(key)
        if raw_values is None:
            continue
        if not isinstance(raw_values, list):
            raise ValueError(f"{config_path} key 'universe.{key}' must be a list")
        return sorted(
            {
                str(value).strip().upper()
                for value in raw_values
                if str(value).strip()
            }
        )
    return []


def build_strategy_buckets(config_path: Path, source_path: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_path} missing required columns: {', '.join(missing)}"
        )

    base = df.loc[:, REQUIRED_COLUMNS].copy()
    base["Ticker"] = base["Ticker"].astype(str).str.strip().str.upper()
    base = base[base["Ticker"] != ""].reset_index(drop=True)
    active_filter = set(_load_active_ticker_filter(config_path))
    if active_filter:
        base = base[base["Ticker"].isin(active_filter)].reset_index(drop=True)
    if base["Ticker"].duplicated().any():
        duplicates = sorted(base.loc[base["Ticker"].duplicated(), "Ticker"].unique().tolist())
        raise ValueError(
            f"{source_path} contains duplicate tickers: {', '.join(duplicates)}"
        )

    portfolio_tickers = _load_portfolio_tickers(_resolve_portfolio_csv(config_path))
    if active_filter:
        portfolio_tickers = [ticker for ticker in portfolio_tickers if ticker in active_filter]
    known = set(base["Ticker"].tolist())
    synthesized_rows = [
        {
            "Ticker": ticker,
            "StrategyBucket": "exit_only",
            "AllowNewBuy": 0,
            "AllowAvgDown": 0,
            "TargetState": "exit_all",
        }
        for ticker in portfolio_tickers
        if ticker not in known
    ]
    if synthesized_rows:
        base = pd.concat([base, pd.DataFrame(synthesized_rows)], ignore_index=True)
    return base.loc[:, REQUIRED_COLUMNS]


def write_strategy_buckets(config_path: Path, source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_strategy_buckets(config_path, source_path).to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Codex strategy buckets from a manual normal-watchlist source plus "
            "live portfolio tickers that should be treated as exit-only."
        )
    )
    parser.add_argument("--config", required=True, help="Path to config/data_engine.yaml")
    parser.add_argument("--source", required=True, help="Path to repo-root strategy_buckets.csv")
    parser.add_argument("--output", required=True, help="Path to output strategy_buckets.csv")
    args = parser.parse_args()

    write_strategy_buckets(
        config_path=Path(args.config).resolve(),
        source_path=Path(args.source).resolve(),
        output_path=Path(args.output).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
