"""Import portfolio CSV exported by the browser extension from Downloads.

Expected input files (created by ../broker-extension on TCBS):

- Location: ``~/Downloads`` (or ``PORTFOLIO_DOWNLOADS_DIR``)
- Pattern: ``tcbs-portfolio-YYYYMMDD-HHMMSS.csv``
- Columns: ``Ticker,Quantity,AvgCost,CostValue`` (CostValue is optional for the engine)

This helper:

1. Locates the most recent matching CSV in the downloads directory.
2. Normalises it to the engine schema ``Ticker,Quantity,AvgPrice``.
3. Writes the result into ``data/portfolios/portfolio.csv`` under the repo root
   (engine contract: thư mục này chỉ có đúng một file `portfolio.csv`).

Fail-fast behaviour:
- If no matching CSV is found, or required columns are missing, or the
  resulting portfolio is empty, the script exits with a clear error message.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger("import_portfolio")


def _repo_root() -> Path:
  cur = Path(__file__).resolve()
  for candidate in [cur.parent, *cur.parents]:
    if (candidate / ".git").exists():
      return candidate
  return Path.cwd()


@dataclass
class ImportConfig:
  downloads_dir: Path
  broker_prefix: str = "tcbs"


def _resolve_downloads_dir(raw: Optional[str]) -> Path:
  if raw:
    path = Path(raw).expanduser()
  else:
    path = Path.home() / "Downloads"
  if not path.exists() or not path.is_dir():
    raise RuntimeError(f"Downloads directory does not exist: {path}")
  return path


def _find_latest_csv(cfg: ImportConfig) -> Path:
  pattern = f"{cfg.broker_prefix}-portfolio-*.csv"
  candidates = sorted(cfg.downloads_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
  if not candidates:
    raise RuntimeError(f"No files matching '{pattern}' found in {cfg.downloads_dir}. "
                       f"Export portfolio from the browser extension first.")
  latest = candidates[0]
  LOGGER.info("Using latest portfolio CSV from Downloads: %s", latest)
  return latest


def _normalise_portfolio(src: Path, portfolios_dir: Path) -> Path:
  df = pd.read_csv(src)
  if df.empty:
    raise RuntimeError(f"Source portfolio is empty: {src}")

  cols = list(df.columns)
  for required in ("Ticker", "Quantity"):
    if required not in cols:
      raise RuntimeError(f"Portfolio {src} missing required column: {required}")

  price_col: Optional[str]
  if "AvgPrice" in cols:
    price_col = "AvgPrice"
  elif "AvgCost" in cols:
    price_col = "AvgCost"
  else:
    raise RuntimeError(f"Portfolio {src} missing required price column: expected 'AvgPrice' or 'AvgCost'")

  df = df[["Ticker", "Quantity", price_col]].copy()
  if price_col != "AvgPrice":
    df = df.rename(columns={price_col: "AvgPrice"})

  df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
  df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
  df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce")

  df = df[(df["Ticker"] != "") & (df["Quantity"] > 0)]
  if df.empty:
    raise RuntimeError(f"Portfolio {src} normalised to zero rows after cleaning")

  portfolios_dir.mkdir(parents=True, exist_ok=True)

  # Safety: dọn các file CSV tcbs-* cũ trong thư mục portfolio để tránh vô tình
  # sử dụng/hiểu nhầm nhiều nguồn danh mục. Engine luôn chỉ đọc portfolio.csv.
  for other in portfolios_dir.glob("tcbs-portfolio-*.csv"):
    try:
      other.unlink()
      LOGGER.info("Removed stale portfolio snapshot: %s", other)
    except Exception as exc:  # pragma: no cover - non-critical path
      LOGGER.warning("Could not remove stale snapshot %s: %s", other, exc)

  canonical = portfolios_dir / "portfolio.csv"

  df.to_csv(canonical, index=False)

  LOGGER.info("Wrote normalised portfolio to %s", canonical)

  # Best-effort: remove original file from Downloads to avoid reusing stale
  # portfolios by accident. Nếu cần import lại, hãy export danh mục mới.
  try:
    if src.exists():
      src.unlink()
      LOGGER.info("Removed source portfolio from downloads: %s", src)
  except Exception as exc:  # pragma: no cover - non-critical path
    LOGGER.warning("Could not remove source portfolio %s: %s", src, exc)
  return canonical


def _parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[ImportConfig, Path]:
  parser = argparse.ArgumentParser(description="Import TCBS portfolio CSV from Downloads into data/portfolios/portfolio.csv")
  parser.add_argument(
    "--downloads-dir",
    default=os.environ.get("PORTFOLIO_DOWNLOADS_DIR", ""),
    help="Directory to search for exported CSVs (default: ~/Downloads or PORTFOLIO_DOWNLOADS_DIR env)",
  )
  parser.add_argument(
    "--broker",
    default="tcbs",
    help="Broker prefix used in filename (default: tcbs, expects tcbs-portfolio-*.csv)",
  )
  args = parser.parse_args(list(argv) if argv is not None else None)
  downloads_dir = _resolve_downloads_dir(args.downloads_dir)
  cfg = ImportConfig(downloads_dir=downloads_dir, broker_prefix=str(args.broker).strip().lower() or "tcbs")
  portfolios_root = (_repo_root() / "data" / "portfolios").resolve()
  return cfg, portfolios_root


def main(argv: Optional[Sequence[str]] = None) -> int:
  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")
  cfg, portfolios_dir = _parse_args(argv)
  latest = _find_latest_csv(cfg)
  _normalise_portfolio(latest, portfolios_dir)
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
