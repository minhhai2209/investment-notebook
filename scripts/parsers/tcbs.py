"""TCBS portfolio table parser (no browser automation).

This module contains only the logic to map a TCBS HTML table (headers + rows)
into the canonical portfolio schema ``Ticker,Quantity,AvgPrice``.

It is kept separate from any scraping logic so the data engine can remain
independent of Playwright/browser specifics.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, List

import pandas as pd


LOGGER = logging.getLogger("tcbs_table_parser")


def _norm_number(text: str) -> float:
  """Parse a locale-ish number like '1,600' or '51.24' to float.

  - Remove all non-digit/non-dot/non-minus characters (commas, spaces, units).
  - Keep the last dot as decimal separator.
  """
  raw = (text or "").strip()
  if not raw:
    return 0.0
  cleaned = "".join(ch for ch in raw if ch.isdigit() or ch in {"-", "."})
  if cleaned in {"", "-", ".", "-."}:
    return 0.0
  try:
    return float(cleaned)
  except Exception:
    return 0.0


@dataclass
class TableMapping:
  symbol_idx: int
  quantity_idx: int
  avgprice_idx: int


def build_mapping(headers: List[str]) -> TableMapping:
  """Derive column indices from header texts.

  Expected Vietnamese labels (robust to whitespace/casing):
    - Ticker: 'Mã'
    - Quantity: prefer 'SL Tổng', fallback to 'Được GD' (tradable)
    - AvgPrice: 'Giá vốn'
  """
  norm = [h.strip().lower() for h in headers]

  def idx_of(*candidates: str) -> int:
    for label in candidates:
      for i, h in enumerate(norm):
        if label in h:
          return i
    raise RuntimeError(f"Missing required column header among: {candidates}")

  symbol_idx = idx_of("mã")
  quantity_idx = idx_of("sl tổng", "được gd", "sl tổng =", "sl tổng")
  avgprice_idx = idx_of("giá vốn")
  return TableMapping(symbol_idx, quantity_idx, avgprice_idx)


def parse_tcbs_table(headers: List[str], rows: List[List[str]]) -> pd.DataFrame:
  """Convert raw TCBS table headers + rows into a portfolio DataFrame."""
  LOGGER.debug("parse_tcbs_table: headers=%s", headers)
  mapping = build_mapping(headers)
  LOGGER.debug(
    "Column mapping: symbol=%d quantity=%d avg=%d",
    mapping.symbol_idx,
    mapping.quantity_idx,
    mapping.avgprice_idx,
  )
  out_rows: List[Dict[str, object]] = []

  def _clean_ticker(raw: str) -> str:
    s = (raw or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    for tok in re.split(r"\s+|,", s):
      if re.fullmatch(r"[A-Z0-9]{1,6}", tok):
        return tok
    s2 = re.sub(r"[^A-Z0-9]", "", s)
    return s2[:6] if s2 else s

  for r in rows:
    if not r or len(r) <= max(mapping.symbol_idx, mapping.quantity_idx, mapping.avgprice_idx):
      continue
    ticker = _clean_ticker(r[mapping.symbol_idx])
    qty = _norm_number(r[mapping.quantity_idx])
    avg = _norm_number(r[mapping.avgprice_idx])
    if not ticker or qty <= 0:
      continue
    out_rows.append({"Ticker": ticker, "Quantity": int(qty), "AvgPrice": float(avg)})

  if out_rows:
    df = pd.DataFrame(out_rows, columns=["Ticker", "Quantity", "AvgPrice"])
  else:
    df = pd.DataFrame(columns=["Ticker", "Quantity", "AvgPrice"])

  LOGGER.info("Parsed portfolio rows: %d (raw=%d)", len(df), len(rows))
  if LOGGER.isEnabledFor(logging.DEBUG) and not df.empty:
    LOGGER.debug("Sample: %s", df.head(10).to_dict(orient="records"))
  return df

