from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright


TABLE_SELECTORS = {
    "income_statement": "table#tbl-data-BCTT-KQ",
    "balance_sheet": "table#tbl-data-BCTT-CD",
    "financial_ratios": "table#tbl-data-BCTT-CSTC",
}

QUARTERLY_LAG_DAYS = 45
Q4_LAG_DAYS = 75

RATIO_FEATURE_COLUMNS = [
    "BCTT_EPS4Q",
    "BCTT_BVPS",
    "BCTT_PB",
    "BCTT_GrossMarginPct",
    "BCTT_NetMarginPct",
    "BCTT_ROEAPct",
    "BCTT_ROAAPct",
    "BCTT_CurrentRatio",
    "BCTT_InterestCoverage",
    "BCTT_DebtToAssetsPct",
    "BCTT_DebtToEquityPct",
]

HYBRID_FEATURE_COLUMNS = RATIO_FEATURE_COLUMNS + [
    "BCTT_RevenueQoQPct",
    "BCTT_RevenueYoYPct",
    "BCTT_NetProfitQoQPct",
    "BCTT_NetProfitYoYPct",
    "BCTT_GrossProfitYoYPct",
    "BCTT_InventoryYoYPct",
    "BCTT_ReceivablesYoYPct",
    "BCTT_EquityYoYPct",
    "BCTT_CashToShortLiability",
    "BCTT_InventoryToRevenue",
    "BCTT_ReceivablesToRevenue",
]

FEATURE_SETS = {
    "ratios_only": RATIO_FEATURE_COLUMNS,
    "hybrid_growth": HYBRID_FEATURE_COLUMNS,
}

ROW_ALIASES = {
    "revenue": {
        "doanh thu thuan ve ban hang va cung cap dich vu",
    },
    "gross_profit": {
        "loi nhuan gop ve ban hang va cung cap dich vu",
    },
    "operating_profit": {
        "loi nhuan thuan tu hoat dong kinh doanh",
    },
    "net_profit": {
        "loi nhuan sau thue thu nhap doanh nghiep",
    },
    "parent_net_profit": {
        "loi nhuan sau thue cua co dong cong ty me",
    },
    "cash": {
        "tien va cac khoan tuong duong tien",
    },
    "receivables": {
        "cac khoan phai thu ngan han",
    },
    "inventory": {
        "hang ton kho",
    },
    "total_assets": {
        "tong cong tai san",
    },
    "total_liabilities": {
        "no phai tra",
    },
    "short_liabilities": {
        "no ngan han",
    },
    "long_liabilities": {
        "no dai han",
    },
    "equity": {
        "von chu so huu",
    },
    "eps_4q": {
        "thu nhap tren moi co phan cua 4 quy gan nhat eps",
    },
    "bvps": {
        "gia tri so sach cua co phieu bvps",
    },
    "pb": {
        "chi so gia thi truong tren gia tri so sach p/b",
    },
    "gross_margin_pct": {
        "ty suat loi nhuan gop bien",
    },
    "net_margin_pct": {
        "ty suat sinh loi tren doanh thu thuan",
    },
    "roea_pct": {
        "ty suat loi nhuan tren von chu so huu binh quan roea",
        "ty suat loi nhuan tren von chu so huu binh quan (roea)",
    },
    "roaa_pct": {
        "ty suat sinh loi tren tong tai san binh quan roaa",
        "ty suat sinh loi tren tong tai san binh quan (roaa)",
    },
    "current_ratio": {
        "ty so thanh toan hien hanh ngan han",
        "ty so thanh toan hien hanh (ngan han)",
    },
    "interest_coverage": {
        "kha nang thanh toan lai vay",
    },
    "debt_to_assets_pct": {
        "ty so no tren tong tai san",
    },
    "debt_to_equity_pct": {
        "ty so no vay tren von chu so huu",
    },
}


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _normalise_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_number(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    text = text.replace("\u2212", "-").replace("%", "").replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_period_end(label: str, date_range: str) -> pd.Timestamp:
    match = re.search(r"(\d{2}/\d{2}/\d{4})$", str(date_range or ""))
    if match:
        return pd.to_datetime(match.group(1), format="%d/%m/%Y")
    label_match = re.match(r"Q([1-4])\/(\d{4})", str(label or "").strip())
    if label_match:
        quarter = int(label_match.group(1))
        year = int(label_match.group(2))
        month = quarter * 3
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    raise ValueError(f"Cannot parse period end from label={label!r} date_range={date_range!r}")


def _available_date(period_end: pd.Timestamp) -> pd.Timestamp:
    quarter = ((int(period_end.month) - 1) // 3) + 1
    lag_days = Q4_LAG_DAYS if quarter == 4 else QUARTERLY_LAG_DAYS
    return period_end + pd.Timedelta(days=lag_days)


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    shifted = series.shift(periods)
    return ((series / shifted) - 1.0) * 100.0


@dataclass
class VietstockBcttRecord:
    ticker: str
    fetched_at: datetime
    tables: Dict[str, Dict[str, object]]

    def to_json(self) -> str:
        payload = {
            "ticker": self.ticker,
            "fetched_at": self.fetched_at.isoformat(),
            "tables": self.tables,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "VietstockBcttRecord":
        raw = json.loads(text)
        return cls(
            ticker=raw["ticker"],
            fetched_at=datetime.fromisoformat(raw["fetched_at"]),
            tables=raw["tables"],
        )


class VietstockBcttCache:
    def __init__(self, base_dir: Path, max_age_hours: int = 0) -> None:
        self.base_dir = base_dir
        self.max_age = None if max_age_hours <= 0 else timedelta(hours=max_age_hours)

    def path_for(self, ticker: str) -> Path:
        return self.base_dir / f"{_normalise_ticker(ticker)}.json"

    def load(self, ticker: str) -> Optional[VietstockBcttRecord]:
        path = self.path_for(ticker)
        if not path.exists():
            return None
        try:
            record = VietstockBcttRecord.from_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if self.max_age is not None and datetime.utcnow() - record.fetched_at > self.max_age:
            return None
        return record

    def store(self, ticker: str, tables: Dict[str, Dict[str, object]]) -> VietstockBcttRecord:
        record = VietstockBcttRecord(
            ticker=_normalise_ticker(ticker),
            fetched_at=datetime.utcnow(),
            tables=tables,
        )
        path = self.path_for(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.to_json(), encoding="utf-8")
        return record


def _extract_table(page, selector: str) -> Dict[str, object]:
    data = page.evaluate(
        r"""
        sel => {
          const table = document.querySelector(sel);
          if (!table) return null;
          const text = el => (el?.textContent || '').replace(/\s+/g, ' ').trim();
          const headers = Array.from(table.querySelectorAll('thead th')).map(text);
          const headerLinks = Array.from(table.querySelectorAll('thead th')).map(th => th.querySelector('a')?.href || null);
          const periodLabels = headers.slice(4);
          const periodLinks = headerLinks.slice(4);
          const rangeRow = table.querySelector('tbody tr[data-row-type="giaidoan"]');
          const dateRanges = rangeRow
            ? Array.from(rangeRow.querySelectorAll('td')).slice(4).map(td => td.getAttribute('title') || td.getAttribute('data-value') || text(td))
            : [];
          const rows = Array.from(table.querySelectorAll('tbody tr[data-row-type="reportnormId"]')).map(tr => {
            const cells = Array.from(tr.querySelectorAll('td'));
            return {
              report_norm_id: tr.getAttribute('data-reportnormid'),
              label: text(cells[0]?.querySelector('.report-norm-name') || cells[0]),
              unit: cells[2]?.getAttribute('data-value') || text(cells[2]),
              values: cells.slice(4).map(td => td.getAttribute('data-value') ?? text(td)),
            };
          });
          return {
            period_labels: periodLabels,
            period_links: periodLinks,
            date_ranges: dateRanges,
            rows,
          };
        }
        """,
        selector,
    )
    if not data:
        raise RuntimeError(f"Missing Vietstock table selector: {selector}")
    return data


def fetch_bctt_tables(
    ticker: str,
    headless: bool = True,
    timeout_ms: int = 60000,
) -> Dict[str, Dict[str, object]]:
    ticker = _normalise_ticker(ticker)
    url = f"https://finance.vietstock.vn/{ticker}/tai-chinh.htm?tab=BCTT"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_selector(TABLE_SELECTORS["income_statement"], timeout=timeout_ms)
        page.wait_for_timeout(1500)
        tables = {name: _extract_table(page, selector) for name, selector in TABLE_SELECTORS.items()}
        browser.close()
    return tables


def _row_lookup(table: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    rows = table.get("rows", []) or []
    return {_normalise_label(row.get("label")): row for row in rows}


def build_quarterly_feature_frame_from_tables(
    ticker: str,
    tables: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    income = tables.get("income_statement", {})
    balance = tables.get("balance_sheet", {})
    ratios = tables.get("financial_ratios", {})
    period_labels = list(income.get("period_labels", []) or [])
    date_ranges = list(income.get("date_ranges", []) or [])
    if not period_labels:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "Ticker": _normalise_ticker(ticker),
            "PeriodLabel": period_labels,
            "DateRange": date_ranges + [None] * max(0, len(period_labels) - len(date_ranges)),
        }
    )
    frame["PeriodEnd"] = [
        _parse_period_end(label, date_range)
        for label, date_range in zip(frame["PeriodLabel"], frame["DateRange"])
    ]
    frame["AvailableDate"] = frame["PeriodEnd"].map(_available_date)
    frame = frame.sort_values("PeriodEnd").reset_index(drop=True)

    lookups = {
        "income": _row_lookup(income),
        "balance": _row_lookup(balance),
        "ratios": _row_lookup(ratios),
    }

    def assign_series(column_name: str, table_key: str, alias_key: str) -> None:
        values: List[Optional[float]] = [None] * len(frame)
        lookup = lookups[table_key]
        matched_row = None
        for candidate in ROW_ALIASES[alias_key]:
            matched_row = lookup.get(_normalise_label(candidate))
            if matched_row is not None:
                break
        if matched_row is not None:
            raw_values = matched_row.get("values", []) or []
            values = [_normalise_number(value) for value in raw_values[: len(frame)]]
        frame[column_name] = pd.Series(values, index=frame.index, dtype="float64")

    assign_series("Revenue", "income", "revenue")
    assign_series("GrossProfit", "income", "gross_profit")
    assign_series("OperatingProfit", "income", "operating_profit")
    assign_series("NetProfit", "income", "net_profit")
    assign_series("ParentNetProfit", "income", "parent_net_profit")
    assign_series("Cash", "balance", "cash")
    assign_series("Receivables", "balance", "receivables")
    assign_series("Inventory", "balance", "inventory")
    assign_series("TotalAssets", "balance", "total_assets")
    assign_series("TotalLiabilities", "balance", "total_liabilities")
    assign_series("ShortLiabilities", "balance", "short_liabilities")
    assign_series("LongLiabilities", "balance", "long_liabilities")
    assign_series("Equity", "balance", "equity")
    assign_series("BCTT_EPS4Q", "ratios", "eps_4q")
    assign_series("BCTT_BVPS", "ratios", "bvps")
    assign_series("BCTT_PB", "ratios", "pb")
    assign_series("BCTT_GrossMarginPct", "ratios", "gross_margin_pct")
    assign_series("BCTT_NetMarginPct", "ratios", "net_margin_pct")
    assign_series("BCTT_ROEAPct", "ratios", "roea_pct")
    assign_series("BCTT_ROAAPct", "ratios", "roaa_pct")
    assign_series("BCTT_CurrentRatio", "ratios", "current_ratio")
    assign_series("BCTT_InterestCoverage", "ratios", "interest_coverage")
    assign_series("BCTT_DebtToAssetsPct", "ratios", "debt_to_assets_pct")
    assign_series("BCTT_DebtToEquityPct", "ratios", "debt_to_equity_pct")

    frame["BCTT_RevenueQoQPct"] = _safe_pct_change(frame["Revenue"], 1)
    frame["BCTT_RevenueYoYPct"] = _safe_pct_change(frame["Revenue"], 4)
    frame["BCTT_NetProfitQoQPct"] = _safe_pct_change(frame["NetProfit"], 1)
    frame["BCTT_NetProfitYoYPct"] = _safe_pct_change(frame["NetProfit"], 4)
    frame["BCTT_GrossProfitYoYPct"] = _safe_pct_change(frame["GrossProfit"], 4)
    frame["BCTT_InventoryYoYPct"] = _safe_pct_change(frame["Inventory"], 4)
    frame["BCTT_ReceivablesYoYPct"] = _safe_pct_change(frame["Receivables"], 4)
    frame["BCTT_EquityYoYPct"] = _safe_pct_change(frame["Equity"], 4)
    frame["BCTT_CashToShortLiability"] = frame["Cash"] / frame["ShortLiabilities"]
    frame["BCTT_InventoryToRevenue"] = frame["Inventory"] / frame["Revenue"]
    frame["BCTT_ReceivablesToRevenue"] = frame["Receivables"] / frame["Revenue"]

    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], pd.NA)
    return frame


def load_or_fetch_bctt_feature_frame(
    tickers: Sequence[str],
    cache_dir: Path,
    max_age_hours: int = 0,
    headless: bool = True,
) -> pd.DataFrame:
    cache = VietstockBcttCache(cache_dir, max_age_hours=max_age_hours)
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        normalized = _normalise_ticker(ticker)
        if not normalized:
            continue
        record = cache.load(normalized)
        if record is None:
            tables = fetch_bctt_tables(normalized, headless=headless)
            record = cache.store(normalized, tables)
        quarterly_df = build_quarterly_feature_frame_from_tables(normalized, record.tables)
        if not quarterly_df.empty:
            frames.append(quarterly_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_daily_bctt_feature_frame(
    sample_df: pd.DataFrame,
    cache_dir: Path,
    feature_set: str,
    max_age_hours: int = 0,
    headless: bool = True,
) -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown BCTT feature set: {feature_set}")
    tickers = sorted({_normalise_ticker(ticker) for ticker in sample_df["Ticker"].unique()})
    quarterly_df = load_or_fetch_bctt_feature_frame(
        tickers=tickers,
        cache_dir=cache_dir,
        max_age_hours=max_age_hours,
        headless=headless,
    )
    if quarterly_df.empty:
        return pd.DataFrame(columns=["Date", "Ticker"] + FEATURE_SETS[feature_set])

    output_frames: List[pd.DataFrame] = []
    columns = FEATURE_SETS[feature_set]
    for ticker, group in sample_df.groupby("Ticker", sort=False):
        normalized = _normalise_ticker(ticker)
        ticker_quarterly = quarterly_df[quarterly_df["Ticker"] == normalized].copy()
        if ticker_quarterly.empty:
            continue
        daily_dates = pd.DataFrame({"Date": sorted(pd.to_datetime(group["Date"]).unique())})
        aligned = pd.merge_asof(
            daily_dates.sort_values("Date"),
            ticker_quarterly[["AvailableDate"] + columns].sort_values("AvailableDate"),
            left_on="Date",
            right_on="AvailableDate",
            direction="backward",
        )
        aligned["Ticker"] = normalized
        output_frames.append(aligned.drop(columns=["AvailableDate"]))
    return pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame(columns=["Date", "Ticker"] + columns)


__all__ = [
    "FEATURE_SETS",
    "HYBRID_FEATURE_COLUMNS",
    "RATIO_FEATURE_COLUMNS",
    "VietstockBcttCache",
    "VietstockBcttRecord",
    "build_daily_bctt_feature_frame",
    "build_quarterly_feature_frame_from_tables",
    "fetch_bctt_tables",
    "load_or_fetch_bctt_feature_frame",
]
