"""Parse basic fundamental stats from Vietstock overview pages via requests/bs4."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def _normalize_number(text: str) -> Optional[float]:
    if text is None:
        return None
    txt = text.strip()
    if not txt or txt == "-":
        return None
    txt = txt.replace(" ", "")
    txt = txt.replace(".", "") if txt.count(",") == 1 and txt.count(".") > 1 else txt
    txt = txt.replace(",", "")
    txt = txt.replace("%", "")
    try:
        return float(txt)
    except ValueError:
        # fallback for decimals using comma
        txt = text.strip().replace("%", "").replace(".", "").replace(",", ".")
        try:
            return float(txt)
        except ValueError:
            return None


def _parse_overview_labels(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, str] = {}
    for p in soup.find_all("p", class_="p8"):
        label_parts = []
        value = None
        for child in p.children:
            if getattr(child, "name", None) == "b":
                value = child.get_text(strip=True)
                break
            if isinstance(child, str):
                segment = child.strip()
                if segment:
                    label_parts.append(segment)
        label = " ".join(label_parts)
        if label and value is not None:
            data[label] = value
    return data


@dataclass
class VietstockOverviewRecord:
    ticker: str
    fetched_at: datetime
    fields: Dict[str, float | None]

    def to_json(self) -> str:
        payload = {
            "ticker": self.ticker,
            "fetched_at": self.fetched_at.isoformat(),
            "fields": self.fields,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "VietstockOverviewRecord":
        raw = json.loads(text)
        return cls(
            ticker=raw["ticker"],
            fetched_at=datetime.fromisoformat(raw["fetched_at"]),
            fields=raw["fields"],
        )


def fetch_overview_metrics(ticker: str) -> Dict[str, Optional[float]]:
    url = f"https://finance.vietstock.vn/{ticker}-ctcp.htm"
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    labels = _parse_overview_labels(resp.text)
    eps = _normalize_number(labels.get("EPS"))
    bvps = _normalize_number(labels.get("BVPS"))
    fpe = _normalize_number(labels.get("F P/E"))
    pb = _normalize_number(labels.get("P/B"))
    result: Dict[str, Optional[float]] = {
        "PE_fwd": fpe,
        "PB": pb,
    }
    if eps is not None and bvps not in (None, 0):
        result["ROE"] = (eps / bvps) * 100.0
    else:
        result["ROE"] = None
    return result


class VietstockOverviewCache:
    def __init__(self, base_dir: Path, max_age_hours: int = 24) -> None:
        self.base_dir = base_dir
        # When max_age_hours <= 0, treat records as non-expiring and
        # only refresh when cache is missing or unreadable.
        self.max_age = None if max_age_hours <= 0 else timedelta(hours=max_age_hours)

    def path_for(self, ticker: str) -> Path:
        return self.base_dir / f"{ticker.upper()}.json"

    def load(self, ticker: str) -> Optional[VietstockOverviewRecord]:
        path = self.path_for(ticker)
        if not path.exists():
            return None
        try:
            record = VietstockOverviewRecord.from_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if self.max_age is not None:
            if datetime.utcnow() - record.fetched_at > self.max_age:
                return None
        return record

    def store(self, ticker: str, fields: Dict[str, Optional[float]]) -> VietstockOverviewRecord:
        record = VietstockOverviewRecord(ticker=ticker.upper(), fetched_at=datetime.utcnow(), fields=fields)
        path = self.path_for(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.to_json(), encoding="utf-8")
        return record


def build_fundamental_frame(
    tickers: list[str], cache_dir: Path, max_age_hours: int = 24
) -> Dict[str, Dict[str, Optional[float]]]:
    cache = VietstockOverviewCache(cache_dir, max_age_hours=max_age_hours)
    output: Dict[str, Dict[str, Optional[float]]] = {}
    for ticker in tickers:
        normalized = ticker.strip().upper()
        if not normalized:
            continue
        record = cache.load(normalized)
        if record is None:
            fields = fetch_overview_metrics(normalized)
            record = cache.store(normalized, fields)
        output[normalized] = record.fields
    return output


__all__ = [
    "build_fundamental_frame",
    "fetch_overview_metrics",
    "VietstockOverviewCache",
]
