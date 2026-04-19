from __future__ import annotations

import html
import re

import requests


SECTOR_BLOCK_RE = re.compile(
    r"<div\b(?=[^>]*\bsector-level\b)[^>]*>(?P<body>.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
ANCHOR_TEXT_RE = re.compile(r"<a\b[^>]*>(?P<text>.*?)</a>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def build_company_profile_url(ticker: str) -> str:
    normalized = ticker.strip().upper()
    return f"https://finance.vietstock.vn/{normalized}/ho-so-doanh-nghiep.htm"


def _clean_html_text(raw: str) -> str:
    text = TAG_RE.sub(" ", html.unescape(raw or ""))
    return re.sub(r"\s+", " ", text).strip(" /")


def parse_vietstock_sector_levels(page_html: str) -> list[str]:
    match = SECTOR_BLOCK_RE.search(page_html or "")
    if not match:
        raise RuntimeError("Vietstock company profile is missing the sector-level block")

    levels: list[str] = []
    for anchor_html in ANCHOR_TEXT_RE.findall(match.group("body")):
        text = _clean_html_text(anchor_html)
        if text:
            levels.append(text)
    if not levels:
        raise RuntimeError("No sector labels found inside the Vietstock sector-level block")
    return levels


def select_vietstock_sector(levels: list[str], mode: str) -> str:
    if not levels:
        raise RuntimeError("Cannot select sector from an empty level list")
    if mode == "top":
        return levels[0]
    if mode == "leaf":
        return levels[-1]
    if mode == "path":
        return " / ".join(levels)
    raise RuntimeError(f"Unsupported Vietstock sector mode: {mode}")


def fetch_company_profile_html(
    ticker: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    }
    client = session or requests.Session()
    resp = client.get(build_company_profile_url(ticker), timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text or ""


def fetch_vietstock_sector_levels(
    ticker: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> list[str]:
    page_html = fetch_company_profile_html(ticker, session=session, timeout=timeout)
    return parse_vietstock_sector_levels(page_html)
