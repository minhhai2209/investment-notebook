#!/usr/bin/env python3
"""Probe which NEXT_STEPS.md data points are accessible via existing scrapers."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE = "https://finance.vietstock.vn"


@dataclass
class CheckResult:
    status: str
    detail: Any
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "source": self.source}


class VietstockNextStepsProbe:
    def __init__(self, ticker: str, headless: bool = True):
        self.ticker = ticker.upper()
        self.headless = headless
        self.results: Dict[str, CheckResult] = {}

    # ------------------------------------------------------------------ helpers
    def _extract_table(self, page, selector: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        try:
            data = page.evaluate(
                r"""
                ({ sel, limit }) => {
                  const table = document.querySelector(sel);
                  if (!table) return null;
                  const makeText = el => (el.textContent || '').replace(/\s+/g, ' ').trim();
                  let headers = Array.from(table.querySelectorAll('thead th')).map(makeText);
                  if (!headers.length) {
                    const firstRow = table.querySelector('tbody tr');
                    if (firstRow) headers = Array.from(firstRow.cells).map((_, idx) => `col${idx + 1}`);
                  }
                  const rows = Array.from(table.querySelectorAll('tbody tr')).slice(0, limit)
                    .map(tr => Array.from(tr.cells).map(makeText));
                  return { headers, rows };
                }
                """,
                {"sel": selector, "limit": limit},
            )
        except PlaywrightTimeoutError:
            return None
        return data

    def _navigate(self, page, path: str):
        page.goto(f"{BASE}/{path}", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)

    # ------------------------------------------------------------------ scrapers
    def probe_foreign_flow_dom(self, page):
        self._navigate(page, f"{self.ticker}/thong-ke-giao-dich.htm?grid=invest")
        table = self._extract_table(page, "table.table-striped.table-hover.table-bordered", limit=10)
        if table and table["rows"]:
            payload = {"headers": table["headers"], "rows": table["rows"][:5]}
            self.results["foreign_flow_dom"] = CheckResult("ok", payload, "vietstock_dom_grid=invest")
        else:
            reason = "foreign flow table missing (possible layout/login change)"
            self.results["foreign_flow_dom"] = CheckResult("missing", reason, "vietstock_dom_grid=invest")

    def probe_intraday_snapshot(self, page):
        table = self._extract_table(page, "table.table-deal", limit=5)
        if table and table["rows"]:
            first = table["rows"][0]
            payload = {"headers": table["headers"], "sample": first}
            self.results["intraday_snapshot"] = CheckResult("ok", payload, "vietstock_dom_table-deal")
        else:
            self.results["intraday_snapshot"] = CheckResult(
                "missing", "intraday tape table not found", "vietstock_dom_table-deal"
            )

    def probe_foreign_flow_api(self):
        url = "https://finfo-api.vndirect.com.vn/v4/foreignTrading"
        params = {"q": f"code:{self.ticker}", "sort": "date:desc", "size": 5}
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            self.results["foreign_flow_vnd_api"] = CheckResult("ok", data, "vndirect_api")
        except Exception as exc:  # noqa: BLE001
            self.results["foreign_flow_vnd_api"] = CheckResult("error", str(exc), "vndirect_api")

    def probe_proprietary_flow(self, page):
        self._navigate(page, f"{self.ticker}/thong-ke-giao-dich.htm?grid=proprietary")
        table = self._extract_table(page, "table.table-striped.table-hover.table-bordered", limit=5)
        if not table:
            detail = "table missing"
            status = "missing"
        elif any("Tổng KLGD" in h for h in table["headers"]):
            detail = "page only returns generic trading stats; no proprietary rows"
            status = "blocked"
        else:
            detail = {"headers": table["headers"], "rows": table["rows"]}
            status = "ok"
        self.results["proprietary_flow_dom"] = CheckResult(status, detail, "vietstock_dom_grid=proprietary")

    def probe_events(self, page):
        self._navigate(page, f"{self.ticker}/tin-tuc-su-kien.htm")
        blocks = page.evaluate(
            r"""
            () => {
              const makeText = el => (el?.textContent || '').replace(/\s+/g, ' ').trim();
              const sections = [];
              const tables = document.querySelectorAll('table.table-striped.table-middle.pos-relative.no-m-b.noscroll');
              tables.forEach(table => {
                const root = table.closest('.widget-home__item, .box-table, .module-box, .col-md-6, .col-sm-6');
                const header = makeText(root?.querySelector('h3, .widget-home__title, .box-table-header h3'))
                  || makeText(table.previousElementSibling);
                const rows = Array.from(table.querySelectorAll('tbody tr')).slice(0,5)
                  .map(tr => Array.from(tr.cells).map(td => makeText(td)));
                sections.push({header, rows});
              });
              return sections;
            }
            """
        )
        if blocks:
            self.results["events_dom"] = CheckResult("ok", blocks, "vietstock_dom_events")
        else:
            self.results["events_dom"] = CheckResult("missing", "no event tables rendered", "vietstock_dom_events")

    def probe_overview_stats(self, page):
        self._navigate(page, f"{self.ticker}-ctcp-tap-doan-hoa-phat.htm")
        labels = [
            "Mở cửa",
            "Cao nhất",
            "Thấp nhất",
            "KLGD",
            "Vốn hóa",
            "NN mua",
            "% NN sở hữu",
            "EPS",
            "P/E",
            "F P/E",
            "BVPS",
            "P/B",
            "Cổ tức TM",
            "T/S cổ tức",
            "ROE",
            "ROA",
            "Debt to Equity",
        ]
        data = {}
        for label in labels:
            value = page.evaluate(
                r"""
                lbl => {
                  const nodes = Array.from(document.querySelectorAll('p, div, span'));
                  for (const node of nodes) {
                    const text = (node.textContent || '').replace(/\s+/g, ' ').trim();
                    if (!text.includes(lbl)) continue;
                    const match = text.slice(text.indexOf(lbl) + lbl.length).match(/[0-9][0-9.,%-]*/);
                    if (match) return match[0];
                  }
                  return null;
                }
                """,
                label,
            )
            if value:
                data[label] = value
        status = "ok" if data else "missing"
        self.results["overview_metrics"] = CheckResult(status, data or "no metrics scraped", "vietstock_dom_overview")

    def mark_unavailable(self):
        self.results["regime_tags"] = CheckResult(
            "unavailable",
            "Requires custom regime calculation (VNINDEX ATR ranks / sector regimes).",
            "not_provided",
        )
        self.results["portfolio_concentration"] = CheckResult(
            "unavailable",
            "Needs internal portfolio data (SectorWeightPct/BetaContribution).",
            "engine_internal",
        )

    # ------------------------------------------------------------------ entrypoint
    def run(self) -> Dict[str, Any]:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            self.probe_foreign_flow_dom(page)
            self.probe_intraday_snapshot(page)
            self.probe_proprietary_flow(page)
            self.probe_events(page)
            self.probe_overview_stats(page)
            browser.close()
        self.probe_foreign_flow_api()
        self.mark_unavailable()
        return {k: v.to_dict() for k, v in self.results.items()}


def main(argv: list[str]) -> None:
    ticker = argv[1] if len(argv) > 1 else "HPG"
    # Default headless; pass --show to run headful for debugging.
    headless = "--show" not in argv
    probe = VietstockNextStepsProbe(ticker, headless=headless)
    result = {
        "ticker": ticker.upper(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "checks": probe.run(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
