#!/usr/bin/env python3
"""Lightweight probe that demonstrates which NEXT_STEPS data CafeF tabs expose."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://cafef.vn"


@dataclass
class Dataset:
    status: str
    detail: Any
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "source": self.source}


class CafeFProbe:
    """Fetch the DOM-rendered tables behind the CafeF VN30 tabs."""

    def __init__(self, ticker: str, headless: bool = True):
        self.ticker = ticker.lower()
        self.headless = headless
        self.results: Dict[str, Dataset] = {}

    # ------------------------------------------------------------------ internals
    def _goto_tab(self, page, tab: int) -> None:
        url = f"{BASE_URL}/du-lieu/lich-su-giao-dich-{self.ticker}-{tab}.chn"
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)

    def _wait_for_table(self, page, selector: str, timeout: int = 12000) -> None:
        try:
            page.wait_for_selector(selector, timeout=timeout)
        except PlaywrightTimeoutError:
            # best effort; some tables (tab5 summary) have no <thead>/<tbody>
            pass

    def _extract_table(
        self,
        page,
        *,
        selector: Optional[str] = None,
        index: Optional[int] = None,
        limit: int = 25,
    ) -> Optional[Dict[str, Any]]:
        script = """
        ({ css, tableIndex, limit }) => {
          const pickTable = () => {
            if (css) return document.querySelector(css);
            const all = Array.from(document.querySelectorAll('table'));
            return all[tableIndex] || null;
          };
          const table = pickTable();
          if (!table) return null;
          const clean = (txt) => (txt || '').replace(/\\s+/g, ' ').trim();
          const grabRow = (row) => Array.from(row.querySelectorAll('th,td'))
            .map(cell => clean(cell.innerText));
          const headerRows = [];
          const head = table.querySelectorAll('thead tr');
          if (head.length) {
            head.forEach(tr => {
              const row = grabRow(tr);
              if (row.length) headerRows.push(row);
            });
          } else {
            const first = table.querySelector('tr');
            if (first) {
              const cells = grabRow(first);
              if (cells.every(txt => txt.length)) headerRows.push(cells);
            }
          }
          const bodyRows = [];
          const bodyTr = table.querySelectorAll('tbody tr');
          const candidates = bodyTr.length ? Array.from(bodyTr) : Array.from(table.querySelectorAll('tr')).slice(headerRows.length);
          for (const tr of candidates) {
            const row = grabRow(tr);
            if (!row.length || row.every(cell => !cell.length)) continue;
            bodyRows.push(row);
            if (limit > 0 && bodyRows.length >= limit) break;
          }
          return { headers: headerRows, rows: bodyRows };
        }
        """
        data = page.evaluate(
            script,
            {
                "css": selector,
                "tableIndex": index,
                "limit": limit,
            },
        )
        return data

    def _record(self, name: str, status: str, detail: Any, source: str) -> None:
        self.results[name] = Dataset(status=status, detail=detail, source=source)

    # ------------------------------------------------------------------ probes
    def probe_price_history(self, page) -> None:
        self._goto_tab(page, 1)
        self._wait_for_table(page, "table.owner-contents-table tbody tr")
        table = self._extract_table(page, selector="table.owner-contents-table", limit=15)
        if table and table["rows"]:
            self._record("price_history", "ok", table, "cafef_tab1_price_history")
        else:
            self._record("price_history", "missing", "table empty", "cafef_tab1_price_history")

    def probe_order_stats(self, page) -> None:
        self._goto_tab(page, 2)
        self._wait_for_table(page, "table.owner-contents-table tbody tr")
        table = self._extract_table(page, selector="table.owner-contents-table", limit=20)
        if table and table["rows"]:
            self._record("order_stats", "ok", table, "cafef_tab2_order_stats")
        else:
            self._record("order_stats", "missing", "order stats table empty", "cafef_tab2_order_stats")

    def probe_foreign_flow(self, page) -> None:
        self._goto_tab(page, 3)
        self._wait_for_table(page, "table.owner-contents-table tbody tr")
        table = self._extract_table(page, selector="table.owner-contents-table", limit=20)
        if table and table["rows"]:
            self._record("foreign_flow", "ok", table, "cafef_tab3_foreign_flow")
        else:
            self._record("foreign_flow", "missing", "foreign flow table empty", "cafef_tab3_foreign_flow")

    def probe_proprietary_flow(self, page) -> None:
        self._goto_tab(page, 4)
        self._wait_for_table(page, "table.owner-contents-table tbody tr")
        table = self._extract_table(page, selector="table.owner-contents-table", limit=20)
        if table and table["rows"]:
            self._record("proprietary_flow", "ok", table, "cafef_tab4_proprietary")
        else:
            self._record("proprietary_flow", "missing", "proprietary table empty", "cafef_tab4_proprietary")

    def probe_intraday(self, page) -> None:
        self._goto_tab(page, 5)
        # summary distribution table is the second <table> element
        summary = self._extract_table(page, index=1, limit=15)
        trades = self._extract_table(page, index=2, limit=50)
        detail = {"price_distribution": summary, "trade_samples": trades}
        status = "ok" if (summary and summary.get("rows") and trades and trades.get("rows")) else "partial"
        self._record("intraday_flow", status, detail, "cafef_tab5_intraday")

    # ------------------------------------------------------------------ entrypoint
    def run(self) -> Dict[str, Any]:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            self.probe_price_history(page)
            self.probe_order_stats(page)
            self.probe_foreign_flow(page)
            self.probe_proprietary_flow(page)
            self.probe_intraday(page)
            browser.close()
        return {name: dataset.to_dict() for name, dataset in self.results.items()}


def main(argv: list[str]) -> None:
    ticker = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "HPG"
    # Default headless; pass --show to run headful for debugging.
    headless = "--show" not in argv
    probe = CafeFProbe(ticker, headless=headless)
    result = {
        "ticker": ticker.upper(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": probe.run(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
