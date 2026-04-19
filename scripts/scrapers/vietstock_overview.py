from __future__ import annotations
import sys
import json
from playwright.sync_api import sync_playwright

LABELS = [
    "Mở cửa",
    "Cao nhất",
    "Thấp nhất",
    "KLGD",
    "Vốn hóa",
    "Dư mua",
    "Dư bán",
    "Cao 52T",
    "Thấp 52T",
    "KLBQ 52T",
    "NN mua",
    "% NN sở hữu",
    "Cổ tức TM",
    "T/S cổ tức",
    "Beta",
    "EPS",
    "P/E",
    "F P/E",
    "BVPS",
    "P/B",
]


def grab_value(page, label: str):
    return page.evaluate(
        """
    lbl => {
      const nodes = Array.from(document.querySelectorAll('p, div, span'));
      for (const n of nodes) {
        const text = (n.textContent || '').replace(/\s+/g, ' ').trim();
        const idx = text.indexOf(lbl);
        if (idx === -1) continue;
        const tail = text.slice(idx + lbl.length);
        const firstDigit = tail.search(/[0-9]/);
        const dashPos = tail.indexOf('-');
        if (dashPos !== -1 && (firstDigit === -1 || dashPos < firstDigit)) return '-';
        const m = tail.match(/[0-9][0-9.,-]*/);
        if (m && m[0]) return m[0].trim();
        if (tail.includes('-')) return '-';
      }
      return null;
    }
    """,
        label,
    )


def grab_price_block(page):
    return page.evaluate(
        """
    () => {
      const h = Array.from(document.querySelectorAll('h1, h2, h3')).find(el => /\d/.test(el.textContent||''));
      const price = h ? h.textContent.replace(/[^0-9.,-]/g,'').trim() : null;
      let change = null;
      if (h && h.parentElement) {
        const sibs = Array.from(h.parentElement.querySelectorAll('div, span, p'));
        const found = sibs.find(el => /%/.test(el.textContent||''));
        if (found) change = found.textContent.trim();
      }
      return {price, change};
    }
    """
    )


def scrape(ticker: str):
    url = f"https://finance.vietstock.vn/{ticker}-ctcp.htm"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        price_block = grab_price_block(page)
        data = {"Ticker": ticker, "Last": price_block.get("price"), "Change": price_block.get("change")}
        for label in LABELS:
            data[label] = grab_value(page, label)

        browser.close()
    return data


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "HPG").strip().upper()
    data = scrape(ticker)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
