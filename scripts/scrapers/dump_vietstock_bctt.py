from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://finance.vietstock.vn/HPG/tai-chinh.htm?tab=BCTT"
OUT_DIR = Path("out")


def grab(page, selector: str) -> str:
    try:
        return page.inner_html(selector)
    except Exception:
        return ""


def dump_tables(page, suffix: str):
    kq = grab(page, "table#tbl-data-BCTT-KQ")
    cd = grab(page, "table#tbl-data-BCTT-CD")
    cstc = grab(page, "table#tbl-data-BCTT-CSTC")
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>BCTT dump {suffix}</title></head><body>
<h2>KQKD</h2><div>{kq}</div>
<h2>Can Doi Ke Toan</h2><div>{cd}</div>
<h2>Chi So Tai Chinh</h2><div>{cstc}</div>
</body></html>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"vietstock_bctt_tables{suffix}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} (KQ len={len(kq)}, CD len={len(cd)}, CSTC len={len(cstc)})")
    return out_path


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)
        dump_tables(page, "_page1")

        # Try go to previous periods (left chevron button name="btn-page-2") if present
        if page.locator('div[name="btn-page-2"]').first.is_visible():
            page.locator('div[name="btn-page-2"]').first.click()
            page.wait_for_timeout(2000)
            dump_tables(page, "_page2")

        browser.close()


if __name__ == "__main__":
    main()
