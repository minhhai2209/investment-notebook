"""Playwright-based scraper to fetch portfolio from TCInvest (TCBS).

Usage:
  python -m scripts.scrapers.tcbs [--headful] [--login-only]

Credentials:
  - Read from environment variables `TCBS_USERNAME` and `TCBS_PASSWORD`.
  - If a `.env` file exists at repo root, it will be loaded automatically
    when python-dotenv is available (optional dependency).

Output:
  - Writes `data/portfolios/portfolio.csv` with columns: Ticker,Quantity,AvgPrice

Fingerprint persistence:
  - Uses a single persistent Chromium user data directory at `.playwright/tcbs-user-data/default`.
    The first run may require device/OTP confirmation. Run with `--headful` to
    complete verification once; subsequent runs reuse the same profile.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import logging
import time
from contextlib import contextmanager

import pandas as pd


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for candidate in [cur.parent, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


# Module logger (keep lightweight; respect app config if present)
LOGGER = logging.getLogger("tcbs")


def _ensure_logging_configured() -> None:
    """Configure basic logging if application hasn't done so.

    Uses a concise format and INFO level by default.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")


@contextmanager
def _log_step(name: str, **fields: object):
    """Log step start/end with duration and structured fields."""
    meta = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
    LOGGER.info("▶ %s%s", name, (" " + meta) if meta else "")
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        LOGGER.exception("✖ %s failed after %.3fs", name, dt)
        raise
    else:
        dt = time.perf_counter() - t0
        LOGGER.info("✓ %s done in %.3fs", name, dt)


def _log_url_after_goto(page, tag: str) -> None:
    """Log the current page URL immediately after goto and after a short settle.

    Accept a Playwright `page` to make this usable across functions.
    """
    try:
        LOGGER.info("current_url=%s tag=%s", page.url, f"{tag}.after_goto")
        page.wait_for_load_state("domcontentloaded")
    except Exception:
        pass
    try:
        page.wait_for_timeout(700)
    except Exception:
        pass
    LOGGER.info("current_url=%s tag=%s", page.url, f"{tag}.after_settle")


def _load_env_if_present(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        import dotenv  # type: ignore

        dotenv.load_dotenv(env_path)
        LOGGER.debug("Loaded .env from %s", env_path)
    except Exception:
        # Optional dependency; proceed if not available
        LOGGER.debug("python-dotenv unavailable; skip .env load")


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    # Do not log secrets; only presence/length
    LOGGER.debug("Env %s present (len=%d)", name, len(val))
    return val


def _default_post_login_wait_ms() -> int:
    raw = os.environ.get("TCBS_POST_LOGIN_WAIT_MS", "20000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        LOGGER.warning("Invalid TCBS_POST_LOGIN_WAIT_MS=%r; using 20000", raw)
        return 20000


def _norm_number(text: str) -> float:
    """Parse a locale-ish number like '1,600' or '51.24' to float.

    - Remove all non-digit/non-dot/non-minus characters (commas, spaces, units).
    - Keep the last dot as decimal separator.
    """
    raw = (text or "").strip()
    if not raw:
        return 0.0
    # Remove thousand separators and non-numeric symbols (keep - and .)
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
    # Prefer SL Tổng (total quantity), fallback to Được GD
    quantity_idx = idx_of("sl tổng", "được gd", "sl tổng =", "sl tổng")
    avgprice_idx = idx_of("giá vốn")
    return TableMapping(symbol_idx, quantity_idx, avgprice_idx)


def parse_tcbs_table(headers: List[str], rows: List[List[str]]) -> pd.DataFrame:
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
        # Collapse whitespace/newlines
        s = re.sub(r"\s+", " ", s)
        # Pick the first all-caps alnum token (1-6 chars), typical VN tickers
        for tok in re.split(r"\s+|,", s):
            if re.fullmatch(r"[A-Z0-9]{1,6}", tok):
                return tok
        # Fallback: remove non-alnum underscores
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
    df = pd.DataFrame(out_rows, columns=["Ticker", "Quantity", "AvgPrice"]) if out_rows else pd.DataFrame(
        columns=["Ticker", "Quantity", "AvgPrice"]
    )
    LOGGER.info("Parsed portfolio rows: %d (raw=%d)", len(df), len(rows))
    if LOGGER.isEnabledFor(logging.DEBUG) and not df.empty:
        LOGGER.debug("Sample: %s", df.head(10).to_dict(orient="records"))
    return df


def _ensure_playwright_installed(pybin: str) -> None:
    # Best-effort: ensure Chromium is installed; skip on failure (user can install manually)
    try:
        import subprocess
        LOGGER.debug("Ensure Playwright Chrome (via %s)", pybin)
        subprocess.run(
            [pybin, "-m", "playwright", "install", "chrome"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        LOGGER.debug("Skip Playwright installation check (subprocess error)")


def fetch_tcbs_portfolio(
    headless: bool = True,
    timeout_ms: int = 300000,
    slow_mo_ms: Optional[int] = None,
    post_login_wait_ms: Optional[int] = None,
    login_only: bool = False,
) -> Optional[Path]:
    """Launch persistent Chromium, log in to TCBS, navigate to portfolio table, parse and write CSV."""
    _ensure_logging_configured()
    with _log_step("setup"):
        root = _repo_root()
        LOGGER.info("repo_root=%s", root)
        _load_env_if_present(root)
        username = _require_env("TCBS_USERNAME")
        password = _require_env("TCBS_PASSWORD")

    # Prepare output dirs
    portfolios_root = (root / "data" / "portfolios").resolve()
    portfolios_root.mkdir(parents=True, exist_ok=True)
    out_path = portfolios_root / "portfolio.csv"
    LOGGER.info("output_file=%s", out_path)

    # Playwright runtime
    pybin = sys.executable
    _ensure_playwright_installed(pybin)

    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    user_data_root = (root / ".playwright" / "tcbs-user-data").resolve()
    user_data_dir = (user_data_root / "default").resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    resolved_post_login_wait_ms = post_login_wait_ms if post_login_wait_ms is not None else _default_post_login_wait_ms()
    LOGGER.info(
        "user_data_dir=%s headless=%s timeout_ms=%d post_login_wait_ms=%d login_only=%s",
        user_data_dir,
        headless,
        timeout_ms,
        resolved_post_login_wait_ms,
        login_only,
    )

    with sync_playwright() as p:
        # Determine pacing: default to slower actions in headful mode
        sm = int(slow_mo_ms) if slow_mo_ms is not None else (250 if not headless else 0)
        sm = max(0, sm)
        def _launch():
            return p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=headless,
                viewport={"width": 1400, "height": 900},
                slow_mo=sm or 0,
                channel="chrome",
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-http-cache",
                ],
            )
        try:
            with _log_step("launch_chromium", headless=headless):
                context = _launch()
        except Exception as exc:
            msg = str(exc)
            if "ProcessSingleton" in msg or "SingletonLock" in msg:
                lock = user_data_dir / "SingletonLock"
                try:
                    if lock.exists():
                        LOGGER.warning("Removing stale lock: %s", lock)
                        lock.unlink()
                except Exception:
                    LOGGER.debug("Failed to remove lock; retrying launch anyway")
                with _log_step("launch_chromium_retry", reason="singleton lock"):
                    context = _launch()
            else:
                raise
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            page.set_default_navigation_timeout(timeout_ms)
        except Exception:
            pass
        LOGGER.info("pacing slow_mo_ms=%d", sm)

        def _has_login_ui() -> bool:
            """Heuristic: detect presence of TCBS login inputs/buttons.

            Prefer feature detection over relying on page.url to handle client-side redirects.
            """
            try:
                loc = page.get_by_placeholder(re.compile(r"Email|Số\s*tài\s*khoản|Điện\s*thoại", re.I))
                if loc.count() > 0 and loc.first.is_visible():
                    return True
            except Exception:
                pass
            selectors = [
                'input[formcontrolname="username"]',
                'input[formcontrolname="password"]',
                'input[type="password"]',
                "button.btn-login",
                "button:has-text('Đăng nhập')",
            ]
            for sel in selectors:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible():
                        return True
                except Exception:
                    continue
            return False

        def _wait_after_login_submit(reason: str) -> None:
            if resolved_post_login_wait_ms <= 0:
                return
            LOGGER.info(
                "login: waiting %.1fs after submit for MFA/device approval (%s)",
                resolved_post_login_wait_ms / 1000.0,
                reason,
            )
            try:
                page.wait_for_timeout(resolved_post_login_wait_ms)
            except Exception:
                pass

        def _logout_if_logged_in() -> None:
            """If a previous session is still logged in, log out first.

            This ensures every run starts from a clean login with the configured
            credentials instead of reusing any browser state.
            """
            try:
                if _has_login_ui():
                    LOGGER.info("logout_check: login UI visible; assuming already logged out")
                    return
                LOGGER.info("logout_check: attempting to open user menu for logout")

                def _click_menu_trigger() -> bool:
                    # Primary: header avatar overlay using cdkoverlayorigin
                    selectors = [
                        "div.mat-ripple[cdkoverlayorigin]",
                        "div[cdkoverlayorigin]",
                    ]
                    for sel in selectors:
                        try:
                            loc = page.locator(sel)
                            if loc.count() > 0 and loc.first.is_visible():
                                with _log_step("open_user_menu", selector=sel):
                                    loc.first.click()
                                    try:
                                        page.wait_for_timeout(500)
                                    except Exception:
                                        pass
                                return True
                        except Exception:
                            continue
                    return False

                if not _click_menu_trigger():
                    LOGGER.info("logout_check: user menu trigger not found; skip explicit logout")
                    return

                def _find_logout():
                    candidates = [
                        lambda: page.get_by_role("button", name=re.compile(r"đăng\s*xuất", re.I)),
                        lambda: page.get_by_text(re.compile(r"đăng\s*xuất", re.I)),
                        lambda: page.locator("div.t-cs-i-t", has_text="Đăng xuất"),
                    ]
                    for c in candidates:
                        try:
                            loc = c()
                            if loc.count() > 0 and loc.first.is_visible():
                                return loc.first
                        except Exception:
                            continue
                    return None

                logout_btn = _find_logout()
                if not logout_btn:
                    LOGGER.warning("logout_check: logout item not found after opening menu")
                    return

                with _log_step("logout_click"):
                    logout_btn.click()
                # Wait briefly for the UI to transition back to login
                for _ in range(6):
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        pass
                    if _has_login_ui():
                        LOGGER.info("logout_check: login UI detected after logout")
                        return
                LOGGER.info("logout_check: logout click sent but login UI not detected; continuing")
            except Exception:
                LOGGER.exception("logout_check: error while trying to log out")

        def attempt_login() -> bool:
            LOGGER.info("login: attempt begin")
            # Add explicit pacing between login actions. TCBS login can be flaky when
            # actions happen too quickly (especially in headless mode where slow_mo=0).
            login_pace_ms = 400 if (sm or 0) == 0 else max(200, min(500, int(sm)))
            login_poll_ms = max(150, min(400, login_pace_ms // 2))

            def _pace(mult: int = 1) -> None:
                try:
                    page.wait_for_timeout(login_pace_ms * max(1, mult))
                except Exception:
                    pass

            # Prefer placeholder-based locators; fallback to formcontrolname
            user_candidates = [
                lambda: page.get_by_placeholder(re.compile(r"Email|Số\s*tài\s*khoản|Điện\s*thoại", re.I)),
                lambda: page.locator('input[formcontrolname="username"]'),
                lambda: page.locator('input[type="text"]'),
            ]
            pass_candidates = [
                lambda: page.get_by_placeholder(re.compile(r"Mật\s*khẩu|Password", re.I)),
                lambda: page.locator('input[formcontrolname="password"]'),
                lambda: page.locator('input[type="password"]'),
            ]

            def pick(cands):
                for c in cands:
                    try:
                        loc = c()
                        if loc.count() > 0:
                            return loc.first
                    except Exception:
                        continue
                return None

            user_input = pick(user_candidates)
            pass_input = pick(pass_candidates)
            if not user_input or not pass_input:
                LOGGER.info("login: inputs not found; maybe already logged in")
                return False

            with _log_step("login_fill"):
                user_input.wait_for(state="visible", timeout=timeout_ms)
                user_input.click()
                _pace()
                user_input.fill(username)
                _pace()
                pass_input.click()
                _pace()
                pass_input.fill(password)
                # Small pause so any oninput validation can enable the button
                _pace(mult=2)

            # Try multiple click strategies for the login button
            for attempt in range(4):
                try:
                    LOGGER.info("login: click button strategy=%d", attempt + 1)
                    if attempt == 0:
                        page.get_by_role("button", name=re.compile(r"đăng\s*nhập", re.I)).first.click()
                    elif attempt == 1:
                        page.locator("button.btn-login").first.click()
                    elif attempt == 2:
                        page.locator("button:has-text('Đăng nhập')").first.click()
                    else:
                        pass_input.press("Enter")
                    _pace()
                    # Wait for login UI to disappear (but avoid long "networkidle" waits).
                    for _ in range(20):
                        if not _has_login_ui():
                            break
                        try:
                            page.wait_for_timeout(login_poll_ms)
                        except Exception:
                            pass
                    return True
                except Exception:
                    continue
            return False

        # Step 0: Open home; if already logged in, log out first, then perform login
        with _log_step("navigate", url="/home"):
            # Use 'commit' to return as soon as headers arrive; SPA may redirect afterward
            resp = page.goto("https://tcinvest.tcbs.com.vn/home", wait_until="commit")
            try:
                LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
            except Exception:
                pass
            _log_url_after_goto(page, "/home")
        if not _has_login_ui():
            _logout_if_logged_in()
        if _has_login_ui():
            login_submitted = attempt_login()
            if login_submitted:
                _wait_after_login_submit("initial login")

        if login_only:
            LOGGER.info("login_only: finished TCBS login flow; closing browser")
            context.close()
            return None

        # Always navigate explicitly to my-asset (site may not redirect)
        with _log_step("navigate", url="/my-asset"):
            resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
            try:
                LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
            except Exception:
                pass
            _log_url_after_goto(page, "/my-asset")
        # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
        if _has_login_ui():
            LOGGER.warning("login: still on login; complete OTP/device confirm if prompted")
            page.wait_for_timeout(5000)
            login_retried = attempt_login()
            if login_retried:
                _wait_after_login_submit("retry login")
            with _log_step("navigate", url="/my-asset"):
                resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
                try:
                    LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
                except Exception:
                    pass
                _log_url_after_goto(page, "/my-asset.retry")
        # Avoid blocking on long-lived sockets
        page.wait_for_load_state("domcontentloaded")

        # Some sessions land on /my-asset correctly on the first navigation. Avoid
        # forcing a redundant second goto because that has been observed to bounce
        # the session back to the guest/login redirect path.
        current_url = ""
        try:
            current_url = page.url or ""
        except Exception:
            current_url = ""
        if _has_login_ui() or "/my-asset" not in current_url:
            with _log_step("navigate", url="/my-asset"):
                resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
                try:
                    LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
                except Exception:
                    pass
                _log_url_after_goto(page, "/my-asset.2")
            # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
            if _has_login_ui():
                LOGGER.warning("login: still on login; complete OTP/device confirm if prompted")
                page.wait_for_timeout(5000)
                login_retried = attempt_login()
                if login_retried:
                    _wait_after_login_submit("retry login after second redirect")
                with _log_step("navigate", url="/my-asset"):
                    resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
                    try:
                        LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
                    except Exception:
                        pass
                    _log_url_after_goto(page, "/my-asset.3")
            # Avoid blocking on long-lived sockets
            page.wait_for_load_state("domcontentloaded")

        # Step 2: Open tabs: 'Cổ phiếu' then 'Tài sản'
        try:
            # Try role-based first, then fallback to text locator
            page.get_by_role("tab", name=re.compile(r"cổ\s*phiếu", re.I)).first.click()
        except Exception:
            page.locator("text=Cổ phiếu").first.click()
        try:
            page.locator("text=Tài sản").first.click()
        except Exception:
            pass  # sometimes the sub-tab is default active

        # Step 3: Locate the data table
        with _log_step("locate_table", selector="table[role=table]"):
            table = page.locator("table[role=table]").first
            table.wait_for(state="visible", timeout=timeout_ms)

        # Extract headers
        headers = [h.inner_text().strip() for h in table.locator("thead th").all()]
        # Extract rows (visible only)
        body_rows = table.locator("tbody tr[role=row]").all()
        LOGGER.info("table: headers=%s rows_found=%d", headers, len(body_rows))
        rows: List[List[str]] = []
        for tr in body_rows:
            cells = tr.locator("td[role=cell]").all()
            rows.append([c.inner_text().strip() for c in cells])

        df = parse_tcbs_table(headers, rows)
        if df.empty:
            # Take a diagnostic screenshot to help debugging selectors/state
            diag_dir = (root / "out" / "diagnostics").resolve()
            diag_dir.mkdir(parents=True, exist_ok=True)
            shot = diag_dir / "tcbs_table_empty.png"
            try:
                page.screenshot(path=str(shot), full_page=True)
                LOGGER.warning("Saved diagnostic screenshot at %s", shot)
            except Exception:
                pass
            raise RuntimeError("TCBS portfolio table parsed empty; review selectors or login state")

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)

        with _log_step("write_portfolio_csv", rows=len(df), path=str(out_path)):
            df.to_csv(out_path, index=False)
        context.close()
        return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_logging_configured()
    parser = argparse.ArgumentParser(description="Fetch portfolio from TCInvest (TCBS) via Playwright")
    parser.add_argument("--headful", action="store_true", help="Run browser with UI for first-time device confirmation")
    parser.add_argument("--timeout-ms", type=int, default=300000, help="Global Playwright default timeout in milliseconds")
    parser.add_argument("--slow-mo-ms", type=int, default=None, help="Delay each Playwright action by N ms (defaults to 250 in headful; 0 in headless)")
    parser.add_argument("--post-login-wait-ms", type=int, default=None, help="Fixed time to wait after clicking login before navigating onward")
    parser.add_argument("--login-only", action="store_true", help="Only run the TCBS login flow, then exit")
    args = parser.parse_args(argv)
    # Compute default pacing for logging visibility (actual decision is inside fetch)
    computed_slow = args.slow_mo_ms if args.slow_mo_ms is not None else (250 if args.headful else 0)
    LOGGER.info(
        "args: headful=%s timeout_ms=%d slow_mo_ms=%s post_login_wait_ms=%s login_only=%s",
        args.headful,
        int(args.timeout_ms),
        computed_slow,
        args.post_login_wait_ms,
        args.login_only,
    )
    p_path = fetch_tcbs_portfolio(
        headless=not args.headful,
        timeout_ms=int(args.timeout_ms),
        slow_mo_ms=args.slow_mo_ms,
        post_login_wait_ms=args.post_login_wait_ms,
        login_only=args.login_only,
    )
    if p_path is None:
        LOGGER.info("tcbs_login done")
        return 0
    LOGGER.info("portfolio_csv=%s", p_path)
    print(str(p_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
