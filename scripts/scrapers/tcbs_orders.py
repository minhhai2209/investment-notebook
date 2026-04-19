"""Batch place TCBS stock orders via Playwright Chrome using a CSV input.

This ports the core behaviour of ../broker-extension/orderBatch.js into Python:

- Parse a CSV of orders (header-aware, same schema as extension).
- For each valid row: ensure TCBS order form is visible, fill ticker/volume/price,
  click Mua/Bán, then confirm nếu có dialog xác nhận.
- Nếu xuất hiện dialog lỗi "Giá không nằm trong khoảng trần sàn" sau khi submit,
  script sẽ bấm nút "Đóng" và ghi lại mã lỗi vào một file dưới out/.

Lệnh CLI (ví dụ):

    python -m scripts.scrapers.tcbs_orders \\
        --csv codex_universe/orders.csv \\
        --headful

Hoặc thông qua broker.sh:

    ./broker.sh tcbs_orders --csv codex_universe/orders.csv --headful
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from playwright.sync_api import TimeoutError as PwTimeout
from playwright.sync_api import sync_playwright

from scripts.scrapers import tcbs as tcbs_scraper


LOGGER = logging.getLogger("tcbs_orders")
DEFAULT_LOGIN_APPROVAL_WAIT_MS = int(os.environ.get("TCBS_ORDERS_LOGIN_APPROVAL_WAIT_MS", "120000"))
DEFAULT_CODEX_DIR = "codex_universe"


@dataclass
class OrderRow:
    ticker: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    signal: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ParsedOrders:
    orders: List[OrderRow]
    source_csv: Path


def _orders_user_data_root(root: Path) -> Path:
    return (root / ".playwright" / "tcbs-orders-user-data").resolve()


def _normalize_header_name(header: str) -> str:
    return (header or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _build_header_schema(headers: List[str]) -> Dict[str, int]:
    def find_index(candidates: Sequence[str]) -> int:
        for cand in candidates:
            try:
                idx = headers.index(cand)
            except ValueError:
                continue
            else:
                return idx
        return -1

    return {
        "ticker": find_index(["ticker", "symbol"]),
        "side": find_index(["side", "action"]),
        "quantity": find_index(["quantity", "qty", "soluong"]),
        "limit_price": find_index(["limitprice", "price", "limit"]),
        "market_price": find_index(["marketprice", "mp"]),
        "signal": find_index(["signal"]),
        "notes": find_index(["notes", "note"]),
    }


def _build_legacy_schema(num_cols: int) -> Dict[str, int]:
    # Legacy format: Ticker,Side,Quantity,LimitPrice,...
    return {
        "ticker": 0 if num_cols > 0 else -1,
        "side": 1 if num_cols > 1 else -1,
        "quantity": 2 if num_cols > 2 else -1,
        "limit_price": 3 if num_cols > 3 else -1,
        "market_price": -1,
        "signal": -1,
        "notes": -1,
    }


def _get_column(parts: List[str], index: int) -> str:
    if index is None or index < 0:
        return ""
    if index >= len(parts):
        return ""
    return (parts[index] or "").strip()


def _normalize_side(raw: str) -> str:
    s = (raw or "").strip().upper()
    if s in {"BUY", "MUA"}:
        return "BUY"
    if s in {"SELL", "BAN", "BÁN"}:
        return "SELL"
    return ""


def _sanitize_price(val: str) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = (val or "").replace(" ", "").replace(",", "")
    if not cleaned:
        return math.nan
    try:
        return float(cleaned)
    except Exception:
        return math.nan


def parse_orders_csv(csv_path: Path) -> ParsedOrders:
    if not csv_path.exists():
        raise RuntimeError(f"Orders CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    rows = [[cell.strip() for cell in row] for row in reader if any((cell or "").strip() for cell in row)]
    if not rows:
        raise RuntimeError(f"Orders CSV is empty or only whitespace: {csv_path}")

    first = rows[0]
    header_tokens = [_normalize_header_name(h) for h in first]
    has_header = "ticker" in header_tokens and "side" in header_tokens
    schema = _build_header_schema(header_tokens) if has_header else _build_legacy_schema(len(first))
    start_index = 1 if has_header else 0

    orders: List[OrderRow] = []
    for i in range(start_index, len(rows)):
        parts = rows[i]
        ticker = _get_column(parts, schema["ticker"]).upper()
        side = _normalize_side(_get_column(parts, schema["side"]))
        qty_raw = _get_column(parts, schema["quantity"]).replace(",", "").replace(" ", "")
        try:
            quantity = int(qty_raw)
        except Exception:
            continue

        limit_price = _sanitize_price(_get_column(parts, schema["limit_price"]))
        market_price = _sanitize_price(_get_column(parts, schema["market_price"])) if schema["market_price"] >= 0 else math.nan
        price = limit_price if math.isfinite(limit_price) else market_price

        if not ticker or not side or not math.isfinite(price) or quantity <= 0:
            continue

        signal = _get_column(parts, schema["signal"]) if schema["signal"] >= 0 else ""
        notes = _get_column(parts, schema["notes"]) if schema["notes"] >= 0 else ""

        orders.append(
            OrderRow(
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=float(price),
                signal=signal or None,
                notes=notes or None,
            )
        )

    if not orders:
        raise RuntimeError(f"No valid orders parsed from CSV: {csv_path}")

    LOGGER.info("Parsed %d orders from %s", len(orders), csv_path)
    return ParsedOrders(orders=orders, source_csv=csv_path)


def snapshot_orders_csv(csv_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_path = out_dir / f"orders_snapshot_{timestamp}.csv"
    snapshot_path.write_bytes(csv_path.read_bytes())
    return snapshot_path


def _has_login_ui(page) -> bool:
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


def _has_order_form_ui(page) -> bool:
    selectors = [
        "input[formcontrolname='ticker']",
        "input[name='ticker']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                return True
        except Exception:
            continue
    return False


def _wait_for_order_form_or_login(page, timeout_ms: int) -> str:
    deadline = time.time() + max(1.0, min(12.0, timeout_ms / 1000.0))
    while time.time() < deadline:
        if _has_order_form_ui(page):
            return "order_form"
        if _has_login_ui(page):
            return "login"
        try:
            page.wait_for_timeout(300)
        except Exception:
            time.sleep(0.1)
    return "unknown"


def _wait_for_login_session_ready(page, timeout_ms: int, approval_wait_ms: int | None = None) -> str:
    max_wait_ms = approval_wait_ms if approval_wait_ms is not None else DEFAULT_LOGIN_APPROVAL_WAIT_MS
    max_wait_ms = max(1000, min(timeout_ms, max_wait_ms))
    poll_ms = max(200, min(1000, max_wait_ms // 20))
    LOGGER.info(
        "login: waiting up to %.1fs for MFA approval or session transition",
        max_wait_ms / 1000.0,
    )
    deadline = time.time() + (max_wait_ms / 1000.0)
    while time.time() < deadline:
        if _has_order_form_ui(page):
            return "order_form"
        if not _has_login_ui(page):
            return "session_ready"
        try:
            page.wait_for_timeout(poll_ms)
        except Exception:
            time.sleep(min(0.2, poll_ms / 1000.0))
    return "login"


def _logout_if_logged_in(page, timeout_ms: int) -> None:
    """If a previous session is still logged in, log out first."""
    try:
        if _has_login_ui(page):
            LOGGER.info("logout_check: login UI visible; assuming already logged out")
            return
        LOGGER.info("logout_check: attempting to open user menu for logout")

        def _click_menu_trigger() -> bool:
            selectors = [
                "div.mat-ripple[cdkoverlayorigin]",
                "div[cdkoverlayorigin]",
            ]
            for sel in selectors:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible():
                        LOGGER.info("logout_check: clicking menu trigger selector=%s", sel)
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

        LOGGER.info("logout_check: clicking logout button")
        logout_btn.click()
        # Wait briefly for the UI to transition back to login
        deadline = time.time() + max(3.0, timeout_ms / 1000.0)
        while time.time() < deadline:
            try:
                page.wait_for_timeout(500)
            except Exception:
                pass
            if _has_login_ui(page):
                LOGGER.info("logout_check: login UI detected after logout")
                return
        LOGGER.info("logout_check: logout click sent but login UI not detected; continuing")
    except Exception:
        LOGGER.exception("logout_check: error while trying to log out")


def _attempt_login(page, username: str, password: str, timeout_ms: int, slow_mo_ms: int) -> None:
    LOGGER.info("login: attempt begin")
    login_pace_ms = 400 if slow_mo_ms == 0 else max(200, min(500, int(slow_mo_ms)))
    login_poll_ms = max(150, min(400, login_pace_ms // 2))

    def _pace(mult: int = 1) -> None:
        try:
            page.wait_for_timeout(login_pace_ms * max(1, mult))
        except Exception:
            pass

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
        return

    with _log_step("login_fill"):
        user_input.wait_for(state="visible", timeout=timeout_ms)
        user_input.click()
        _pace()
        user_input.fill(username)
        _pace()
        pass_input.click()
        _pace()
        pass_input.fill(password)
        _pace(mult=2)

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
            for _ in range(20):
                if not _has_login_ui(page):
                    break
                try:
                    page.wait_for_timeout(login_poll_ms)
                except Exception:
                    pass
            break
        except Exception:
            continue


def _ensure_authenticated(page, username: str, password: str, timeout_ms: int, slow_mo_ms: int) -> None:
    if _has_login_ui(page):
        LOGGER.info("auth: login UI visible; performing fresh login")
        _attempt_login(page, username, password, timeout_ms, slow_mo_ms)
        login_state = _wait_for_login_session_ready(page, timeout_ms)
        LOGGER.info("auth: login_state_after_submit=%s", login_state)
        return
    LOGGER.info("auth: login UI not visible; keeping current session and skipping forced logout")


def _log_step(name: str, **fields: object):
    meta = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
    LOGGER.info("▶ %s%s", name, (" " + meta) if meta else "")
    t0 = time.perf_counter()

    class _Ctx:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            dt = time.perf_counter() - t0
            if exc is not None:
                LOGGER.exception("✖ %s failed after %.3fs", name, dt)
            else:
                LOGGER.info("✓ %s done in %.3fs", name, dt)
            # Do not suppress exceptions
            return False

    return _Ctx()


def _ensure_on_order_form(page, timeout_ms: int) -> None:
    """Ensure the TCBS order form (ticker/volume/price inputs) is present."""
    selectors = "input[formcontrolname='ticker'], input[name='ticker']"
    with _log_step("ensure_order_form", selector=selectors):
        try:
            _dismiss_any_open_dialog(page, timeout_ms=min(timeout_ms, 2000))
        except Exception:
            LOGGER.exception("Failed to dismiss blocking dialog before waiting for order form")
        page.wait_for_selector(selectors, timeout=timeout_ms)


def _focus_and_fill_input(page, selector: str, value: str, timeout_ms: int) -> None:
    loc = page.locator(selector).first
    wait_timeout_ms = min(timeout_ms, 15000)
    click_timeout_ms = min(timeout_ms, 5000)
    loc.wait_for(state="visible", timeout=wait_timeout_ms)
    deadline = time.time() + (wait_timeout_ms / 1000.0)
    while time.time() < deadline:
        try:
            if loc.is_enabled():
                break
        except Exception:
            pass
        time.sleep(0.2)
    try:
        loc.scroll_into_view_if_needed(timeout=click_timeout_ms)
    except Exception:
        pass
    try:
        loc.focus(timeout=click_timeout_ms)
    except Exception:
        try:
            loc.evaluate("el => el.focus()")
        except Exception:
            pass
    try:
        loc.click(timeout=click_timeout_ms)
    except Exception:
        LOGGER.warning("Input click timed out for selector=%s; retrying with force click", selector)
        try:
            loc.click(timeout=min(timeout_ms, 2000), force=True)
        except Exception:
            LOGGER.warning("Force click also failed for selector=%s; continuing with direct fill fallback", selector)
    # TCBS sometimes reuses the previous ticker; clear aggressively before typing.
    try:
        loc.fill("")
    except Exception:
        pass
    for combo in ("Control+A", "Meta+A"):
        try:
            loc.press(combo)
            loc.press("Backspace")
        except Exception:
            pass
    try:
        loc.evaluate(
            "el => { el.value = ''; el.dispatchEvent(new Event('input', { bubbles: true }));"
            " el.dispatchEvent(new Event('change', { bubbles: true })); }"
        )
    except Exception:
        pass
    try:
        loc.fill(value, timeout=click_timeout_ms)
    except Exception:
        loc.type(value, delay=50, timeout=click_timeout_ms)


def _select_autocomplete_if_present(page, ticker: str, timeout_ms: int) -> None:
    try:
        panel = page.wait_for_selector(
            ".mat-autocomplete-panel, .cdk-overlay-pane .mat-autocomplete-panel",
            timeout=timeout_ms,
        )
    except PwTimeout:
        try:
            active = page.evaluate_handle("document.activeElement")
            active.as_element().press("Enter")  # type: ignore[union-attr]
        except Exception:
            pass
        return

    options = panel.locator(".mat-option-text, mat-option")
    count = options.count()
    target = None
    for i in range(count):
        el = options.nth(i)
        txt = (el.inner_text() or "").strip().upper()
        if txt.startswith(ticker.upper()):
            target = el
            break
    if target is None and count:
        target = options.first
    if target is not None:
        el = target
        try:
            el.click()
        except Exception:
            pass


def _submit_order(page, side: str, timeout_ms: int) -> None:
    btn = None
    if side == "BUY":
        # Prefer dedicated buy buttons, fallback to any matching text
        for sel in ["button.btn.btn-buy", "button.btn-buy"]:
            loc = page.locator(sel)
            if loc.count() > 0:
                btn = loc.first
                break
        if btn is None:
            btn = page.get_by_role("button", name=re.compile(r"\bmua\b", re.I)).first
    else:
        for sel in ["button.btn.btn-sell", "button.btn-sell"]:
            loc = page.locator(sel)
            if loc.count() > 0:
                btn = loc.first
                break
        if btn is None:
            btn = page.get_by_role("button", name=re.compile(r"\bbán\b|\bban\b", re.I)).first

    if btn is None:
        raise RuntimeError(f"Không tìm thấy nút {'Mua' if side == 'BUY' else 'Bán'}")

    try:
        _dismiss_any_open_dialog(page, timeout_ms=min(timeout_ms, 1500))
    except Exception:
        pass

    # Wait until enabled and then click
    end = time.time() + timeout_ms / 1000.0
    while time.time() < end:
        try:
            disabled = btn.is_disabled()
        except Exception:
            disabled = False
        if not disabled:
            break
        time.sleep(0.2)

    last_error: Optional[Exception] = None
    while time.time() < end:
        remaining_ms = max(250, int((end - time.time()) * 1000))
        try:
            btn.scroll_into_view_if_needed(timeout=min(remaining_ms, 2000))
        except Exception:
            pass
        try:
            btn.click(timeout=min(remaining_ms, 2000))
            return
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "intercepts pointer events" in message or "subtree intercepts pointer events" in message:
                try:
                    _dismiss_any_open_dialog(page, timeout_ms=min(remaining_ms, 4000))
                except Exception:
                    pass
                try:
                    page.wait_for_timeout(300)
                except Exception:
                    time.sleep(0.3)
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Timed out clicking {'Mua' if side == 'BUY' else 'Bán'} button")


_DIALOG_CONFIRM_RE = re.compile(r"xác\s*nhận|xac\s*nhan|confirm", re.I)
_DIALOG_CLOSE_RE = re.compile(r"đóng|dong|ok|close|hủy|huy|cancel", re.I)
_DIALOG_REJECT_RE = re.compile(r"từ\s*chối|tu\s*choi|reject", re.I)
_REGISTER_NOTIFICATION_DIALOG_RE = re.compile(
    r"chưa\s*đăng\s*ký\s*nhận\s*thông\s*báo|nhận\s*thông\s*báo\s*từ\s*tcinvest",
    re.I,
)
_ORDER_ERROR_DIALOG_RE = re.compile(
    r"trần\s*sàn|tran\s*san|không\s*đủ|khong\s*du|sức\s*mua|"
    r"chứng\s*khoán|chung\s*khoan|vượt\s*quá|vuot\s*qua|"
    r"không\s*hợp\s*lệ|khong\s*hop\s*le|thất\s*bại|that\s*bai|"
    r"lỗi|loi|error|failed",
    re.I,
)
_POST_CONFIRM_ERROR_GRACE_SEC = 0.8
_DIALOG_CONTAINER_SELECTORS: Sequence[str] = (
    "mat-dialog-container",
    ".mat-mdc-dialog-container",
    "[role='dialog'], [role='alertdialog']",
    ".cdk-overlay-pane",
    ".cdk-global-overlay-wrapper",
)
_REGISTER_NOTIFICATION_REJECT_SELECTORS: Sequence[str] = (
    "button.reject",
    "button.mat-flat-button.reject",
    "button.mat-mdc-unelevated-button.reject",
    "button:has-text('TỪ CHỐI')",
    "[role='button']:has-text('TỪ CHỐI')",
    ".cdk-overlay-container button:has-text('TỪ CHỐI')",
    ".cdk-overlay-container [role='button']:has-text('TỪ CHỐI')",
    ".cdk-overlay-pane button:has-text('TỪ CHỐI')",
    ".cdk-overlay-pane [role='button']:has-text('TỪ CHỐI')",
    ".mat-mdc-dialog-container button:has-text('TỪ CHỐI')",
    "button .mat-button-wrapper:text-is('TỪ CHỐI')",
)


def _chrome_launch_args() -> List[str]:
    return [
        "--disable-blink-features=AutomationControlled",
        "--disable-http-cache",
        # TCInvest sometimes triggers native notification permission prompts; auto-deny them.
        "--disable-notifications",
        "--deny-permission-prompts",
    ]


def _normalize_dialog_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _is_known_order_error_dialog(reason: Optional[str]) -> bool:
    normalized = _normalize_dialog_text(reason or "")
    if not normalized:
        return False
    return bool(_ORDER_ERROR_DIALOG_RE.search(normalized))


def _first_visible_dialog(page):
    """Return the first visible TCBS dialog-like container, or None."""
    candidates = [page.locator(selector) for selector in _DIALOG_CONTAINER_SELECTORS]
    for cand in candidates:
        try:
            count = cand.count()
        except Exception:
            continue
        # Limit scan to avoid pathological cases with many overlays.
        for i in range(min(count, 6)):
            loc = cand.nth(i)
            try:
                if loc.is_visible():
                    return loc
            except Exception:
                continue
    return None


def _wait_dialog_gone(page, dialog, timeout_ms: int) -> None:
    """Best-effort wait until the specific dialog (and its backdrop) is gone."""
    try:
        dialog.wait_for(state="hidden", timeout=timeout_ms)
    except Exception:
        pass
    try:
        page.locator(".cdk-overlay-backdrop.cdk-overlay-backdrop-showing").first.wait_for(state="hidden", timeout=timeout_ms)
    except Exception:
        pass


def _click_dialog_button(dialog, name_re: re.Pattern[str]) -> bool:
    """Click the first visible button matching name regex inside dialog; return True if clicked."""
    try:
        btn = dialog.get_by_role("button", name=name_re)
        if btn.count() > 0:
            target = btn.first
            try:
                target.scroll_into_view_if_needed()
            except Exception:
                pass
            try:
                target.click(timeout=2000)
                return True
            except Exception:
                try:
                    target.click(timeout=1000, force=True)
                    return True
                except Exception:
                    try:
                        target.evaluate("el => el.click()")
                        return True
                    except Exception:
                        pass
    except Exception:
        pass
    return False


def _click_register_notification_reject(scope, page=None) -> bool:
    scopes = [scope]
    if page is not None and page is not scope:
        scopes.append(page)

    for current_scope in scopes:
        for selector in _REGISTER_NOTIFICATION_REJECT_SELECTORS:
            try:
                btn = current_scope.locator(selector)
                if btn.count() <= 0:
                    continue
                target = btn.first
                try:
                    target.scroll_into_view_if_needed()
                except Exception:
                    pass
                try:
                    target.click(timeout=2000)
                    return True
                except Exception:
                    try:
                        target.click(timeout=1000, force=True)
                        return True
                    except Exception:
                        try:
                            target.evaluate("el => el.click()")
                            return True
                        except Exception:
                            pass
            except Exception:
                continue

    for current_scope in scopes:
        if _click_dialog_button(current_scope, _DIALOG_REJECT_RE):
            return True

    if page is not None:
        try:
            clicked = page.evaluate(
                """() => {
                    const norm = (text) => (text || "").replace(/\\s+/g, " ").trim().toLowerCase();
                    const isVisible = (el) => {
                        if (!el) return false;
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
                    };
                    const buttons = Array.from(
                        document.querySelectorAll(
                            ".cdk-overlay-container button, .cdk-overlay-container [role='button'], button.reject, [role='button'].reject"
                        )
                    );
                    const target = buttons.find((el) => {
                        if (!isVisible(el)) return false;
                        return norm(el.innerText || el.textContent || "").includes("từ chối");
                    });
                    if (!target) return false;
                    target.click();
                    return true;
                }"""
            )
            if clicked:
                return True
        except Exception:
            pass

    return False


def _dismiss_any_open_dialog(page, timeout_ms: int) -> Optional[str]:
    """Dismiss any currently visible blocking dialog; return its normalized text if dismissed."""
    end = time.time() + timeout_ms / 1000.0
    last_reason: Optional[str] = None

    while time.time() < end:
        dialog = _first_visible_dialog(page)
        if dialog is None:
            return last_reason

        text = ""
        try:
            text = dialog.inner_text() or ""
        except Exception:
            pass
        last_reason = _normalize_dialog_text(text) or last_reason or "Unknown dialog"

        if _REGISTER_NOTIFICATION_DIALOG_RE.search(last_reason or ""):
            LOGGER.info("Detected register-notification dialog; clicking reject")
            if _click_register_notification_reject(dialog, page=page):
                _wait_dialog_gone(page, dialog, timeout_ms=4000)
                continue

        # Prefer explicit actions if available.
        if _click_dialog_button(dialog, _DIALOG_CONFIRM_RE):
            _wait_dialog_gone(page, dialog, timeout_ms=4000)
            continue
        if _click_dialog_button(dialog, _DIALOG_CLOSE_RE):
            _wait_dialog_gone(page, dialog, timeout_ms=4000)
            continue
        try:
            page.keyboard.press("Escape")
        except Exception:
            break
        _wait_dialog_gone(page, dialog, timeout_ms=4000)

    raise RuntimeError(f"Blocking dialog persisted after {timeout_ms}ms: {last_reason}")


def _handle_confirm_or_error(
    page,
    order: OrderRow,
    timeout_ms: int,
    error_orders: List[Tuple[OrderRow, str]],
    confirmed_orders: List[OrderRow],
) -> str:
    """After submit, either confirm the order, detect an error dialog, or time out."""
    end = time.time() + timeout_ms / 1000.0
    confirm_clicked_at: Optional[float] = None

    while time.time() < end:
        # If we already clicked confirm and no error dialog appears shortly after,
        # consider the order confirmed.
        if confirm_clicked_at is not None and (time.time() - confirm_clicked_at) >= _POST_CONFIRM_ERROR_GRACE_SEC:
            confirmed_orders.append(order)
            return "confirmed"

        # Check for any dialog first (TCBS uses Angular Material mat-dialog)
        try:
            dialog = _first_visible_dialog(page)
            if dialog is not None:
                text = ""
                try:
                    text = dialog.inner_text() or ""
                except Exception:
                    pass
                reason = _normalize_dialog_text(text)

                if _REGISTER_NOTIFICATION_DIALOG_RE.search(reason or ""):
                    LOGGER.info("Order flow hit register-notification dialog for %s; rejecting popup", order.ticker)
                    if _click_register_notification_reject(dialog, page=page):
                        _wait_dialog_gone(page, dialog, timeout_ms=4000)
                        continue

                # If there's a confirm button, treat as confirmation flow.
                if _click_dialog_button(dialog, _DIALOG_CONFIRM_RE):
                    LOGGER.info("Confirmed order for %s via dialog.", order.ticker)
                    _wait_dialog_gone(page, dialog, timeout_ms=4000)
                    confirm_clicked_at = time.time()
                    continue

                if confirm_clicked_at is not None and not _is_known_order_error_dialog(reason):
                    LOGGER.info(
                        "Order for %s hit post-confirm cleanup dialog; dismissing without marking failure (%s)",
                        order.ticker,
                        reason or "Unknown dialog",
                    )
                    if not _click_dialog_button(dialog, _DIALOG_CLOSE_RE):
                        try:
                            page.keyboard.press("Escape")
                        except Exception:
                            LOGGER.warning("Could not dismiss post-confirm dialog for %s", order.ticker)
                    _wait_dialog_gone(page, dialog, timeout_ms=4000)
                    continue

                # Otherwise, treat as an error/informational dialog that blocks the UI.
                LOGGER.warning("Order for %s reported blocking dialog; dismissing", order.ticker)
                error_orders.append((order, reason or "Unknown dialog error"))
                if not _click_dialog_button(dialog, _DIALOG_CLOSE_RE):
                    try:
                        page.keyboard.press("Escape")
                    except Exception:
                        LOGGER.warning("Could not dismiss dialog (no close button) for %s", order.ticker)
                _wait_dialog_gone(page, dialog, timeout_ms=4000)
                return "error"
        except Exception:
            # If dialog locator fails, continue checking confirm
            pass

        # Then look for generic confirm button
        try:
            container = page.locator(".cdk-overlay-container")
            btns = container.locator("button, .mat-button, .mat-raised-button, .mat-flat-button")
            count = btns.count()
            for i in range(count):
                el = btns.nth(i)
                txt = (el.inner_text() or "").strip()
                if re.search(r"xác\s*nhận", txt, re.I):
                    LOGGER.info("Found confirm button for %s; clicking.", order.ticker)
                    el.click()
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        pass
                    confirm_clicked_at = time.time()
                    break
        except Exception:
            pass

        try:
            page.wait_for_timeout(300)
        except Exception:
            pass

    if confirm_clicked_at is not None:
        confirmed_orders.append(order)
        return "confirmed"

    LOGGER.info("No confirm/error dialog observed for %s within timeout", order.ticker)
    return "none"


def place_orders_from_csv(
    csv_path: Path,
    headless: bool = True,
    timeout_ms: int = 300000,
    slow_mo_ms: Optional[int] = None,
) -> Tuple[int, int, Path]:
    """Place all orders from CSV; return (success_count, total_count, invalid_log_path)."""
    parsed = parse_orders_csv(csv_path)

    root = tcbs_scraper._repo_root()
    LOGGER.info("repo_root=%s", root)
    tcbs_scraper._load_env_if_present(root)
    username = tcbs_scraper._require_env("TCBS_USERNAME")
    password = tcbs_scraper._require_env("TCBS_PASSWORD")

    pybin = sys.executable
    tcbs_scraper._ensure_playwright_installed(pybin)

    user_data_root = _orders_user_data_root(root)
    user_data_root.mkdir(parents=True, exist_ok=True)

    error_orders: List[Tuple[OrderRow, str]] = []
    confirmed_orders: List[OrderRow] = []
    success = 0

    with tempfile.TemporaryDirectory(prefix="session-", dir=str(user_data_root)) as tmp_user_data_dir:
        user_data_dir = Path(tmp_user_data_dir).resolve()
        LOGGER.info(
            "user_data_dir=%s headless=%s timeout_ms=%d ephemeral=true",
            user_data_dir,
            headless,
            timeout_ms,
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
                    args=_chrome_launch_args(),
                )

            try:
                with _log_step("launch_chrome", headless=headless):
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
                    with _log_step("launch_chrome_retry", reason="singleton lock"):
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

            # Step 0: Open home; if login UI is visible then log in, otherwise keep the current session.
            with _log_step("navigate_home"):
                resp = page.goto("https://tcinvest.tcbs.com.vn/home", wait_until="commit")
                try:
                    LOGGER.info(
                        "goto_response_url=%s status=%s",
                        getattr(resp, "url", None),
                        getattr(resp, "status", None),
                    )
                except Exception:
                    pass
                tcbs_scraper._log_url_after_goto(page, "/home.orders")
            _ensure_authenticated(page, username, password, timeout_ms, sm or 0)

            # Navigate thẳng tới trang đặt lệnh cổ phiếu trước khi thao tác form.
            with _log_step("navigate_placing_stock"):
                try:
                    resp = page.goto("https://tcinvest.tcbs.com.vn/placing-stock", wait_until="commit")
                    LOGGER.info(
                        "placing_stock_goto_url=%s status=%s",
                        getattr(resp, "url", None),
                        getattr(resp, "status", None),
                    )
                except Exception:
                    LOGGER.exception("Failed to navigate to /placing-stock; will try fallback")
                tcbs_scraper._log_url_after_goto(page, "/placing-stock.orders")

            auth_state = _wait_for_order_form_or_login(page, timeout_ms=min(timeout_ms, 15000))
            LOGGER.info("auth_state_after_placing_stock=%s", auth_state)
            if auth_state == "login":
                LOGGER.warning("login: order page still shows login UI; complete OTP/device confirm if prompted")
                _attempt_login(page, username, password, timeout_ms, sm or 0)
                login_state = _wait_for_login_session_ready(page, timeout_ms)
                LOGGER.info("login_state_after_order_page_submit=%s", login_state)
                with _log_step("navigate_placing_stock_after_login"):
                    try:
                        resp = page.goto("https://tcinvest.tcbs.com.vn/placing-stock", wait_until="commit")
                        LOGGER.info(
                            "placing_stock_after_login_url=%s status=%s",
                            getattr(resp, "url", None),
                                getattr(resp, "status", None),
                        )
                    except Exception:
                        LOGGER.exception("Failed to navigate to /placing-stock after login")
                    tcbs_scraper._log_url_after_goto(page, "/placing-stock.after-login.orders")

            try:
                _dismiss_any_open_dialog(page, timeout_ms=2000)
            except Exception:
                LOGGER.exception("Failed to dismiss blocking dialog after auth/navigation")

            # Best-effort: đảm bảo form đặt lệnh đã render trên trang placing-stock.
            with _log_step("ensure_order_form"):
                if not _has_order_form_ui(page):
                    _ensure_on_order_form(page, timeout_ms=timeout_ms)

            # Process each order
            for idx, order in enumerate(parsed.orders, start=1):
                LOGGER.info(
                    "Processing order %d/%d: %s %s @ %s",
                    idx,
                    len(parsed.orders),
                    order.ticker,
                    order.quantity,
                    order.price,
                )
                try:
                    # If any dialog from previous step is still open, it will block the form.
                    try:
                        _dismiss_any_open_dialog(page, timeout_ms=1500)
                    except Exception:
                        LOGGER.exception("Failed to dismiss blocking dialog before processing %s", order.ticker)

                    with _log_step("fill_order", ticker=order.ticker, side=order.side, qty=order.quantity):
                        # Ticker
                        _focus_and_fill_input(
                            page, "input[formcontrolname='ticker'], input[name='ticker']", order.ticker, timeout_ms
                        )
                        try:
                            page.wait_for_timeout(800)
                        except Exception:
                            pass
                        try:
                            _select_autocomplete_if_present(page, order.ticker, timeout_ms=1500)
                            page.wait_for_timeout(1200)
                        except Exception:
                            pass

                        # Volume
                        _focus_and_fill_input(
                            page,
                            "input[formcontrolname='volume'], input[name='volume']",
                            str(order.quantity),
                            timeout_ms,
                        )
                        try:
                            page.wait_for_timeout(500)
                        except Exception:
                            pass

                        # Price
                        price_str = f"{order.price:.2f}"
                        _focus_and_fill_input(
                            page,
                            "input[formcontrolname='price'], input[name='price']",
                            price_str,
                            timeout_ms,
                        )
                        try:
                            page.wait_for_timeout(500)
                        except Exception:
                            pass

                    with _log_step("submit_order", ticker=order.ticker, side=order.side):
                        _submit_order(page, order.side, timeout_ms=timeout_ms)
                        try:
                            page.wait_for_timeout(2000)
                        except Exception:
                            pass
                        status = _handle_confirm_or_error(
                            page,
                            order,
                            timeout_ms=15000,
                            error_orders=error_orders,
                            confirmed_orders=confirmed_orders,
                        )
                    if status == "confirmed":
                        success += 1
                    elif status == "none":
                        # We did not observe confirm nor error dialog. Outcome is unknown; log it for review.
                        error_orders.append(
                            (
                                order,
                                f"idx={idx}/{len(parsed.orders)}: No confirm/error dialog observed within timeout",
                            )
                        )
                except Exception:
                    exc = sys.exc_info()[1]
                    LOGGER.exception("Order for %s failed", order.ticker)
                    reason = f"idx={idx}/{len(parsed.orders)}: Exception while placing order"
                    if exc is not None:
                        reason = f"{reason}: {type(exc).__name__}: {exc}"
                    # Best-effort: close any error dialog so subsequent orders can continue and capture its text.
                    try:
                        dlg_reason = _dismiss_any_open_dialog(page, timeout_ms=6000)
                        if dlg_reason:
                            reason = f"{reason} | Dialog: {dlg_reason}"
                    except Exception as dlg_exc:
                        reason = f"{reason} | DialogDismissFailed: {type(dlg_exc).__name__}: {dlg_exc}"
                    error_orders.append((order, reason))
                    # continue with next order
                    continue

            context.close()

    # Write error orders (if any) to out/
    out_dir = (root / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "tcbs_error_orders.csv"
    with error_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "Side", "Quantity", "Price", "Reason"])
        for o, reason in error_orders:
            writer.writerow([o.ticker, o.side, o.quantity, f"{o.price:.2f}", reason])
    if error_orders:
        LOGGER.warning("Wrote %d error orders to %s", len(error_orders), error_path)
    else:
        LOGGER.info("Wrote 0 error orders to %s", error_path)

    confirmed_path = out_dir / "tcbs_confirmed_orders.csv"
    with confirmed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "Side", "Quantity", "Price"])
        for o in confirmed_orders:
            writer.writerow([o.ticker, o.side, o.quantity, f"{o.price:.2f}"])
    LOGGER.info("Wrote %d confirmed orders to %s", len(confirmed_orders), confirmed_path)

    return success, len(parsed.orders), error_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place TCBS stock orders from codex_universe/orders.csv using Playwright (Chrome).")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the orders CSV. Defaults to $CODEX_DIR/orders.csv or codex_universe/orders.csv.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Chạy Chrome với UI để quan sát (mặc định headless).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=300000,
        help="Playwright default timeout (ms).",
    )
    parser.add_argument(
        "--slow-mo-ms",
        type=int,
        default=None,
        help="Delay mỗi action Playwright (ms). Mặc định 250 ở headful, 0 ở headless.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    args = _parse_args(argv)
    root = tcbs_scraper._repo_root()
    codex_dir = os.environ.get("CODEX_DIR", DEFAULT_CODEX_DIR).strip() or DEFAULT_CODEX_DIR
    default_csv = root / codex_dir / "orders.csv"
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else default_csv.resolve()
    snapshot_dir = (root / "out" / "orders_snapshots").resolve()
    snapshot_path = snapshot_orders_csv(csv_path, snapshot_dir)
    LOGGER.info("Orders CSV: %s", csv_path)
    LOGGER.info("Orders snapshot: %s", snapshot_path)
    success, total, error_path = place_orders_from_csv(
        csv_path=snapshot_path,
        headless=not args.headful,
        timeout_ms=int(args.timeout_ms),
        slow_mo_ms=args.slow_mo_ms,
    )
    LOGGER.info("Finished placing orders: %d/%d succeeded", success, total)
    if error_path.exists():
        LOGGER.info("Error orders logged at: %s", error_path)
    print(f"{success}/{total} orders processed. Error log: {error_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
