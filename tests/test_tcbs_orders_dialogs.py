from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.scrapers import tcbs_orders


class TcbsOrdersDialogTest(unittest.TestCase):
    class _FakeKeyboard:
        def __init__(self) -> None:
            self.pressed: list[str] = []

        def press(self, key: str) -> None:
            self.pressed.append(key)

    class _FakeButton:
        def __init__(self, owner: "TcbsOrdersDialogTest._FakeDialog", name: str) -> None:
            self.owner = owner
            self.name = name

        def scroll_into_view_if_needed(self) -> None:
            return None

        def click(self, *args, **kwargs) -> None:
            self.owner.clicked_buttons.append(self.name)

        def evaluate(self, _script: str) -> None:
            self.owner.clicked_buttons.append(self.name)

    class _FakeButtonList:
        def __init__(self, buttons: list["TcbsOrdersDialogTest._FakeButton"]) -> None:
            self._buttons = buttons

        def count(self) -> int:
            return len(self._buttons)

        @property
        def first(self) -> "TcbsOrdersDialogTest._FakeButton":
            return self._buttons[0]

    class _FakeDialog:
        def __init__(self, text: str, buttons: list[str], role_query_works: bool = True, locator_query_works: bool = True) -> None:
            self._text = text
            self._buttons = buttons
            self.clicked_buttons: list[str] = []
            self.role_query_works = role_query_works
            self.locator_query_works = locator_query_works

        def inner_text(self) -> str:
            return self._text

        def wait_for(self, state: str, timeout: int) -> None:
            return None

        def get_by_role(self, role: str, name) -> "TcbsOrdersDialogTest._FakeButtonList":
            assert role == "button"
            if not self.role_query_works:
                raise RuntimeError("role query failed")
            matched = [
                TcbsOrdersDialogTest._FakeButton(self, button_name)
                for button_name in self._buttons
                if name.search(button_name)
            ]
            return TcbsOrdersDialogTest._FakeButtonList(matched)

        def locator(self, selector: str) -> "TcbsOrdersDialogTest._FakeButtonList":
            if not self.locator_query_works:
                raise RuntimeError("locator query failed")
            selector_upper = selector.upper()
            matched = []
            for button_name in self._buttons:
                button_upper = button_name.upper()
                if "REJECT" in selector_upper and "TỪ CHỐI" in button_upper:
                    matched.append(TcbsOrdersDialogTest._FakeButton(self, button_name))
                elif "TỪ CHỐI" in selector_upper and "TỪ CHỐI" in button_upper:
                    matched.append(TcbsOrdersDialogTest._FakeButton(self, button_name))
            return TcbsOrdersDialogTest._FakeButtonList(matched)

    class _FakeOverlayWaiter:
        def wait_for(self, state: str, timeout: int) -> None:
            return None

    class _FakeOverlayLocator:
        @property
        def first(self) -> "TcbsOrdersDialogTest._FakeOverlayWaiter":
            return TcbsOrdersDialogTest._FakeOverlayWaiter()

    class _FakePage:
        def __init__(self, buttons: list[str] | None = None, role_query_works: bool = True) -> None:
            self.wait_calls: list[int] = []
            self.keyboard = TcbsOrdersDialogTest._FakeKeyboard()
            self._buttons = buttons or []
            self.role_query_works = role_query_works
            self.clicked_buttons: list[str] = []
            self.last_selector = ""

        def wait_for_timeout(self, _ms: int) -> None:
            self.wait_calls.append(_ms)
            return None

        def locator(self, selector: str):
            self.last_selector = selector
            selector_upper = selector.upper()
            matched = []
            for button_name in self._buttons:
                button_upper = button_name.upper()
                if "REJECT" in selector_upper and "TỪ CHỐI" in button_upper:
                    matched.append(TcbsOrdersDialogTest._FakeButton(self, button_name))
                elif "TỪ CHỐI" in selector_upper and "TỪ CHỐI" in button_upper:
                    matched.append(TcbsOrdersDialogTest._FakeButton(self, button_name))
            if matched:
                return TcbsOrdersDialogTest._FakeButtonList(matched)
            return TcbsOrdersDialogTest._FakeOverlayLocator()

        def get_by_role(self, role: str, name) -> "TcbsOrdersDialogTest._FakeButtonList":
            assert role == "button"
            if not self.role_query_works:
                raise RuntimeError("role query failed")
            matched = [
                TcbsOrdersDialogTest._FakeButton(self, button_name)
                for button_name in self._buttons
                if name.search(button_name)
            ]
            return TcbsOrdersDialogTest._FakeButtonList(matched)

        def evaluate(self, _script: str) -> bool:
            for button_name in self._buttons:
                if "TỪ CHỐI" in button_name.upper():
                    self.clicked_buttons.append(button_name)
                    return True
            return False

    class _FakeSubmitButton:
        def __init__(self, fail_messages: list[str] | None = None) -> None:
            self._fail_messages = fail_messages or []
            self.click_count = 0

        def is_disabled(self) -> bool:
            return False

        def scroll_into_view_if_needed(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            self.click_count += 1
            if self._fail_messages:
                raise RuntimeError(self._fail_messages.pop(0))

    class _FakeSubmitLocator:
        def __init__(self, button: "TcbsOrdersDialogTest._FakeSubmitButton" | None = None) -> None:
            self._button = button

        def count(self) -> int:
            return 1 if self._button is not None else 0

        @property
        def first(self):
            return self._button

    class _FakeSubmitPage:
        def __init__(self, button: "TcbsOrdersDialogTest._FakeSubmitButton") -> None:
            self._button = button
            self.wait_calls: list[int] = []

        def locator(self, selector: str):
            if selector in {"button.btn.btn-sell", "button.btn-sell", "button.btn.btn-buy", "button.btn-buy"}:
                return TcbsOrdersDialogTest._FakeSubmitLocator(self._button)
            return TcbsOrdersDialogTest._FakeOverlayLocator()

        def get_by_role(self, role: str, name):
            return TcbsOrdersDialogTest._FakeSubmitLocator(None)

        def wait_for_timeout(self, ms: int) -> None:
            self.wait_calls.append(ms)

    def test_normalize_dialog_text_collapses_whitespace(self) -> None:
        raw = "  Giá  không  nằm   trong  khoảng   trần sàn  \n\n Đóng  "
        self.assertEqual(tcbs_orders._normalize_dialog_text(raw), "Giá không nằm trong khoảng trần sàn Đóng")

    def test_dialog_button_regexes_cover_common_labels(self) -> None:
        self.assertIsNotNone(tcbs_orders._DIALOG_CLOSE_RE.search("Đóng"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CLOSE_RE.search("Dong"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CLOSE_RE.search("OK"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CLOSE_RE.search("Hủy"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CLOSE_RE.search("Cancel"))

        self.assertIsNotNone(tcbs_orders._DIALOG_CONFIRM_RE.search("Xác nhận"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CONFIRM_RE.search("Xac nhan"))
        self.assertIsNotNone(tcbs_orders._DIALOG_CONFIRM_RE.search("Confirm"))
        self.assertIsNotNone(tcbs_orders._DIALOG_REJECT_RE.search("TỪ CHỐI"))
        self.assertIsNotNone(tcbs_orders._DIALOG_REJECT_RE.search("Tu choi"))
        self.assertTrue(tcbs_orders._is_known_order_error_dialog("Giá không nằm trong khoảng trần sàn"))
        self.assertTrue(tcbs_orders._is_known_order_error_dialog("Không đủ sức mua"))
        self.assertFalse(tcbs_orders._is_known_order_error_dialog("Thông báo"))

    def test_dismiss_any_open_dialog_rejects_register_notification(self) -> None:
        page = self._FakePage()
        dialog = self._FakeDialog(
            "Bạn chưa đăng ký nhận thông báo từ TCInvest trên thiết bị này. Bạn có muốn nhận thông báo trên thiết bị này không?",
            ["TỪ CHỐI", "ĐỒNG Ý"],
        )
        with patch("scripts.scrapers.tcbs_orders._first_visible_dialog", side_effect=[dialog, None]):
            reason = tcbs_orders._dismiss_any_open_dialog(page, timeout_ms=1000)
        self.assertIn("TỪ CHỐI", dialog.clicked_buttons)
        self.assertIn("đăng ký nhận thông báo", reason)

    def test_dismiss_any_open_dialog_rejects_register_notification_via_selector_fallback(self) -> None:
        page = self._FakePage()
        dialog = self._FakeDialog(
            "Bạn chưa đăng ký nhận thông báo từ TCInvest trên thiết bị này. Bạn có muốn nhận thông báo trên thiết bị này không?",
            ["TỪ CHỐI", "ĐỒNG Ý"],
            role_query_works=False,
        )
        with patch("scripts.scrapers.tcbs_orders._first_visible_dialog", side_effect=[dialog, None]):
            reason = tcbs_orders._dismiss_any_open_dialog(page, timeout_ms=1000)
        self.assertIn("TỪ CHỐI", dialog.clicked_buttons)
        self.assertIn("đăng ký nhận thông báo", reason)

    def test_dismiss_any_open_dialog_rejects_register_notification_via_page_scope_fallback(self) -> None:
        page = self._FakePage(buttons=["TỪ CHỐI", "ĐỒNG Ý"], role_query_works=False)
        dialog = self._FakeDialog(
            "Bạn chưa đăng ký nhận thông báo từ TCInvest trên thiết bị này. Bạn có muốn nhận thông báo trên thiết bị này không?",
            ["TỪ CHỐI", "ĐỒNG Ý"],
            role_query_works=False,
            locator_query_works=False,
        )
        with patch("scripts.scrapers.tcbs_orders._first_visible_dialog", side_effect=[dialog, None]):
            reason = tcbs_orders._dismiss_any_open_dialog(page, timeout_ms=1000)
        self.assertIn("TỪ CHỐI", page.clicked_buttons)
        self.assertIn("đăng ký nhận thông báo", reason)

    def test_chrome_launch_args_disable_notification_prompts(self) -> None:
        args = tcbs_orders._chrome_launch_args()
        self.assertIn("--disable-notifications", args)
        self.assertIn("--deny-permission-prompts", args)

    def test_submit_order_dismisses_overlay_and_retries_after_intercept(self) -> None:
        button = self._FakeSubmitButton(
            fail_messages=[
                "Locator.click: <div class=\"cdk-overlay-backdrop cdk-overlay-backdrop-showing\"></div> intercepts pointer events"
            ]
        )
        page = self._FakeSubmitPage(button)
        with patch("scripts.scrapers.tcbs_orders._dismiss_any_open_dialog") as dismiss_dialog:
            tcbs_orders._submit_order(page, "SELL", timeout_ms=2000)
        self.assertEqual(button.click_count, 2)
        self.assertGreaterEqual(dismiss_dialog.call_count, 2)
        self.assertEqual(page.wait_calls, [300])

    def test_orders_use_dedicated_ephemeral_profile_root(self) -> None:
        root = Path("/tmp/tcbs-orders-test")
        expected = (root / ".playwright" / "tcbs-orders-user-data").resolve()
        self.assertEqual(tcbs_orders._orders_user_data_root(root), expected)

    def test_snapshot_orders_csv_copies_source(self) -> None:
        root = Path("/tmp/tcbs-orders-test")
        source = root / "orders.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("Ticker,Side,Quantity,LimitPrice\nAAA,BUY,100,10.0\n", encoding="utf-8")
        snapshot = tcbs_orders.snapshot_orders_csv(source, root / "snapshots")
        self.assertTrue(snapshot.exists())
        self.assertEqual(snapshot.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))

    @patch("scripts.scrapers.tcbs_orders._wait_for_login_session_ready", return_value="session_ready")
    @patch("scripts.scrapers.tcbs_orders._attempt_login")
    @patch("scripts.scrapers.tcbs_orders._has_login_ui", return_value=True)
    def test_ensure_authenticated_logs_in_when_login_ui_visible(self, has_login_ui, attempt_login, wait_for_login_session_ready) -> None:
        page = object()
        tcbs_orders._ensure_authenticated(page, "user", "pass", 1234, 250)
        has_login_ui.assert_called_with(page)
        attempt_login.assert_called_once_with(page, "user", "pass", 1234, 250)
        wait_for_login_session_ready.assert_called_once_with(page, 1234)

    @patch("scripts.scrapers.tcbs_orders._attempt_login")
    @patch("scripts.scrapers.tcbs_orders._has_login_ui", return_value=False)
    def test_ensure_authenticated_keeps_existing_session_without_logout(self, has_login_ui, attempt_login) -> None:
        page = object()
        with patch("scripts.scrapers.tcbs_orders._logout_if_logged_in") as logout_if_logged_in:
            tcbs_orders._ensure_authenticated(page, "user", "pass", 1234, 250)
        has_login_ui.assert_called_once_with(page)
        attempt_login.assert_not_called()
        logout_if_logged_in.assert_not_called()

    @patch("scripts.scrapers.tcbs_orders._has_order_form_ui", side_effect=[False, True])
    @patch("scripts.scrapers.tcbs_orders._has_login_ui", return_value=False)
    def test_wait_for_order_form_or_login_prefers_order_form_when_it_appears(self, has_login_ui, has_order_form_ui) -> None:
        state = tcbs_orders._wait_for_order_form_or_login(self._FakePage(), 1000)
        self.assertEqual(state, "order_form")
        self.assertEqual(has_order_form_ui.call_count, 2)
        self.assertEqual(has_login_ui.call_count, 1)

    @patch("scripts.scrapers.tcbs_orders._has_order_form_ui", return_value=False)
    @patch("scripts.scrapers.tcbs_orders._has_login_ui", side_effect=[False, True])
    def test_wait_for_order_form_or_login_detects_login_ui(self, has_login_ui, has_order_form_ui) -> None:
        state = tcbs_orders._wait_for_order_form_or_login(self._FakePage(), 1000)
        self.assertEqual(state, "login")
        self.assertEqual(has_order_form_ui.call_count, 2)
        self.assertEqual(has_login_ui.call_count, 2)

    @patch("scripts.scrapers.tcbs_orders._has_order_form_ui", return_value=False)
    @patch("scripts.scrapers.tcbs_orders._has_login_ui", side_effect=[True, False])
    def test_wait_for_login_session_ready_allows_mfa_window(self, has_login_ui, has_order_form_ui) -> None:
        page = self._FakePage()
        state = tcbs_orders._wait_for_login_session_ready(page, 5000, approval_wait_ms=1200)
        self.assertEqual(state, "session_ready")
        self.assertEqual(has_order_form_ui.call_count, 2)
        self.assertEqual(has_login_ui.call_count, 2)
        self.assertEqual(page.wait_calls, [200])

    def test_handle_confirm_or_error_treats_unknown_post_confirm_dialog_as_confirmed(self) -> None:
        order = tcbs_orders.OrderRow(ticker="AAA", side="BUY", quantity=100, price=10.0)
        confirm_dialog = self._FakeDialog("Xác nhận đặt lệnh", ["Xác nhận"])
        post_confirm_dialog = self._FakeDialog("Thông báo", ["Đóng"])
        error_orders = []
        confirmed_orders = []
        page = self._FakePage()

        with (
            patch("scripts.scrapers.tcbs_orders._first_visible_dialog", side_effect=[confirm_dialog, post_confirm_dialog]),
            patch("scripts.scrapers.tcbs_orders.time.time", side_effect=[0, 0, 1, 2, 2, 12, 12]),
            patch("scripts.scrapers.tcbs_orders._POST_CONFIRM_ERROR_GRACE_SEC", 10),
        ):
            status = tcbs_orders._handle_confirm_or_error(
                page,
                order,
                timeout_ms=15000,
                error_orders=error_orders,
                confirmed_orders=confirmed_orders,
            )

        self.assertEqual(status, "confirmed")
        self.assertEqual(error_orders, [])
        self.assertEqual(confirmed_orders, [order])
        self.assertIn("Xác nhận", confirm_dialog.clicked_buttons)
        self.assertIn("Đóng", post_confirm_dialog.clicked_buttons)

    def test_handle_confirm_or_error_keeps_known_post_confirm_error(self) -> None:
        order = tcbs_orders.OrderRow(ticker="AAA", side="BUY", quantity=100, price=10.0)
        confirm_dialog = self._FakeDialog("Xác nhận đặt lệnh", ["Xác nhận"])
        error_dialog = self._FakeDialog("Giá không nằm trong khoảng trần sàn", ["Đóng"])
        error_orders = []
        confirmed_orders = []
        page = self._FakePage()

        with (
            patch("scripts.scrapers.tcbs_orders._first_visible_dialog", side_effect=[confirm_dialog, error_dialog]),
            patch("scripts.scrapers.tcbs_orders.time.time", side_effect=[0, 0, 1, 2, 2, 2, 2, 2]),
            patch("scripts.scrapers.tcbs_orders._POST_CONFIRM_ERROR_GRACE_SEC", 10),
        ):
            status = tcbs_orders._handle_confirm_or_error(
                page,
                order,
                timeout_ms=15000,
                error_orders=error_orders,
                confirmed_orders=confirmed_orders,
            )

        self.assertEqual(status, "error")
        self.assertEqual(error_orders, [(order, "Giá không nằm trong khoảng trần sàn")])
        self.assertEqual(confirmed_orders, [])
        self.assertIn("Xác nhận", confirm_dialog.clicked_buttons)
        self.assertIn("Đóng", error_dialog.clicked_buttons)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
