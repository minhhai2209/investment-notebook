import unittest
from unittest.mock import patch

from scripts.scrapers import tcbs


class TcbsCliTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("scripts.scrapers.tcbs._ensure_logging_configured")
    @patch("scripts.scrapers.tcbs.fetch_tcbs_portfolio", return_value=None)
    def test_main_login_only_passes_post_login_wait(
        self,
        fetch_tcbs_portfolio,
        _ensure_logging_configured,
        print_mock,
    ) -> None:
        exit_code = tcbs.main(["--headful", "--login-only", "--post-login-wait-ms", "30000"])
        self.assertEqual(exit_code, 0)
        fetch_tcbs_portfolio.assert_called_once_with(
            headless=False,
            timeout_ms=300000,
            slow_mo_ms=None,
            post_login_wait_ms=30000,
            login_only=True,
        )
        print_mock.assert_not_called()

    @patch("scripts.scrapers.tcbs._ensure_logging_configured")
    @patch("scripts.scrapers.tcbs.fetch_tcbs_portfolio", return_value="data/portfolios/portfolio.csv")
    def test_main_full_flow_also_passes_post_login_wait(
        self,
        fetch_tcbs_portfolio,
        _ensure_logging_configured,
    ) -> None:
        exit_code = tcbs.main(["--headful", "--post-login-wait-ms", "30000"])
        self.assertEqual(exit_code, 0)
        fetch_tcbs_portfolio.assert_called_once_with(
            headless=False,
            timeout_ms=300000,
            slow_mo_ms=None,
            post_login_wait_ms=30000,
            login_only=False,
        )

    @patch("scripts.scrapers.tcbs.LOGGER.warning")
    def test_default_post_login_wait_ms_falls_back_when_env_invalid(self, warning_mock) -> None:
        with patch.dict("os.environ", {"TCBS_POST_LOGIN_WAIT_MS": "invalid"}, clear=False):
            self.assertEqual(tcbs._default_post_login_wait_ms(), 20000)
        warning_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
