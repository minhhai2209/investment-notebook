from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tools import refresh_industry_map


class RefreshIndustryMapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_refresh_from_vietstock_profiles_csv_uses_top_level_by_default(self) -> None:
        source = self.base / "tickers.csv"
        source.write_text("Ticker\nDPM\nDGC\n", encoding="utf-8")
        output = self.base / "industry_map.csv"

        levels = {
            "DPM": ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"],
            "DGC": ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"],
        }

        with mock.patch.object(
            refresh_industry_map,
            "fetch_vietstock_sector_levels",
            side_effect=lambda ticker, **_: levels[ticker],
        ):
            exit_code = refresh_industry_map.main(
                [
                    "--from-vietstock-profiles-csv",
                    str(source),
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows,
            [
                {"Ticker": "DGC", "Sector": "Nguyên vật liệu"},
                {"Ticker": "DPM", "Sector": "Nguyên vật liệu"},
            ],
        )

    def test_refresh_from_vietstock_profiles_hose_can_select_leaf_sector(self) -> None:
        output = self.base / "industry_map.csv"
        levels = {
            "DPM": ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"],
            "HPG": ["Nguyên vật liệu", "Nguyên vật liệu", "Kim loại và khai khoáng"],
        }

        with mock.patch.object(refresh_industry_map, "fetch_hose_members", return_value={"HPG", "DPM"}), mock.patch.object(
            refresh_industry_map,
            "fetch_vietstock_sector_levels",
            side_effect=lambda ticker, **_: levels[ticker],
        ):
            exit_code = refresh_industry_map.main(
                [
                    "--from-vietstock-profiles-hose",
                    "--sector-level",
                    "leaf",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows,
            [
                {"Ticker": "DPM", "Sector": "Hóa chất"},
                {"Ticker": "HPG", "Sector": "Kim loại và khai khoáng"},
            ],
        )

    def test_refresh_from_vietstock_profiles_vn30_uses_live_vn30_members(self) -> None:
        output = self.base / "industry_map.csv"
        levels = {
            "FPT": ["Công nghệ thông tin", "Phần mềm", "Phần mềm"],
            "MBB": ["Tài chính", "Ngân hàng", "Ngân hàng thương mại"],
        }

        with mock.patch.object(refresh_industry_map, "fetch_vn30_members", return_value={"MBB", "FPT"}), mock.patch.object(
            refresh_industry_map,
            "fetch_vietstock_sector_levels",
            side_effect=lambda ticker, **_: levels[ticker],
        ):
            exit_code = refresh_industry_map.main(
                [
                    "--from-vietstock-profiles-vn30",
                    "--sector-level",
                    "top",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows,
            [
                {"Ticker": "FPT", "Sector": "Công nghệ thông tin"},
                {"Ticker": "MBB", "Sector": "Tài chính"},
            ],
        )

    def test_refresh_from_vietstock_profiles_skips_cached_rows_by_default(self) -> None:
        source = self.base / "tickers.csv"
        source.write_text("Ticker\nDPM\nDGC\n", encoding="utf-8")
        output = self.base / "industry_map.csv"
        output.write_text("Ticker,Sector\nDPM,Nguyên vật liệu\n", encoding="utf-8")

        fetched: list[str] = []
        levels = {"DGC": ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"]}

        with mock.patch.object(
            refresh_industry_map,
            "fetch_vietstock_sector_levels",
            side_effect=lambda ticker, **_: fetched.append(ticker) or levels[ticker],
        ):
            exit_code = refresh_industry_map.main(
                [
                    "--from-vietstock-profiles-csv",
                    str(source),
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(fetched, ["DGC"])
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows,
            [
                {"Ticker": "DGC", "Sector": "Nguyên vật liệu"},
                {"Ticker": "DPM", "Sector": "Nguyên vật liệu"},
            ],
        )

    def test_refresh_from_live_vn100_portfolio_profiles_merges_portfolio_and_extra_tickers(self) -> None:
        portfolio = self.base / "portfolio.csv"
        portfolio.write_text("Ticker,Quantity,AvgPrice\nHPG,100,25\nNVL,200,12\n", encoding="utf-8")
        output = self.base / "industry_map.csv"
        output.write_text("Ticker,Sector\nDPM,Nguyên vật liệu\n", encoding="utf-8")

        fetched: list[str] = []
        levels = {
            "HPG": ["Nguyên vật liệu", "Nguyên vật liệu", "Kim loại và khai khoáng"],
            "NVL": ["Bất động sản", "Bất động sản", "Phát triển bất động sản"],
        }

        with mock.patch.object(refresh_industry_map, "_fetch_investing_vn100_members", return_value={"DPM"}), mock.patch.object(
            refresh_industry_map,
            "fetch_vietstock_sector_levels",
            side_effect=lambda ticker, **_: fetched.append(ticker) or levels[ticker],
        ):
            exit_code = refresh_industry_map.main(
                [
                    "--from-live-vn100-portfolio-profiles",
                    "--expect-count",
                    "1",
                    "--portfolio-csv",
                    str(portfolio),
                    "--extra-ticker",
                    "NVL",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(sorted(fetched), ["HPG", "NVL"])
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows,
            [
                {"Ticker": "DPM", "Sector": "Nguyên vật liệu"},
                {"Ticker": "HPG", "Sector": "Nguyên vật liệu"},
                {"Ticker": "NVL", "Sector": "Bất động sản"},
            ],
        )

    def test_script_entrypoint_help_works_when_run_directly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        python_bin = repo_root / "venv" / "bin" / "python"
        result = subprocess.run(
            [str(python_bin), "scripts/tools/refresh_industry_map.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Refresh data/industry_map.csv", result.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
