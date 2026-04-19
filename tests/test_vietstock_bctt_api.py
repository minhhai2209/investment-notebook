from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_fetching.vietstock_bctt_api import (
    VietstockBcttCache,
    build_daily_bctt_feature_frame,
    build_quarterly_feature_frame_from_tables,
)


def _sample_tables() -> dict:
    periods = ["Q4/2024", "Q1/2025", "Q2/2025", "Q3/2025", "Q4/2025"]
    date_ranges = [
        "01/10/2024-31/12/2024",
        "01/01/2025-31/03/2025",
        "01/04/2025-30/06/2025",
        "01/07/2025-30/09/2025",
        "01/10/2025-31/12/2025",
    ]
    return {
        "income_statement": {
            "period_labels": periods,
            "date_ranges": date_ranges,
            "rows": [
                {
                    "label": "Doanh thu thuần về bán hàng và cung cấp dịch vụ",
                    "values": ["100", "110", "120", "130", "150"],
                },
                {
                    "label": "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ",
                    "values": ["20", "24", "26", "28", "40"],
                },
                {
                    "label": "Lợi nhuận sau thuế thu nhập doanh nghiệp",
                    "values": ["10", "12", "14", "16", "25"],
                },
            ],
        },
        "balance_sheet": {
            "period_labels": periods,
            "date_ranges": date_ranges,
            "rows": [
                {
                    "label": "Tiền và các khoản tương đương tiền",
                    "values": ["30", "28", "32", "35", "45"],
                },
                {
                    "label": "Các khoản phải thu ngắn hạn",
                    "values": ["18", "19", "20", "21", "24"],
                },
                {
                    "label": "Hàng tồn kho",
                    "values": ["25", "27", "29", "31", "40"],
                },
                {
                    "label": "Nợ ngắn hạn",
                    "values": ["15", "16", "18", "20", "22"],
                },
                {
                    "label": "Nợ phải trả",
                    "values": ["45", "47", "50", "52", "60"],
                },
                {
                    "label": "Vốn chủ sở hữu",
                    "values": ["55", "58", "61", "64", "70"],
                },
                {
                    "label": "Tổng cộng tài sản",
                    "values": ["100", "105", "111", "116", "130"],
                },
            ],
        },
        "financial_ratios": {
            "period_labels": periods,
            "date_ranges": date_ranges,
            "rows": [
                {"label": "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)", "values": ["4", "4.2", "4.4", "4.6", "5.1"]},
                {"label": "Giá trị sổ sách của cổ phiếu (BVPS)", "values": ["10", "10.2", "10.4", "10.7", "11.0"]},
                {"label": "Chỉ số giá thị trường trên giá trị sổ sách (P/B)", "values": ["1.2", "1.3", "1.35", "1.4", "1.5"]},
                {"label": "Tỷ suất lợi nhuận gộp biên", "values": ["20", "21.8", "21.7", "21.5", "26.7"]},
                {"label": "Tỷ suất sinh lợi trên doanh thu thuần", "values": ["10", "10.9", "11.7", "12.3", "16.7"]},
                {"label": "Tỷ suất lợi nhuận trên vốn chủ sở hữu bình quân (ROEA)", "values": ["18", "18.5", "19", "19.5", "22"]},
                {"label": "Tỷ suất sinh lợi trên tổng tài sản bình quân (ROAA)", "values": ["9", "9.3", "9.7", "10", "12"]},
                {"label": "Tỷ số thanh toán hiện hành (ngắn hạn)", "values": ["1.8", "1.75", "1.72", "1.7", "1.85"]},
                {"label": "Khả năng thanh toán lãi vay", "values": ["4", "4.1", "4.2", "4.4", "5.0"]},
                {"label": "Tỷ số Nợ trên Tổng tài sản", "values": ["45", "44.8", "45.0", "44.8", "46.2"]},
                {"label": "Tỷ số Nợ vay trên Vốn chủ sở hữu", "values": ["60", "58", "57", "56", "55"]},
            ],
        },
    }


class VietstockBcttApiTest(unittest.TestCase):
    def test_build_quarterly_feature_frame_derives_growth_and_lag(self) -> None:
        frame = build_quarterly_feature_frame_from_tables("HPG", _sample_tables())
        latest = frame.iloc[-1]

        self.assertEqual(str(latest["PeriodEnd"].date()), "2025-12-31")
        self.assertEqual(str(latest["AvailableDate"].date()), "2026-03-16")
        self.assertAlmostEqual(float(latest["BCTT_RevenueQoQPct"]), (150 / 130 - 1) * 100, places=6)
        self.assertAlmostEqual(float(latest["BCTT_RevenueYoYPct"]), 50.0, places=6)
        self.assertAlmostEqual(float(latest["BCTT_CashToShortLiability"]), 45 / 22, places=6)

    def test_build_daily_feature_frame_uses_last_available_quarter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VietstockBcttCache(Path(tmpdir), max_age_hours=0)
            cache.store("HPG", _sample_tables())

            sample_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-02-15", "2026-03-20"]),
                    "Ticker": ["HPG", "HPG"],
                }
            )
            daily = build_daily_bctt_feature_frame(
                sample_df=sample_df,
                cache_dir=Path(tmpdir),
                feature_set="hybrid_growth",
                max_age_hours=0,
                headless=True,
            ).sort_values("Date")

            before_release = daily.iloc[0]
            after_release = daily.iloc[1]
            self.assertAlmostEqual(float(before_release["BCTT_PB"]), 1.4, places=6)
            self.assertAlmostEqual(float(after_release["BCTT_PB"]), 1.5, places=6)
            self.assertAlmostEqual(float(after_release["BCTT_RevenueYoYPct"]), 50.0, places=6)


if __name__ == "__main__":
    unittest.main()
