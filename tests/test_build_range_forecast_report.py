from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_range_forecast_report import (
    _build_variant_prediction_file,
    _recent_focus_weight,
    _select_best_range_model,
    _validate_prediction_file_coverage,
    _variant_train_slice,
    resolve_report_tickers,
)


class BuildRangeForecastReportTest(unittest.TestCase):
    def test_variant_train_slice_limits_recent_focus_window(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=10, freq="D"),
                "Ticker": ["AAA"] * 10,
            }
        )

        recent = _variant_train_slice(frame, "recent_focus", recent_focus_dates=4)

        self.assertEqual(int(recent.shape[0]), 4)
        self.assertEqual(str(recent["Date"].min().date()), "2025-01-07")
        self.assertEqual(str(recent["Date"].max().date()), "2025-01-10")

    def test_variant_train_slice_limits_quarter_focus_window(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=10, freq="D"),
                "Ticker": ["AAA"] * 10,
            }
        )

        quarter = _variant_train_slice(frame, "quarter_focus", recent_focus_dates=6, quarter_focus_dates=3)

        self.assertEqual(int(quarter.shape[0]), 3)
        self.assertEqual(str(quarter["Date"].min().date()), "2025-01-08")
        self.assertEqual(str(quarter["Date"].max().date()), "2025-01-10")

    def test_recent_focus_weight_decreases_with_horizon(self) -> None:
        self.assertAlmostEqual(_recent_focus_weight(1), 0.75)
        self.assertAlmostEqual(_recent_focus_weight(5), 0.5944444444)
        self.assertAlmostEqual(_recent_focus_weight(10), 0.40)

    def test_resolve_report_tickers_uses_full_universe_when_universe_csv_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            universe_csv = Path(tmp) / "universe.csv"
            pd.DataFrame(
                {
                    "Ticker": ["VNINDEX", "AAA", "BBB", "CCC"],
                    "PositionQuantity": [0, 100, 0, 200],
                    "RelStrength20Rank": [0, 90, 80, 10],
                    "RelStrength60Rank": [0, 85, 70, 20],
                    "Ret20dVsIndex": [0, 5.0, 3.0, -7.0],
                    "Ret60dVsIndex": [0, 4.0, 2.0, -5.0],
                    "SectorBreadthAboveSMA20Pct": [0, 80, 65, 15],
                    "ADTV20PctRank": [0, 90, 70, 85],
                    "DistSMA20Pct": [0, 1.0, -2.0, -9.0],
                }
            ).to_csv(universe_csv, index=False)

            tickers = resolve_report_tickers(
                tickers=None,
                universe_csv=universe_csv,
                max_report_tickers=2,
                dynamic_buy_tickers=2,
                dynamic_sell_tickers=2,
            )

            self.assertEqual(tickers, ["VNINDEX", "AAA", "BBB", "CCC"])
            self.assertIn("AAA", tickers)
            self.assertIn("BBB", tickers)
            self.assertIn("CCC", tickers)
            self.assertEqual(tickers[0], "VNINDEX")

    def test_validate_prediction_file_coverage_rejects_missing_pairs(self) -> None:
        prediction_view = pd.DataFrame(
            {
                "Ticker": ["VNINDEX", "VNINDEX", "AAA", "AAA", "AAA"],
                "Horizon": [1, 5, 1, 5, 10],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "coverage incomplete"):
            _validate_prediction_file_coverage(prediction_view, ["VNINDEX", "AAA"], "full_2y")

    def test_build_variant_prediction_file_uses_single_variant_schema(self) -> None:
        easy_view = pd.DataFrame(
            {
                "SnapshotDate": ["2026-03-20", "2026-03-20"],
                "Variant": ["full_2y", "recent_focus"],
                "Ticker": ["AAA", "AAA"],
                "Horizon": [1, 1],
                "ForecastWindow": ["T+1", "T+1"],
                "Base": [100.0, 100.0],
                "Low": [99.0, 98.0],
                "Mid": [101.0, 100.0],
                "High": [102.0, 103.0],
                "PredLowRetPct": [-1.0, -2.0],
                "PredMidRetPct": [1.0, 0.0],
                "PredHighRetPct": [2.0, 3.0],
                "CloseMAEPct": [2.0, 2.5],
                "RangeMAEPct": [1.0, 1.2],
                "CloseDirHitPct": [52.0, 50.0],
            }
        )

        prediction_view = _build_variant_prediction_file(easy_view, "recent_focus")

        self.assertEqual(int(prediction_view.shape[0]), 1)
        row = prediction_view.iloc[0]
        self.assertAlmostEqual(float(row["RecentFocusWeight"]), 0.75)
        self.assertAlmostEqual(float(row["Mid"]), 100.0)
        self.assertAlmostEqual(float(row["PredMidRetPct"]), 0.0)
        self.assertEqual(str(row["ForecastWindow"]), "T+1")

    def test_select_best_range_model_picks_lowest_selection_score_per_ticker_variant_horizon(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Variant": "full_2y",
                    "Model": "ridge",
                    "Horizon": 5,
                    "SelectionScore": 1.5,
                    "CloseMAEPct": 1.0,
                    "RangeMAEPct": 1.2,
                    "CloseDirHitPct": 55.0,
                },
                {
                    "Ticker": "AAA",
                    "Variant": "full_2y",
                    "Model": "hist_gbm",
                    "Horizon": 5,
                    "SelectionScore": 1.1,
                    "CloseMAEPct": 0.9,
                    "RangeMAEPct": 1.0,
                    "CloseDirHitPct": 57.0,
                },
                {
                    "Ticker": "AAA",
                    "Variant": "recent_focus",
                    "Model": "random_forest",
                    "Horizon": 5,
                    "SelectionScore": 0.8,
                    "CloseMAEPct": 0.7,
                    "RangeMAEPct": 0.9,
                    "CloseDirHitPct": 60.0,
                },
            ]
        )

        selected = _select_best_range_model(metrics)

        self.assertEqual(int(selected.shape[0]), 2)
        full_row = selected[selected["Variant"] == "full_2y"].iloc[0]
        self.assertEqual(str(full_row["Model"]), "hist_gbm")


if __name__ == "__main__":
    unittest.main()
