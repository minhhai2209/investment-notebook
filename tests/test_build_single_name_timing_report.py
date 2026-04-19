from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.build_single_name_timing_report import (
    _compute_trade_efficiency_targets,
    _validate_output_coverage,
    select_best_single_name_configs,
)


class BuildSingleNameTimingReportTest(unittest.TestCase):
    def test_compute_trade_efficiency_targets_tracks_peak_day_drawdown_and_close(self) -> None:
        index = pd.date_range("2025-01-01", periods=5, freq="D")
        close = pd.Series([100, 100, 100, 100, 100], index=index)
        high = pd.Series([101, 105, 103, 102, 101], index=index)
        low = pd.Series([99, 97, 96, 98, 99], index=index)

        targets = _compute_trade_efficiency_targets(close, high, low, horizon_days=3)

        first = targets.iloc[0]
        self.assertAlmostEqual(float(first["TargetPeakRetPct"]), 5.0)
        self.assertAlmostEqual(float(first["TargetPeakDay"]), 1.0)
        self.assertAlmostEqual(float(first["TargetDrawdownPct"]), -4.0)
        self.assertAlmostEqual(float(first["TargetCloseRetPct"]), 0.0)

    def test_select_best_single_name_configs_keeps_lowest_selection_score_per_ticker_horizon(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Horizon": 5,
                    "Variant": "full_2y",
                    "Model": "ridge",
                    "SelectionScore": 2.0,
                    "TradeScoreMAEPct": 1.0,
                    "PeakRetMAEPct": 1.0,
                    "DrawdownMAEPct": 1.0,
                    "CloseMAEPct": 1.0,
                    "TradeScoreHitPct": 55.0,
                },
                {
                    "Ticker": "AAA",
                    "Horizon": 5,
                    "Variant": "recent_focus",
                    "Model": "hist_gbm",
                    "SelectionScore": 1.6,
                    "TradeScoreMAEPct": 0.9,
                    "PeakRetMAEPct": 0.8,
                    "DrawdownMAEPct": 1.0,
                    "CloseMAEPct": 0.9,
                    "TradeScoreHitPct": 57.0,
                },
                {
                    "Ticker": "AAA",
                    "Horizon": 10,
                    "Variant": "full_2y",
                    "Model": "ridge",
                    "SelectionScore": 1.2,
                    "TradeScoreMAEPct": 0.7,
                    "PeakRetMAEPct": 0.8,
                    "DrawdownMAEPct": 0.9,
                    "CloseMAEPct": 0.8,
                    "TradeScoreHitPct": 58.0,
                },
                {
                    "Ticker": "AAA",
                    "Horizon": 5,
                    "Variant": "quarter_focus",
                    "Model": "ridge",
                    "SelectionScore": 1.4,
                    "TradeScoreMAEPct": 0.8,
                    "PeakRetMAEPct": 0.7,
                    "DrawdownMAEPct": 0.9,
                    "CloseMAEPct": 0.8,
                    "TradeScoreHitPct": 58.0,
                },
            ]
        )

        selected = select_best_single_name_configs(metrics)

        self.assertEqual(int(selected.shape[0]), 2)
        row_5 = selected[selected["Horizon"] == 5].iloc[0]
        self.assertEqual(str(row_5["Variant"]), "quarter_focus")
        self.assertEqual(str(row_5["Model"]), "ridge")

    def test_validate_output_coverage_rejects_missing_ticker_horizon_pairs(self) -> None:
        report = pd.DataFrame(
            {
                "Ticker": ["AAA", "AAA", "BBB"],
                "Horizon": [3, 5, 3],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "coverage incomplete"):
            _validate_output_coverage(report, ["AAA", "BBB"], [3, 5])


if __name__ == "__main__":
    unittest.main()
