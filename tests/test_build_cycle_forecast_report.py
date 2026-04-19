from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_cycle_forecast_report import (
    _compute_cycle_targets,
    _load_tickers_from_universe_csv,
    build_best_horizon_summary,
    build_selected_cycle_matrix,
    select_best_cycle_configs,
)


class BuildCycleForecastReportTest(unittest.TestCase):
    def test_compute_cycle_targets_respects_sell_delay_for_peak_but_not_drawdown(self) -> None:
        index = pd.date_range("2025-01-01", periods=7, freq="D")
        close = pd.Series([100, 101, 102, 103, 104, 105, 106], index=index)
        high = pd.Series([101, 103, 120, 110, 130, 108, 107], index=index)
        low = pd.Series([99, 90, 95, 96, 97, 98, 99], index=index)

        targets = _compute_cycle_targets(close, high, low, horizon_days=4, sell_delay_days=3)

        first = targets.iloc[0]
        self.assertAlmostEqual(float(first["TargetPeakRetPct"]), 30.0)
        self.assertAlmostEqual(float(first["TargetPeakDay"]), 4.0)
        self.assertAlmostEqual(float(first["TargetDrawdownPct"]), -10.0)

    def test_select_best_cycle_configs_prefers_lower_peak_error_then_day_error(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "Variant": "full_2y",
                    "Model": "ridge",
                    "PeakRetMAEPct": 4.0,
                    "PeakDayMAE": 3.0,
                    "DrawdownMAEPct": 2.0,
                },
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "Variant": "recent_focus",
                    "Model": "hist_gbm",
                    "PeakRetMAEPct": 3.0,
                    "PeakDayMAE": 5.0,
                    "DrawdownMAEPct": 2.0,
                },
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "Variant": "full_2y",
                    "Model": "random_forest",
                    "PeakRetMAEPct": 3.0,
                    "PeakDayMAE": 4.0,
                    "DrawdownMAEPct": 3.0,
                },
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "Variant": "quarter_focus",
                    "Model": "hist_gbm",
                    "PeakRetMAEPct": 2.8,
                    "PeakDayMAE": 3.8,
                    "DrawdownMAEPct": 1.9,
                },
            ]
        )

        best = select_best_cycle_configs(metrics)

        self.assertEqual(int(best.shape[0]), 1)
        row = best.iloc[0]
        self.assertEqual(str(row["Model"]), "hist_gbm")
        self.assertEqual(str(row["Variant"]), "quarter_focus")

    def test_build_selected_cycle_matrix_pivots_one_row_per_ticker(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "Variant": "full_2y",
                    "Model": "ridge",
                    "PredPeakRetPct": 6.0,
                    "PredPeakDays": 8.0,
                    "PredPeakPrice": 106.0,
                    "PredDrawdownPct": -4.0,
                    "PredDrawdownPrice": 96.0,
                    "PeakRetMAEPct": 2.0,
                    "PeakDayMAE": 3.0,
                    "DrawdownMAEPct": 1.5,
                },
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 2,
                    "Variant": "recent_focus",
                    "Model": "hist_gbm",
                    "PredPeakRetPct": 10.0,
                    "PredPeakDays": 24.0,
                    "PredPeakPrice": 110.0,
                    "PredDrawdownPct": -7.0,
                    "PredDrawdownPrice": 93.0,
                    "PeakRetMAEPct": 3.0,
                    "PeakDayMAE": 5.0,
                    "DrawdownMAEPct": 2.5,
                },
            ]
        )

        matrix = build_selected_cycle_matrix(selected)

        self.assertEqual(int(matrix.shape[0]), 1)
        self.assertIn("PredPeakRetPct_1M", matrix.columns)
        self.assertIn("PredPeakRetPct_2M", matrix.columns)
        self.assertAlmostEqual(float(matrix.iloc[0]["PredPeakDays_2M"]), 24.0)

    def test_build_best_horizon_summary_keeps_lowest_selection_score_per_ticker(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 1,
                    "SelectionScore": 8.0,
                    "PeakRetMAEPct": 4.0,
                    "PeakDayMAE": 5.0,
                    "DrawdownMAEPct": 3.0,
                },
                {
                    "Ticker": "AAA",
                    "HorizonMonths": 2,
                    "SelectionScore": 6.0,
                    "PeakRetMAEPct": 5.0,
                    "PeakDayMAE": 6.0,
                    "DrawdownMAEPct": 2.0,
                },
            ]
        )

        summary = build_best_horizon_summary(selected)

        self.assertEqual(int(summary.shape[0]), 1)
        self.assertEqual(int(summary.iloc[0]["HorizonMonths"]), 2)

    def test_load_tickers_from_universe_csv_reads_ticker_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "universe.csv"
            pd.DataFrame({"Ticker": ["VNINDEX", "REE", "MBB", "REE"]}).to_csv(path, index=False)

            tickers = _load_tickers_from_universe_csv(path)

            self.assertEqual(tickers, ["VNINDEX", "REE", "MBB"])


if __name__ == "__main__":
    unittest.main()
