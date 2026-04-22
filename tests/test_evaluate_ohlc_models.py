from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.evaluate_ohlc_models import build_ticker_ohlc_sample, choose_best_ohlc_model, summarise_ohlc_models


class OhlcReplayAnalysisTest(unittest.TestCase):
    @staticmethod
    def _write_daily_cache(path: Path, closes: list[float], *, volume_start: float) -> None:
        dates = pd.bdate_range("2025-01-06", periods=len(closes))
        frame = pd.DataFrame(
            {
                "date_vn": dates.strftime("%Y-%m-%d"),
                "open": [value - 1.0 for value in closes],
                "high": [value + 2.0 for value in closes],
                "low": [value - 2.0 for value in closes],
                "close": closes,
                "volume": [volume_start + (100.0 * idx) for idx in range(len(closes))],
            }
        )
        frame.to_csv(path, index=False)

    def test_choose_best_ohlc_model_prefers_lower_mean_ohlc_error(self) -> None:
        summary = pd.DataFrame(
            [
                {"Model": "ridge", "MeanOHLCMAEPct": 2.1, "CloseMAEPct": 1.8, "CloseDirHitPct": 52.0},
                {"Model": "random_forest", "MeanOHLCMAEPct": 1.5, "CloseMAEPct": 1.4, "CloseDirHitPct": 55.0},
                {"Model": "hist_gbm", "MeanOHLCMAEPct": 1.6, "CloseMAEPct": 1.3, "CloseDirHitPct": 57.0},
            ]
        )
        self.assertEqual(choose_best_ohlc_model(summary), "random_forest")

    def test_summarise_ohlc_models_aggregates_errors_and_current_bias(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Model": "ridge",
                    "Ticker": "AAA",
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.5,
                    "ActualHighRetPct": 3.0,
                    "PredHighRetPct": 2.5,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -0.5,
                    "ActualCloseRetPct": 2.0,
                    "PredCloseRetPct": 1.0,
                    "ActualRangePct": 4.0,
                    "PredRangePct": 3.0,
                },
                {
                    "Model": "ridge",
                    "Ticker": "BBB",
                    "ActualOpenRetPct": -1.0,
                    "PredOpenRetPct": -0.5,
                    "ActualHighRetPct": 1.0,
                    "PredHighRetPct": 1.5,
                    "ActualLowRetPct": -3.0,
                    "PredLowRetPct": -2.0,
                    "ActualCloseRetPct": -2.0,
                    "PredCloseRetPct": -1.5,
                    "ActualRangePct": 4.0,
                    "PredRangePct": 3.5,
                },
                {
                    "Model": "random_forest",
                    "Ticker": "CCC",
                    "ActualOpenRetPct": 0.5,
                    "PredOpenRetPct": 0.4,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 1.8,
                    "ActualLowRetPct": -0.5,
                    "PredLowRetPct": -0.6,
                    "ActualCloseRetPct": 1.0,
                    "PredCloseRetPct": 1.2,
                    "ActualRangePct": 2.5,
                    "PredRangePct": 2.4,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {"Model": "ridge", "Ticker": "AAA", "PredCloseRetPct": 0.8},
                {"Model": "ridge", "Ticker": "BBB", "PredCloseRetPct": -0.2},
                {"Model": "random_forest", "Ticker": "CCC", "PredCloseRetPct": 1.1},
            ]
        )

        summary = summarise_ohlc_models(history, current).set_index("Model")

        self.assertAlmostEqual(float(summary.loc["ridge", "OpenMAEPct"]), 0.5)
        self.assertAlmostEqual(float(summary.loc["ridge", "CloseDirHitPct"]), 100.0)
        self.assertAlmostEqual(float(summary.loc["random_forest", "CurrentBullishPct"]), 100.0)

    def test_summarise_ohlc_models_keeps_horizon_rows_and_best_model_can_be_selected_per_horizon(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Model": "ridge",
                    "Ticker": "AAA",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.2,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.1,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -0.9,
                    "ActualCloseRetPct": 1.0,
                    "PredCloseRetPct": 1.1,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 3.0,
                },
                {
                    "Model": "random_forest",
                    "Ticker": "AAA",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.0,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.0,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -1.0,
                    "ActualCloseRetPct": 1.0,
                    "PredCloseRetPct": 1.0,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 3.0,
                },
                {
                    "Model": "ridge",
                    "Ticker": "AAA",
                    "Horizon": 2,
                    "ActualOpenRetPct": 2.0,
                    "PredOpenRetPct": 1.0,
                    "ActualHighRetPct": 4.0,
                    "PredHighRetPct": 3.0,
                    "ActualLowRetPct": -2.0,
                    "PredLowRetPct": -1.0,
                    "ActualCloseRetPct": 2.0,
                    "PredCloseRetPct": 1.0,
                    "ActualRangePct": 6.0,
                    "PredRangePct": 4.0,
                },
                {
                    "Model": "random_forest",
                    "Ticker": "AAA",
                    "Horizon": 2,
                    "ActualOpenRetPct": 2.0,
                    "PredOpenRetPct": 2.4,
                    "ActualHighRetPct": 4.0,
                    "PredHighRetPct": 4.5,
                    "ActualLowRetPct": -2.0,
                    "PredLowRetPct": -2.2,
                    "ActualCloseRetPct": 2.0,
                    "PredCloseRetPct": 2.4,
                    "ActualRangePct": 6.0,
                    "PredRangePct": 6.7,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {"Model": "ridge", "Ticker": "AAA", "Horizon": 1, "PredCloseRetPct": 0.5},
                {"Model": "random_forest", "Ticker": "AAA", "Horizon": 1, "PredCloseRetPct": 0.6},
                {"Model": "ridge", "Ticker": "AAA", "Horizon": 2, "PredCloseRetPct": 1.0},
                {"Model": "random_forest", "Ticker": "AAA", "Horizon": 2, "PredCloseRetPct": 1.1},
            ]
        )

        summary = summarise_ohlc_models(history, current)

        self.assertEqual(set(summary["Horizon"].tolist()), {1, 2})
        self.assertEqual(choose_best_ohlc_model(summary, horizon=1), "random_forest")
        self.assertEqual(choose_best_ohlc_model(summary, horizon=2), "random_forest")
        self.assertEqual(choose_best_ohlc_model(summary), "random_forest")

    def test_build_ticker_ohlc_sample_adds_weekly_context_features_without_leaking_future_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            self._write_daily_cache(
                history_dir / "AAA_daily.csv",
                closes=[100, 102, 104, 106, 108, 110, 111, 112, 113, 114, 116, 118, 120, 122, 124],
                volume_start=1000.0,
            )
            self._write_daily_cache(
                history_dir / "VNINDEX_daily.csv",
                closes=[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1011, 1013, 1015, 1017, 1019],
                volume_start=2000.0,
            )

            sample = build_ticker_ohlc_sample("AAA", history_dir, max_horizon=1)

            row = sample.loc[sample["Date"] == pd.Timestamp("2025-01-21")].iloc[0]

            self.assertAlmostEqual(float(row["TickerWeekToDateRetPct"]), 3.5087719298)
            self.assertAlmostEqual(float(row["TickerWeekToDateRangePct"]), 5.2631578947)
            self.assertAlmostEqual(float(row["TickerDistPrevWeekHighPct"]), 1.7241379310)
            self.assertAlmostEqual(float(row["TickerPrevWeekRetPct"]), 5.5555555556)
            self.assertAlmostEqual(float(row["TickerPrevWeekRangePct"]), 7.4074074074)
            self.assertAlmostEqual(float(row["TickerWeekToDateVolumePctPrevWeek"]), 48.2352941176)
            self.assertTrue(pd.isna(row["TickerPrevWeekVolRatio4"]))
            self.assertAlmostEqual(float(row["RelWeekToDateRetPct"]), 3.1123397956)
            self.assertAlmostEqual(float(row["RelPrevWeekRetPct"]), 5.0575475874, places=6)

    def test_build_ticker_ohlc_sample_adds_discrete_state_features_for_hot_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            dates = pd.bdate_range("2025-01-06", periods=90)
            closes = [100.0 + (0.15 * idx) for idx in range(70)] + [111.0, 118.0, 126.0] + [124.0 + (0.1 * idx) for idx in range(17)]
            ticker_frame = pd.DataFrame(
                {
                    "date_vn": dates.strftime("%Y-%m-%d"),
                    "open": [value - 1.0 for value in closes],
                    "high": [value + 2.0 for value in closes],
                    "low": [value - 2.0 for value in closes],
                    "close": closes,
                    "volume": [1000.0 + (10.0 * idx) for idx in range(len(closes))],
                }
            )
            ticker_frame.loc[71, "high"] = ticker_frame.loc[71, "close"] + 8.0
            ticker_frame.loc[71, "low"] = ticker_frame.loc[71, "close"] - 3.0
            ticker_frame.to_csv(history_dir / "AAA_daily.csv", index=False)

            index_closes = [1000.0 + (0.5 * idx) for idx in range(90)]
            index_frame = pd.DataFrame(
                {
                    "date_vn": dates.strftime("%Y-%m-%d"),
                    "open": [value - 1.0 for value in index_closes],
                    "high": [value + 2.0 for value in index_closes],
                    "low": [value - 2.0 for value in index_closes],
                    "close": index_closes,
                    "volume": [2000.0 + (10.0 * idx) for idx in range(len(index_closes))],
                }
            )
            index_frame.to_csv(history_dir / "VNINDEX_daily.csv", index=False)

            sample = build_ticker_ohlc_sample("AAA", history_dir, max_horizon=1)

            shock_row = sample.loc[sample["Date"] == pd.Timestamp(dates[71])].iloc[0]
            impulse_row = sample.loc[sample["Date"] == pd.Timestamp(dates[72])].iloc[0]

            self.assertEqual(float(shock_row["TickerShockState1D"]), 1.0)
            self.assertEqual(float(shock_row["TickerWideRangeState"]), 1.0)
            self.assertEqual(float(impulse_row["TickerImpulseState3D"]), 1.0)
            self.assertEqual(float(impulse_row["TickerTrendRegimeState"]), 1.0)


if __name__ == "__main__":
    unittest.main()
