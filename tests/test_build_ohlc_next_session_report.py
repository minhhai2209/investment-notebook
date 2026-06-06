from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.build_ohlc_next_session_report import (
    select_best_multi_session_forecasts,
    select_best_next_session_forecasts,
    summarise_ohlc_model_metrics_by_ticker,
)


class BuildOhlcNextSessionReportTest(unittest.TestCase):
    def test_summarise_ohlc_model_metrics_by_ticker_scores_each_ticker_model_pair(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Model": "ridge",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.2,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.2,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -0.8,
                    "ActualCloseRetPct": 1.5,
                    "PredCloseRetPct": 1.1,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 3.0,
                },
                {
                    "Ticker": "AAA",
                    "Model": "random_forest",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.0,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.0,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -1.0,
                    "ActualCloseRetPct": 1.5,
                    "PredCloseRetPct": 1.4,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 3.0,
                },
            ]
        )

        summary = summarise_ohlc_model_metrics_by_ticker(history, horizon=1)

        self.assertEqual(summary.iloc[0]["Ticker"], "AAA")
        self.assertEqual(summary.iloc[0]["Model"], "random_forest")
        self.assertAlmostEqual(float(summary.iloc[0]["CloseMAEPct"]), 0.1)

    def test_summarise_ohlc_model_metrics_by_ticker_penalises_underpredicted_highs(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Model": "ridge",
                    "Horizon": 1,
                    "ActualOpenRetPct": 0.2,
                    "PredOpenRetPct": 0.2,
                    "ActualHighRetPct": 5.0,
                    "PredHighRetPct": 1.0,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -1.0,
                    "ActualCloseRetPct": 0.5,
                    "PredCloseRetPct": 0.6,
                    "ActualRangePct": 6.0,
                    "PredRangePct": 2.0,
                },
                {
                    "Ticker": "AAA",
                    "Model": "hist_gbm",
                    "Horizon": 1,
                    "ActualOpenRetPct": 0.2,
                    "PredOpenRetPct": 0.2,
                    "ActualHighRetPct": 5.0,
                    "PredHighRetPct": 4.7,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -0.9,
                    "ActualCloseRetPct": 0.5,
                    "PredCloseRetPct": 0.8,
                    "ActualRangePct": 6.0,
                    "PredRangePct": 5.6,
                },
            ]
        )

        summary = summarise_ohlc_model_metrics_by_ticker(history, horizon=1)

        self.assertEqual(str(summary.iloc[0]["Model"]), "hist_gbm")
        self.assertGreater(float(summary.loc[summary["Model"] == "ridge", "UpsideMissMAEPct"].iloc[0]), 3.0)

    def test_select_best_next_session_forecasts_keeps_only_the_best_model_per_ticker(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Model": "ridge",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.5,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.4,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -0.4,
                    "ActualCloseRetPct": 1.0,
                    "PredCloseRetPct": 0.2,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.8,
                },
                {
                    "Ticker": "AAA",
                    "Model": "random_forest",
                    "Horizon": 1,
                    "ActualOpenRetPct": 1.0,
                    "PredOpenRetPct": 1.0,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 2.0,
                    "ActualLowRetPct": -1.0,
                    "PredLowRetPct": -1.0,
                    "ActualCloseRetPct": 1.0,
                    "PredCloseRetPct": 0.9,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 3.0,
                },
                {
                    "Ticker": "BBB",
                    "Model": "ridge",
                    "Horizon": 1,
                    "ActualOpenRetPct": -0.5,
                    "PredOpenRetPct": -0.4,
                    "ActualHighRetPct": 1.0,
                    "PredHighRetPct": 1.2,
                    "ActualLowRetPct": -1.5,
                    "PredLowRetPct": -1.1,
                    "ActualCloseRetPct": -0.8,
                    "PredCloseRetPct": -0.7,
                    "ActualRangePct": 2.5,
                    "PredRangePct": 2.3,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    "Date": "2026-03-30",
                    "Ticker": "AAA",
                    "Horizon": 1,
                    "ForecastWindow": "T+1",
                    "ForecastDate": "2026-03-31",
                    "BaseClose": 10.0,
                    "Model": "ridge",
                    "PredOpen": 10.1,
                    "PredHigh": 10.4,
                    "PredLow": 9.8,
                    "PredClose": 10.2,
                    "PredCloseRetPct": 2.0,
                    "PredRangePct": 6.0,
                    "ForecastCandleBias": "BULLISH",
                    "TickerShockState1D": 1.0,
                    "TickerImpulseState3D": 1.0,
                    "TickerWideRangeState": 0.0,
                    "TickerTrendRegimeState": 1.0,
                },
                {
                    "Date": "2026-03-30",
                    "Ticker": "AAA",
                    "Horizon": 1,
                    "ForecastWindow": "T+1",
                    "ForecastDate": "2026-03-31",
                    "BaseClose": 10.0,
                    "Model": "random_forest",
                    "PredOpen": 10.0,
                    "PredHigh": 10.3,
                    "PredLow": 9.9,
                    "PredClose": 10.1,
                    "PredCloseRetPct": 1.0,
                    "PredRangePct": 4.0,
                    "ForecastCandleBias": "BULLISH",
                    "TickerShockState1D": 1.0,
                    "TickerImpulseState3D": 1.0,
                    "TickerWideRangeState": 0.0,
                    "TickerTrendRegimeState": 1.0,
                },
                {
                    "Date": "2026-03-30",
                    "Ticker": "BBB",
                    "Horizon": 1,
                    "ForecastWindow": "T+1",
                    "ForecastDate": "2026-03-31",
                    "BaseClose": 20.0,
                    "Model": "ridge",
                    "PredOpen": 19.9,
                    "PredHigh": 20.2,
                    "PredLow": 19.6,
                    "PredClose": 19.8,
                    "PredCloseRetPct": -1.0,
                    "PredRangePct": 3.0,
                    "ForecastCandleBias": "BEARISH",
                    "TickerShockState1D": -1.0,
                    "TickerImpulseState3D": -1.0,
                    "TickerWideRangeState": 0.0,
                    "TickerTrendRegimeState": -1.0,
                },
            ]
        )

        report = select_best_next_session_forecasts(history, current, horizon=1)

        self.assertEqual(report["Ticker"].tolist(), ["AAA", "BBB"])
        aaa_row = report.loc[report["Ticker"] == "AAA"].iloc[0]
        self.assertEqual(str(aaa_row["Model"]), "random_forest")
        self.assertAlmostEqual(float(aaa_row["ForecastClose"]), 10.1)
        self.assertAlmostEqual(float(aaa_row["TickerShockState1D"]), 1.0)
        self.assertAlmostEqual(float(aaa_row["TickerTrendRegimeState"]), 1.0)

    def test_select_best_multi_session_forecasts_exports_each_requested_horizon(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Model": "ridge",
                    "Horizon": 1,
                    "ActualOpenRetPct": 0.2,
                    "PredOpenRetPct": 0.1,
                    "ActualHighRetPct": 1.0,
                    "PredHighRetPct": 0.9,
                    "ActualLowRetPct": -0.5,
                    "PredLowRetPct": -0.4,
                    "ActualCloseRetPct": 0.8,
                    "PredCloseRetPct": 0.7,
                    "ActualRangePct": 1.5,
                    "PredRangePct": 1.3,
                },
                {
                    "Ticker": "AAA",
                    "Model": "ridge",
                    "Horizon": 2,
                    "ActualOpenRetPct": 0.4,
                    "PredOpenRetPct": 0.3,
                    "ActualHighRetPct": 2.0,
                    "PredHighRetPct": 1.8,
                    "ActualLowRetPct": -0.7,
                    "PredLowRetPct": -0.6,
                    "ActualCloseRetPct": 1.4,
                    "PredCloseRetPct": 1.2,
                    "ActualRangePct": 2.7,
                    "PredRangePct": 2.4,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    "Date": "2026-03-30",
                    "Ticker": "AAA",
                    "Horizon": 1,
                    "ForecastWindow": "T+1",
                    "ForecastDate": "2026-03-31",
                    "BaseClose": 10.0,
                    "Model": "ridge",
                    "PredOpen": 10.0,
                    "PredHigh": 10.2,
                    "PredLow": 9.9,
                    "PredClose": 10.1,
                    "PredCloseRetPct": 1.0,
                    "PredRangePct": 3.0,
                    "ForecastCandleBias": "BULLISH",
                },
                {
                    "Date": "2026-03-30",
                    "Ticker": "AAA",
                    "Horizon": 2,
                    "ForecastWindow": "T+2",
                    "ForecastDate": "2026-04-01",
                    "BaseClose": 10.0,
                    "Model": "ridge",
                    "PredOpen": 10.1,
                    "PredHigh": 10.5,
                    "PredLow": 9.8,
                    "PredClose": 10.3,
                    "PredCloseRetPct": 3.0,
                    "PredRangePct": 7.0,
                    "ForecastCandleBias": "BULLISH",
                },
            ]
        )

        report = select_best_multi_session_forecasts(history, current, horizons=[1, 2])

        self.assertEqual(report["Horizon"].tolist(), [1, 2])
        self.assertEqual(report["ForecastWindow"].tolist(), ["T+1", "T+2"])
        self.assertAlmostEqual(float(report.loc[report["Horizon"].eq(2), "ForecastClose"].iloc[0]), 10.3)


if __name__ == "__main__":
    unittest.main()
