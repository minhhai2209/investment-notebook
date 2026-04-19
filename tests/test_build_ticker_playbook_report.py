from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.build_ticker_playbook_report import (
    build_human_summary,
    score_trade_metrics,
    select_best_playbook,
    summarise_trade_metrics,
)


class BuildTickerPlaybookReportTest(unittest.TestCase):
    def test_summarise_trade_metrics_aggregates_returns_drawdown_and_hold_days(self) -> None:
        trades = pd.DataFrame(
            [
                {"ReturnPct": 10.0, "WorstDrawdownPct": -4.0, "HoldDays": 5},
                {"ReturnPct": -5.0, "WorstDrawdownPct": -7.0, "HoldDays": 4},
                {"ReturnPct": 2.0, "WorstDrawdownPct": -2.0, "HoldDays": 3},
            ]
        )

        metrics = summarise_trade_metrics(trades)

        self.assertEqual(float(metrics["Trades"]), 3.0)
        self.assertAlmostEqual(float(metrics["WinRatePct"]), 66.66666666666666)
        self.assertAlmostEqual(float(metrics["AvgReturnPct"]), 7.0 / 3.0)
        self.assertAlmostEqual(float(metrics["MedianReturnPct"]), 2.0)
        self.assertAlmostEqual(float(metrics["ProfitFactor"]), 12.0 / 5.0)
        self.assertAlmostEqual(float(metrics["AvgHoldDays"]), 4.0)
        self.assertAlmostEqual(float(metrics["WorstDrawdownPct"]), -7.0)

    def test_score_trade_metrics_requires_minimum_trade_count(self) -> None:
        metrics = {
            "Trades": 2.0,
            "WinRatePct": 100.0,
            "AvgReturnPct": 9.0,
            "MedianReturnPct": 9.0,
            "ProfitFactor": 5.0,
            "AvgHoldDays": 4.0,
            "WorstDrawdownPct": -3.0,
        }
        self.assertEqual(score_trade_metrics(metrics), -999.0)

    def test_select_best_playbook_keeps_highest_robust_score_per_ticker(self) -> None:
        configs = pd.DataFrame(
            [
                {"Ticker": "AAA", "RobustScore": 8.0, "TestScore": 4.0, "TrainScore": 9.0, "AllScore": 7.0},
                {"Ticker": "AAA", "RobustScore": 9.0, "TestScore": 3.0, "TrainScore": 11.0, "AllScore": 8.0},
                {"Ticker": "BBB", "RobustScore": 7.0, "TestScore": 7.0, "TrainScore": 7.0, "AllScore": 7.0},
            ]
        )

        best = select_best_playbook(configs)

        self.assertEqual(int(best.shape[0]), 2)
        self.assertEqual(float(best.loc[best["Ticker"] == "AAA", "RobustScore"].iloc[0]), 9.0)

    def test_build_human_summary_includes_ticker_and_rule_lines(self) -> None:
        best = pd.DataFrame(
            [
                {
                    "Ticker": "VIC",
                    "StrategyFamily": "washout_reclaim",
                    "StrategyLabel": "washout_reclaim/dd60<=-12%/rsi<=35",
                    "TrainTrades": 8,
                    "TestTrades": 3,
                    "AllTrades": 11,
                    "TestWinRatePct": 66.7,
                    "TestAvgReturnPct": 4.2,
                    "TestMedianReturnPct": 3.1,
                    "TestAvgHoldDays": 5.0,
                    "TestWorstDrawdownPct": -6.5,
                    "AllWinRatePct": 72.7,
                    "AllAvgReturnPct": 5.4,
                    "AllMedianReturnPct": 4.0,
                    "AllAvgHoldDays": 5.3,
                    "AllWorstDrawdownPct": -9.1,
                    "LatestSignal": True,
                }
            ]
        )

        summary = build_human_summary(best)

        self.assertIn("## VIC", summary)
        self.assertIn("Playbook: `washout_reclaim`", summary)
        self.assertIn("Latest signal on snapshot: `True`", summary)


if __name__ == "__main__":
    unittest.main()
