from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_deterministic_strategies import choose_market_algorithm, summarise_ticker_algorithms


class DeterministicStrategyAnalysisTest(unittest.TestCase):
    def test_choose_market_algorithm_prefers_positive_deploy_edge(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "Algorithm": "engine_hybrid",
                    "DEPLOYDays": 80,
                    "DeployVsAll10Pct": 0.1,
                    "DEPLOYHit10Pct": 55.0,
                    "DeployMinusReduce10Pct": -0.2,
                },
                {
                    "Algorithm": "exhaustion_overlay",
                    "DEPLOYDays": 70,
                    "DeployVsAll10Pct": 0.5,
                    "DEPLOYHit10Pct": 67.0,
                    "DeployMinusReduce10Pct": 0.3,
                },
                {
                    "Algorithm": "trend_breadth",
                    "DEPLOYDays": 90,
                    "DeployVsAll10Pct": 0.2,
                    "DEPLOYHit10Pct": 60.0,
                    "DeployMinusReduce10Pct": -0.1,
                },
            ]
        )
        self.assertEqual(choose_market_algorithm(summary), "exhaustion_overlay")

    def test_summarise_ticker_algorithms_aggregates_basket_and_pick_metrics(self) -> None:
        replay = pd.DataFrame(
            [
                {"Date": "2026-03-01", "Algorithm": "trend_leader", "Ticker": "AAA", "Fwd5Pct": 2.0, "Fwd10Pct": 4.0, "Excess5Pct": 1.0, "Excess10Pct": 2.0, "FutureDrawdown10Pct": -3.0},
                {"Date": "2026-03-01", "Algorithm": "trend_leader", "Ticker": "BBB", "Fwd5Pct": 0.0, "Fwd10Pct": 2.0, "Excess5Pct": -1.0, "Excess10Pct": 0.5, "FutureDrawdown10Pct": -4.0},
                {"Date": "2026-03-02", "Algorithm": "trend_leader", "Ticker": "CCC", "Fwd5Pct": -1.0, "Fwd10Pct": 1.0, "Excess5Pct": -0.5, "Excess10Pct": 0.2, "FutureDrawdown10Pct": -2.0},
                {"Date": "2026-03-01", "Algorithm": "recovery_strength", "Ticker": "DDD", "Fwd5Pct": 3.0, "Fwd10Pct": 6.0, "Excess5Pct": 2.0, "Excess10Pct": 3.0, "FutureDrawdown10Pct": -2.5},
            ]
        )
        summary = summarise_ticker_algorithms(replay, top_k=5).set_index("Algorithm")

        self.assertEqual(int(summary.loc["trend_leader", "SignalDays"]), 2)
        self.assertEqual(int(summary.loc["trend_leader", "TotalPicks"]), 3)
        self.assertAlmostEqual(float(summary.loc["trend_leader", "BasketAvgFwd10Pct"]), 2.0)
        self.assertAlmostEqual(float(summary.loc["trend_leader", "PickHit10Pct"]), 100.0)
        self.assertEqual(int(summary.loc["recovery_strength", "SignalDays"]), 1)
        self.assertAlmostEqual(float(summary.loc["recovery_strength", "BasketAvgExcess10Pct"]), 3.0)


if __name__ == "__main__":
    unittest.main()
