from __future__ import annotations

import unittest

from scripts.analysis.ticker_specialization_overlay import summarise_specialized_ticker_setup


class TickerSpecializationOverlayTest(unittest.TestCase):
    def test_vic_style_momentum_burst_distribution_scores_negative(self) -> None:
        overlay = summarise_specialized_ticker_setup(
            "VIC",
            {
                "Archetype": "momentum_high_beta",
                "BestTimingNetEdgePct": -0.28,
                "T10NetEdgePct": -5.17,
                "LatestBurstSignalAge": 0,
                "BurstNextDayPositiveRate": 70.45,
                "BurstNextDayStrongRate": 45.45,
                "BurstAvgThreeDayDrawdownPct": -2.15,
                "ExecutionBias": "distribution",
                "BurstExecutionBias": "failed_day2_followthrough",
                "ExecutionNote": "Burst còn rất mới nhưng tape mở kéo rồi fail.",
            },
        )

        self.assertEqual(overlay["Regime"], "fresh_burst_distribution")
        self.assertLess(overlay["OverlayScore"], 0)
        self.assertIn("burst rất mới", overlay["Summary"])
        self.assertIn("fail follow-through", " | ".join(overlay["Signals"]))

    def test_quality_trend_scores_positive(self) -> None:
        overlay = summarise_specialized_ticker_setup(
            "MBB",
            {
                "Archetype": "quality_trend",
                "BestTimingNetEdgePct": 1.73,
                "T10NetEdgePct": 1.04,
                "LatestBurstSignalAge": 31,
                "BurstNextDayPositiveRate": 83.33,
                "BurstAvgThreeDayDrawdownPct": -0.53,
                "ExecutionBias": "neutral",
                "BurstExecutionBias": "normal_tactical_management",
                "ExecutionNote": "Đã qua giai đoạn burst đầu.",
            },
        )

        self.assertEqual(overlay["Regime"], "trend_persistence_pullback_add")
        self.assertGreater(overlay["OverlayScore"], 0)
        self.assertIn("giữ trend", overlay["Summary"])

    def test_post_burst_t25_supply_regime_is_detected(self) -> None:
        overlay = summarise_specialized_ticker_setup(
            "VHM",
            {
                "Archetype": "momentum_high_beta",
                "LatestBurstSignalAge": 2,
                "ExecutionBias": "distribution",
                "BurstExecutionBias": "respect_t25_supply",
                "BurstNextDayPositiveRate": 70.97,
                "BurstNextDayStrongRate": 35.48,
                "BurstThirdDayNegativeRate": 40.0,
                "BurstAvgThreeDayDrawdownPct": -2.95,
                "ExecutionNote": "Burst đã sang nhịp dễ gặp cung T+2.5.",
            },
        )

        self.assertEqual(overlay["Regime"], "post_burst_t25_supply")
        self.assertLess(overlay["OverlayScore"], 0)
        self.assertIn("cung T+2.5", overlay["Summary"])


if __name__ == "__main__":
    unittest.main()
