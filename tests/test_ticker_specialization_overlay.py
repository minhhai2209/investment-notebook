from __future__ import annotations

import unittest

from scripts.analysis.ticker_specialization_overlay import summarise_specialized_ticker_setup


class TickerSpecializationOverlayTest(unittest.TestCase):
    def test_vic_style_momentum_overlay_describes_without_scoring(self) -> None:
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

        self.assertEqual(overlay["Regime"], "momentum_high_beta")
        self.assertEqual(overlay["OverlayScore"], 0)
        self.assertIn("momentum high beta", overlay["Summary"])
        self.assertIn("fail follow-through", " | ".join(overlay["Signals"]))

    def test_clean_momentum_high_beta_still_has_zero_overlay_score(self) -> None:
        overlay = summarise_specialized_ticker_setup(
            "VIC",
            {
                "Archetype": "momentum_high_beta",
                "BestTimingNetEdgePct": 5.2,
                "T10NetEdgePct": 2.4,
                "LatestBurstSignalAge": 5,
                "BurstNextDayPositiveRate": 72.0,
                "BurstNextDayStrongRate": 44.0,
                "BurstAvgThreeDayDrawdownPct": -0.6,
                "ExecutionBias": "accumulation",
                "BurstExecutionBias": "clean_followthrough",
                "ExecutionNote": "Continuation sạch sau burst.",
            },
        )

        self.assertEqual(overlay["Regime"], "momentum_high_beta")
        self.assertEqual(overlay["OverlayScore"], 0)
        self.assertIn("OverlayScore", overlay)

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
        self.assertEqual(overlay["OverlayScore"], 0)
        self.assertIn("quality_trend", overlay["Summary"])

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
        self.assertEqual(overlay["OverlayScore"], 0)
        self.assertIn("cung T+2.5", overlay["Summary"])


if __name__ == "__main__":
    unittest.main()
