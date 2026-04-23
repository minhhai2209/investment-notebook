from __future__ import annotations

import unittest

from scripts.analysis.ticker_color_overlay import summarise_ticker_overlay


class TickerColorOverlayTest(unittest.TestCase):
    def test_summarise_ticker_overlay_flags_usable_rows_and_scores_bias(self) -> None:
        rows = [
            {
                "TargetName": "GREEN",
                "Horizon": 10,
                "BestModel": "logistic_balanced",
                "BestAUC": 0.592,
                "BestStrongSignalHitPct": 65.6,
                "BestStrongSignalAvgSignedEdgePct": 1.38,
                "ProbabilityPositive": 0.79,
                "CurrentBias": "YES",
                "IsUsable": True,
            },
            {
                "TargetName": "RED",
                "Horizon": 10,
                "BestModel": "logistic_balanced",
                "BestAUC": 0.601,
                "BestStrongSignalHitPct": 49.0,
                "BestStrongSignalAvgSignedEdgePct": -1.50,
                "ProbabilityPositive": 0.28,
                "CurrentBias": "NO",
                "IsUsable": False,
            },
        ]

        summary = summarise_ticker_overlay(rows)

        self.assertEqual(summary["OverlayScore"], 4)
        self.assertEqual(len(summary["UsableRows"]), 1)
        self.assertIn("green T+10 YES", summary["Summary"])


if __name__ == "__main__":
    unittest.main()
