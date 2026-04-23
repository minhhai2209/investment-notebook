from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_ticker_color_models import build_comparison_summary, summarise_ticker_color_models


class TickerColorModelsTest(unittest.TestCase):
    def test_summarise_ticker_color_models_computes_signal_quality(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "VIC",
                    "Horizon": 3,
                    "TargetName": "GREEN",
                    "Model": "hist_gbm",
                    "TargetPositive": 1,
                    "ProbabilityPositive": 0.72,
                    "FutureCloseRetPct": 3.4,
                    "FutureHighRetPct": 5.2,
                    "FutureLowRetPct": -1.1,
                },
                {
                    "Ticker": "VIC",
                    "Horizon": 3,
                    "TargetName": "GREEN",
                    "Model": "hist_gbm",
                    "TargetPositive": 0,
                    "ProbabilityPositive": 0.21,
                    "FutureCloseRetPct": -2.2,
                    "FutureHighRetPct": 0.8,
                    "FutureLowRetPct": -3.5,
                },
            ]
        )

        summary = summarise_ticker_color_models(history)
        row = summary.iloc[0]

        self.assertEqual(row["Ticker"], "VIC")
        self.assertEqual(row["TargetName"], "GREEN")
        self.assertAlmostEqual(float(row["Accuracy"]), 1.0)
        self.assertAlmostEqual(float(row["StrongSignalHitPct"]), 100.0)
        self.assertAlmostEqual(float(row["StrongSignalAvgSignedEdgePct"]), 3.4)

    def test_build_comparison_summary_selects_best_model_per_target(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "Ticker": "HPG",
                    "Horizon": 5,
                    "TargetName": "GREEN",
                    "Model": "random_forest",
                    "EvalDays": 100,
                    "PositiveBaseRatePct": 55.0,
                    "AUC": 0.58,
                    "Brier": 0.24,
                    "Accuracy": 0.57,
                    "StrongSignalDays": 15,
                    "StrongSignalHitPct": 60.0,
                    "StrongSignalAvgSignedEdgePct": 1.2,
                    "StrongSignalAvgCloseRetPct": 1.2,
                },
                {
                    "Ticker": "HPG",
                    "Horizon": 5,
                    "TargetName": "GREEN",
                    "Model": "hist_gbm",
                    "EvalDays": 100,
                    "PositiveBaseRatePct": 55.0,
                    "AUC": 0.62,
                    "Brier": 0.23,
                    "Accuracy": 0.59,
                    "StrongSignalDays": 18,
                    "StrongSignalHitPct": 66.0,
                    "StrongSignalAvgSignedEdgePct": 1.6,
                    "StrongSignalAvgCloseRetPct": 1.6,
                },
            ]
        )

        comparison = build_comparison_summary(summary_df)
        row = comparison.iloc[0]

        self.assertEqual(row["Ticker"], "HPG")
        self.assertEqual(row["BestModel"], "hist_gbm")
        self.assertAlmostEqual(float(row["BestAUC"]), 0.62)
        self.assertAlmostEqual(float(row["BestStrongSignalAvgSignedEdgePct"]), 1.6)


if __name__ == "__main__":
    unittest.main()
