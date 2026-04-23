from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_deep_market_experiment import build_comparison_summary, summarise_model_group


class DeepMarketExperimentTest(unittest.TestCase):
    def test_summarise_model_group_computes_auc_accuracy_and_hit_rates(self) -> None:
        history = pd.DataFrame(
            [
                {"HorizonDays": 5, "Model": "mlp_small", "TargetUp": 1, "ProbabilityUp": 0.8, "FutureRetPct": 2.5},
                {"HorizonDays": 5, "Model": "mlp_small", "TargetUp": 0, "ProbabilityUp": 0.3, "FutureRetPct": -1.2},
                {"HorizonDays": 5, "Model": "mlp_small", "TargetUp": 1, "ProbabilityUp": 0.6, "FutureRetPct": 1.4},
                {"HorizonDays": 5, "Model": "mlp_small", "TargetUp": 0, "ProbabilityUp": 0.2, "FutureRetPct": -0.8},
            ]
        )

        summary = summarise_model_group(history, group_label="deep")
        row = summary.iloc[0]

        self.assertEqual(row["ModelGroup"], "deep")
        self.assertEqual(int(row["EvalDays"]), 4)
        self.assertAlmostEqual(float(row["Accuracy"]), 1.0)
        self.assertAlmostEqual(float(row["BullishHitPct"]), 100.0)
        self.assertAlmostEqual(float(row["BearishHitPct"]), 0.0)

    def test_build_comparison_summary_compares_best_deep_and_baseline_rows(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "ModelGroup": "baseline",
                    "HorizonDays": 5,
                    "Model": "hist_gbm",
                    "EvalDays": 100,
                    "AUC": 0.61,
                    "Brier": 0.245,
                    "Accuracy": 0.57,
                    "BullishDays": 30,
                    "BullishAvgFutureRetPct": 1.4,
                    "BullishHitPct": 60.0,
                    "BearishDays": 25,
                    "BearishAvgFutureRetPct": -0.5,
                    "BearishHitPct": 44.0,
                },
                {
                    "ModelGroup": "deep",
                    "HorizonDays": 5,
                    "Model": "mlp_small",
                    "EvalDays": 100,
                    "AUC": 0.66,
                    "Brier": 0.231,
                    "Accuracy": 0.6,
                    "BullishDays": 28,
                    "BullishAvgFutureRetPct": 1.8,
                    "BullishHitPct": 64.0,
                    "BearishDays": 21,
                    "BearishAvgFutureRetPct": -0.7,
                    "BearishHitPct": 39.0,
                },
                {
                    "ModelGroup": "deep",
                    "HorizonDays": 5,
                    "Model": "mlp_deep",
                    "EvalDays": 100,
                    "AUC": 0.64,
                    "Brier": 0.238,
                    "Accuracy": 0.58,
                    "BullishDays": 31,
                    "BullishAvgFutureRetPct": 1.6,
                    "BullishHitPct": 62.0,
                    "BearishDays": 22,
                    "BearishAvgFutureRetPct": -0.6,
                    "BearishHitPct": 40.0,
                },
            ]
        )

        comparison = build_comparison_summary(summary_df)
        row = comparison.iloc[0]

        self.assertEqual(int(row["HorizonDays"]), 5)
        self.assertEqual(row["WinningGroup"], "deep")
        self.assertEqual(row["WinningModel"], "mlp_small")
        self.assertAlmostEqual(float(row["DeepMinusBaselineAUC"]), 0.05)
        self.assertAlmostEqual(float(row["DeepMinusBaselineAccuracy"]), 0.03)
        self.assertAlmostEqual(float(row["DeepMinusBaselineBullishAvgFutureRetPct"]), 0.4)


if __name__ == "__main__":
    unittest.main()
