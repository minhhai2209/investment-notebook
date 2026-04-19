from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_vnindex_models import choose_best_vnindex_model, summarise_vnindex_models


class VnindexMlReplayTest(unittest.TestCase):
    def test_choose_best_vnindex_model_prefers_higher_auc(self) -> None:
        summary = pd.DataFrame(
            [
                {"HorizonDays": 5, "Model": "logistic_balanced", "AUC": 0.51, "BullishAvgFutureRetPct": 0.4, "Accuracy": 0.5},
                {"HorizonDays": 5, "Model": "random_forest", "AUC": 0.63, "BullishAvgFutureRetPct": 0.8, "Accuracy": 0.58},
                {"HorizonDays": 10, "Model": "hist_gbm", "AUC": 0.59, "BullishAvgFutureRetPct": 1.2, "Accuracy": 0.54},
                {"HorizonDays": 10, "Model": "random_forest", "AUC": 0.57, "BullishAvgFutureRetPct": 0.9, "Accuracy": 0.53},
            ]
        )
        self.assertEqual(choose_best_vnindex_model(summary, 5), "random_forest")
        self.assertEqual(choose_best_vnindex_model(summary, 10), "hist_gbm")

    def test_summarise_vnindex_models_aggregates_states(self) -> None:
        history = pd.DataFrame(
            [
                {"HorizonDays": 5, "Model": "random_forest", "ProbabilityUp": 0.7, "TargetUp": 1, "FutureRetPct": 2.0},
                {"HorizonDays": 5, "Model": "random_forest", "ProbabilityUp": 0.3, "TargetUp": 0, "FutureRetPct": -1.0},
                {"HorizonDays": 5, "Model": "logistic_balanced", "ProbabilityUp": 0.6, "TargetUp": 1, "FutureRetPct": 1.5},
                {"HorizonDays": 10, "Model": "hist_gbm", "ProbabilityUp": 0.2, "TargetUp": 0, "FutureRetPct": -2.0},
                {"HorizonDays": 10, "Model": "hist_gbm", "ProbabilityUp": 0.8, "TargetUp": 1, "FutureRetPct": 3.0},
            ]
        )
        summary = summarise_vnindex_models(history)
        row = summary[(summary["HorizonDays"] == 5) & (summary["Model"] == "random_forest")].iloc[0]
        self.assertAlmostEqual(float(row["Accuracy"]), 1.0)
        self.assertEqual(int(row["BullishDays"]), 1)
        self.assertAlmostEqual(float(row["BullishAvgFutureRetPct"]), 2.0)


if __name__ == "__main__":
    unittest.main()
