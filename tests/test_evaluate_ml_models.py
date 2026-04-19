from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_ml_models import choose_best_ml_model, summarise_ml_models


class MlReplayAnalysisTest(unittest.TestCase):
    def test_choose_best_ml_model_prefers_topk_excess(self) -> None:
        summary = pd.DataFrame(
            [
                {"Model": "logistic_balanced", "TopKAvgExcess10Pct": 0.5, "AUC": 0.58, "TopKHit10Pct": 55.0},
                {"Model": "random_forest", "TopKAvgExcess10Pct": 0.9, "AUC": 0.57, "TopKHit10Pct": 54.0},
                {"Model": "hist_gbm", "TopKAvgExcess10Pct": 0.8, "AUC": 0.61, "TopKHit10Pct": 57.0},
            ]
        )
        self.assertEqual(choose_best_ml_model(summary), "random_forest")

    def test_summarise_ml_models_aggregates_classification_and_topk(self) -> None:
        history = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-03-01"), "Model": "logistic_balanced", "Ticker": "AAA", "ProbabilityOutperform10d": 0.9, "TargetOutperform10d": 1, "Fwd10Pct": 4.0, "Excess10Pct": 2.0},
                {"Date": pd.Timestamp("2026-03-01"), "Model": "logistic_balanced", "Ticker": "BBB", "ProbabilityOutperform10d": 0.2, "TargetOutperform10d": 0, "Fwd10Pct": -1.0, "Excess10Pct": -2.0},
                {"Date": pd.Timestamp("2026-03-02"), "Model": "logistic_balanced", "Ticker": "CCC", "ProbabilityOutperform10d": 0.8, "TargetOutperform10d": 1, "Fwd10Pct": 3.0, "Excess10Pct": 1.0},
                {"Date": pd.Timestamp("2026-03-01"), "Model": "random_forest", "Ticker": "DDD", "ProbabilityOutperform10d": 0.7, "TargetOutperform10d": 1, "Fwd10Pct": 5.0, "Excess10Pct": 2.5},
                {"Date": pd.Timestamp("2026-03-01"), "Model": "random_forest", "Ticker": "EEE", "ProbabilityOutperform10d": 0.4, "TargetOutperform10d": 0, "Fwd10Pct": -2.0, "Excess10Pct": -1.5},
            ]
        )
        current = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2026-03-20"), "Model": "logistic_balanced", "Ticker": "AAA", "ProbabilityOutperform10d": 0.6},
                {"Date": pd.Timestamp("2026-03-20"), "Model": "logistic_balanced", "Ticker": "BBB", "ProbabilityOutperform10d": 0.4},
                {"Date": pd.Timestamp("2026-03-20"), "Model": "random_forest", "Ticker": "DDD", "ProbabilityOutperform10d": 0.7},
                {"Date": pd.Timestamp("2026-03-20"), "Model": "random_forest", "Ticker": "EEE", "ProbabilityOutperform10d": 0.3},
            ]
        )
        summary = summarise_ml_models(history, current, top_k=1).set_index("Model")

        self.assertAlmostEqual(float(summary.loc["logistic_balanced", "Accuracy"]), 1.0)
        self.assertAlmostEqual(float(summary.loc["logistic_balanced", "TopKAvgFwd10Pct"]), 3.5)
        self.assertAlmostEqual(float(summary.loc["random_forest", "CurrentTopKMeanProbability"]), 0.7)


if __name__ == "__main__":
    unittest.main()
