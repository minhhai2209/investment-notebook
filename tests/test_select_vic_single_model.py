from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.select_vic_single_model import add_engineered_features, select_winner


class SelectVicSingleModelTest(unittest.TestCase):
    def test_engineered_features_add_pressure_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "TickerRet1Pct": [1.0, -1.0],
                "TickerRet5Pct": [5.0, -5.0],
                "TickerRet20Pct": [20.0, -20.0],
                "IndexRet5Pct": [2.0, -2.0],
                "IndexRet20Pct": [10.0, -10.0],
                "VN30Ret5Pct": [1.5, -1.5],
                "TickerVolRatio20": [2.0, 0.5],
                "TickerRangePct": [3.0, 4.0],
                "TickerDistSMA20Pct": [10.0, -10.0],
                "TickerDistSMA50Pct": [15.0, -15.0],
                "TickerVolatility10": [2.0, 5.0],
                "TickerUpperWickPct": [1.0, 4.0],
                "TickerLowerWickPct": [2.0, 1.0],
                "TickerRangePos20": [0.9, 0.2],
                "TickerRangePos60": [0.7, 0.4],
            }
        )

        enriched, added = add_engineered_features(frame)

        self.assertIn("TickerRetStackMeanPct", added)
        self.assertIn("TickerRet20VolumePressure", added)
        self.assertIn("TickerWickImbalancePct", added)
        self.assertAlmostEqual(float(enriched.loc[0, "TickerRet20VolumePressure"]), 40.0)
        self.assertAlmostEqual(float(enriched.loc[0, "TickerWickImbalancePct"]), -1.0)

    def test_select_winner_prefers_qualified_direction(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "FeatureSet": "engineered_all",
                    "Model": "logit_direction",
                    "Objective": "direction",
                    "SelectionScore": 15.0,
                    "HoldoutRows": 5,
                    "HoldoutDirectionHitPct": 80.0,
                    "WalkbackDirectionHitPct": 55.0,
                    "HoldoutCloseMAEPct": float("nan"),
                    "WalkbackCloseMAEPct": float("nan"),
                },
                {
                    "FeatureSet": "engineered_all",
                    "Model": "ridge_close_ret",
                    "Objective": "price",
                    "SelectionScore": 1.0,
                    "HoldoutRows": 5,
                    "HoldoutDirectionHitPct": 40.0,
                    "WalkbackDirectionHitPct": 50.0,
                    "HoldoutCloseMAEPct": 2.0,
                    "WalkbackCloseMAEPct": 3.0,
                },
            ]
        )

        winner = select_winner(candidates, expected_holdout_rows=5)

        self.assertEqual(winner["Model"], "logit_direction")
        self.assertEqual(winner["Objective"], "direction")

    def test_select_winner_uses_price_when_direction_unqualified(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "FeatureSet": "engineered_all",
                    "Model": "logit_direction",
                    "Objective": "direction",
                    "SelectionScore": 15.0,
                    "HoldoutRows": 5,
                    "HoldoutDirectionHitPct": 80.0,
                    "WalkbackDirectionHitPct": 45.0,
                    "HoldoutCloseMAEPct": float("nan"),
                    "WalkbackCloseMAEPct": float("nan"),
                },
                {
                    "FeatureSet": "engineered_all",
                    "Model": "ridge_close_ret",
                    "Objective": "price",
                    "SelectionScore": 1.0,
                    "HoldoutRows": 5,
                    "HoldoutDirectionHitPct": 40.0,
                    "WalkbackDirectionHitPct": 50.0,
                    "HoldoutCloseMAEPct": 2.0,
                    "WalkbackCloseMAEPct": 3.0,
                },
            ]
        )

        winner = select_winner(candidates, expected_holdout_rows=5)

        self.assertEqual(winner["Model"], "ridge_close_ret")
        self.assertEqual(winner["Objective"], "price")


if __name__ == "__main__":
    unittest.main()
