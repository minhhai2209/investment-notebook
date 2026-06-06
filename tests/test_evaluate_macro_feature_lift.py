import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.evaluate_macro_feature_lift import _build_lift_summary, build_macro_feature_frame


class MacroFeatureLiftTest(unittest.TestCase):
    def test_build_macro_feature_frame_lags_cached_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            pd.DataFrame(
                {
                    "Date": pd.date_range("2026-01-01", periods=30, freq="D"),
                    "Value": [100.0 + i for i in range(30)],
                }
            ).to_csv(cache_dir / "SP500.csv", index=False)

            sample = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=30, freq="D")})
            features, columns = build_macro_feature_frame(sample, cache_dir, factor_names=["SP500"], shift_days=1)

        self.assertIn("Macro_SP500_Ret1Pct", columns)
        row = features.loc[features["Date"] == pd.Timestamp("2026-01-03")].iloc[0]
        self.assertAlmostEqual(row["Macro_SP500_Ret1Pct"], 1.0, places=6)

    def test_build_lift_summary_compares_against_baseline(self) -> None:
        best = pd.DataFrame(
            [
                {
                    "FeatureSet": "baseline",
                    "Model": "baseline::ridge",
                    "TopKAvgExcess10Pct": 1.0,
                    "TopKHit10Pct": 55.0,
                    "AUC": 0.51,
                },
                {
                    "FeatureSet": "macro_global",
                    "Model": "macro_global::hist_gbm",
                    "TopKAvgExcess10Pct": 1.4,
                    "TopKHit10Pct": 58.0,
                    "AUC": 0.54,
                },
            ]
        )
        lift = _build_lift_summary(best)
        macro = lift[lift["FeatureSet"] == "macro_global"].iloc[0]
        self.assertAlmostEqual(macro["TopKAvgExcess10PctLift"], 0.4, places=6)
        self.assertAlmostEqual(macro["TopKHit10PctLift"], 3.0, places=6)
        self.assertAlmostEqual(macro["AUCLift"], 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
