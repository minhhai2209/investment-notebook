from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_bctt_feature_lift import _with_variant_columns


class BcttFeatureLiftTest(unittest.TestCase):
    def test_with_variant_columns_splits_scoped_model_name(self) -> None:
        frame = pd.DataFrame(
            [
                {"Model": "baseline::random_forest"},
                {"Model": "hybrid_growth::hist_gbm"},
            ]
        )

        out = _with_variant_columns(frame)

        self.assertEqual(list(out["FeatureSet"]), ["baseline", "hybrid_growth"])
        self.assertEqual(list(out["BaseModel"]), ["random_forest", "hist_gbm"])

    def test_with_variant_columns_keeps_empty_frame(self) -> None:
        empty = pd.DataFrame()
        out = _with_variant_columns(empty)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
