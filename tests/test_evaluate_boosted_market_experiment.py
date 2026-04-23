from __future__ import annotations

import unittest

from scripts.analysis.evaluate_boosted_market_experiment import build_boosted_model_factories


class BoostedMarketExperimentTest(unittest.TestCase):
    def test_build_boosted_model_factories_exposes_expected_models(self) -> None:
        factories = build_boosted_model_factories()

        self.assertEqual(set(factories), {"xgb_small", "xgb_mid"})


if __name__ == "__main__":
    unittest.main()
