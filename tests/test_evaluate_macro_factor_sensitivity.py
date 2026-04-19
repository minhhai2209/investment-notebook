from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analysis.evaluate_macro_factor_sensitivity import build_case_studies, summarise_factor_shocks


class MacroFactorSensitivityTest(unittest.TestCase):
    def test_summarise_factor_shocks_counts_and_returns(self) -> None:
        index = pd.date_range("2026-01-01", periods=20, freq="D")
        factor = pd.Series(np.linspace(-3.0, 3.0, len(index)), index=index)
        asset_ret = pd.Series(np.linspace(-1.0, 1.0, len(index)), index=index)
        asset_fwd5 = pd.Series(np.linspace(-2.0, 2.0, len(index)), index=index)

        summary = summarise_factor_shocks(asset_ret, asset_fwd5, factor)

        self.assertGreater(summary["UpShockCount"], 0.0)
        self.assertGreater(summary["DownShockCount"], 0.0)
        self.assertTrue(np.isfinite(summary["AvgRetOnUpShockPct"]))

    def test_build_case_studies_picks_highest_abs_corr(self) -> None:
        frame = pd.DataFrame(
            [
                {"Ticker": "FPT", "Factor": "NASDAQ", "CurrentCorr60": 0.7, "CurrentAlignment20Pct": 1.2},
                {"Ticker": "FPT", "Factor": "VIX", "CurrentCorr60": -0.5, "CurrentAlignment20Pct": 0.8},
                {"Ticker": "FPT", "Factor": "WTI_USD", "CurrentCorr60": 0.1, "CurrentAlignment20Pct": -0.1},
                {"Ticker": "HPG", "Factor": "WTI_USD", "CurrentCorr60": 0.6, "CurrentAlignment20Pct": 0.4},
                {"Ticker": "HPG", "Factor": "USD_BROAD", "CurrentCorr60": -0.2, "CurrentAlignment20Pct": 0.3},
            ]
        )

        case = build_case_studies(frame, case_tickers=["FPT", "HPG"], top_factors=2)

        self.assertEqual(len(case), 4)
        self.assertEqual(list(case[case["Ticker"] == "FPT"]["Factor"]), ["NASDAQ", "VIX"])


if __name__ == "__main__":
    unittest.main()
