from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.evaluate_vic_index_expiry_features import (
    add_index_expiry_features,
    build_derivative_expiry_features,
    build_ex_vin_market_features,
)
from scripts.analysis.evaluate_ohlc_models import build_ticker_ohlc_sample


class VicIndexExpiryFeatureExperimentTest(unittest.TestCase):
    @staticmethod
    def _write_daily_cache(path: Path, closes: list[float], *, start: str = "2026-05-01") -> None:
        dates = pd.bdate_range(start, periods=len(closes))
        frame = pd.DataFrame(
            {
                "date_vn": dates.strftime("%Y-%m-%d"),
                "open": [value - 0.5 for value in closes],
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "volume": [1_000_000 + (1_000 * idx) for idx in range(len(closes))],
            }
        )
        frame.to_csv(path, index=False)

    def test_derivative_expiry_features_use_third_thursday_trading_date(self) -> None:
        dates = pd.bdate_range("2026-06-15", "2026-06-22")

        features = build_derivative_expiry_features(dates).set_index("Date")

        self.assertEqual(float(features.loc[pd.Timestamp("2026-06-18"), "DerivExpiryDay"]), 1.0)
        self.assertEqual(float(features.loc[pd.Timestamp("2026-06-17"), "DerivPreExpiry3Sessions"]), 1.0)
        self.assertEqual(float(features.loc[pd.Timestamp("2026-06-19"), "DerivPostExpiry3Sessions"]), 1.0)

    def test_derivative_expiry_features_estimate_next_expiry_after_cache_tail(self) -> None:
        dates = pd.bdate_range("2026-06-15", "2026-06-23")

        features = build_derivative_expiry_features(dates).set_index("Date")

        self.assertGreater(float(features.loc[pd.Timestamp("2026-06-23"), "DerivDaysToExpiry"]), 0.0)
        self.assertEqual(float(features.loc[pd.Timestamp("2026-06-23"), "DerivPostExpiry3Sessions"]), 1.0)

    def test_build_ex_vin_market_features_uses_non_vin_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            self._write_daily_cache(history_dir / "VIC_daily.csv", [100, 102, 104, 106, 108, 110])
            self._write_daily_cache(history_dir / "VHM_daily.csv", [50, 51, 52, 53, 54, 55])
            self._write_daily_cache(history_dir / "FPT_daily.csv", [20, 21, 22, 23, 24, 25])
            self._write_daily_cache(history_dir / "VNINDEX_daily.csv", [1000, 1005, 1010, 1015, 1020, 1025])

            features = build_ex_vin_market_features(history_dir, ticker="VIC")

            self.assertIn("ExVinRet1Pct", features.columns)
            self.assertIn("VNIndexRet1MinusExVinRet1Pct", features.columns)
            self.assertAlmostEqual(float(features["ExVinRet1Pct"].iloc[1]), 5.0)

    def test_add_index_expiry_features_appends_lagged_experiment_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            closes = [100 + idx for idx in range(90)]
            self._write_daily_cache(history_dir / "VIC_daily.csv", closes)
            self._write_daily_cache(history_dir / "VNINDEX_daily.csv", [1000 + idx for idx in range(90)])
            self._write_daily_cache(history_dir / "FPT_daily.csv", [50 + idx for idx in range(90)])

            sample = build_ticker_ohlc_sample("VIC", history_dir, max_horizon=1)
            enhanced, added_columns = add_index_expiry_features(sample, history_dir, ticker="VIC")

            self.assertIn("DerivDaysToExpiry", added_columns)
            self.assertIn("ExVinRet1Pct", added_columns)
            self.assertIn("ExVinRet1Pct_Lag1", added_columns)
            self.assertIn("DerivDaysToExpiry", enhanced.columns)
            self.assertIn("ExVinRet1Pct_Lag1", enhanced.columns)


if __name__ == "__main__":
    unittest.main()
