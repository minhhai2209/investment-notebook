from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_daily_feature_families import (
    summarise_daily_signals,
    walk_back_daily_feature_families,
)


class DailyFeatureFamilyWalkbackTest(unittest.TestCase):
    def test_daily_walkback_scores_predictive_family_by_horizon(self) -> None:
        rows = []
        dates = pd.bdate_range("2026-01-01", periods=170)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        for horizon in [1, 3]:
            for date_idx, trade_date in enumerate(dates):
                for ticker_idx, ticker in enumerate(tickers):
                    signal = float(ticker_idx - 2)
                    rows.append(
                        {
                            "Date": trade_date,
                            "Ticker": ticker,
                            "Horizon": horizon,
                            "ForecastWindow": f"T+{horizon}",
                            "PredictiveFeature": signal,
                            "TargetCloseRetPct": signal * 0.1 * horizon,
                        }
                    )
        sample = pd.DataFrame(rows)

        signals, _ = walk_back_daily_feature_families(
            sample,
            {"predictive": ["PredictiveFeature"]},
            target_column="TargetCloseRetPct",
            min_train_dates=40,
            test_block_dates=10,
        )
        summary = summarise_daily_signals(signals, top_fraction=0.2)

        self.assertEqual(set(summary["Horizon"].tolist()), {1, 3})
        self.assertTrue((summary["MeanRankIC"] > 0.9).all())
        self.assertTrue((summary["TopVsAllPct"] > 0.0).all())


if __name__ == "__main__":
    unittest.main()
