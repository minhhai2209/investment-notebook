from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.evaluate_intraday_feature_families import (
    _micro_volume_rows,
    build_intraday_feature_families,
    summarise_family_signals,
    summarise_feature_stats,
    walk_back_feature_families,
)


class IntradayFeatureFamilyWalkbackTest(unittest.TestCase):
    def test_walkback_scores_predictive_feature_family(self) -> None:
        rows = []
        dates = pd.bdate_range("2026-01-01", periods=70)
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        for date_idx, trade_date in enumerate(dates):
            for ticker_idx, ticker in enumerate(tickers):
                signal = float(ticker_idx - 2)
                rows.append(
                    {
                        "TradeDate": trade_date,
                        "SnapshotTs": trade_date + pd.Timedelta(hours=10),
                        "SnapshotDate": trade_date.strftime("%Y-%m-%d"),
                        "SnapshotTimeBucket": "AM_EARLY",
                        "Ticker": ticker,
                        "PredictiveFeature": signal,
                        "NoiseFeature": float((date_idx + ticker_idx) % 3),
                        "TargetCloseRetPct": signal * 0.2,
                    }
                )
        sample = pd.DataFrame(rows)
        families = {
            "predictive": ["PredictiveFeature"],
            "noise": ["NoiseFeature"],
        }

        signals, feature_stats = walk_back_feature_families(
            sample,
            families,
            target_column="TargetCloseRetPct",
            min_train_dates=20,
            test_block_dates=10,
        )
        summary = summarise_family_signals(signals, top_fraction=0.2, min_group_size=5)
        feature_summary = summarise_feature_stats(feature_stats)

        predictive = summary[summary["Family"] == "predictive"].iloc[0]
        self.assertGreater(float(predictive["MeanRankIC"]), 0.9)
        self.assertGreater(float(predictive["TopVsAllPct"]), 0.0)
        predictive_feature = feature_summary[feature_summary["Feature"] == "PredictiveFeature"].iloc[0]
        self.assertEqual(float(predictive_feature["SelectedPct"]), 100.0)

    def test_micro_volume_rows_capture_trailing_pressure_and_family(self) -> None:
        timestamps = pd.date_range("2026-06-01 09:00", periods=70, freq="min", tz="Asia/Ho_Chi_Minh")
        frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "TradeDate": timestamps.normalize(),
                "Close": [100 + (idx * 0.1) for idx in range(len(timestamps))],
                "Volume": [100.0 + idx for idx in range(len(timestamps))],
            }
        )

        micro = _micro_volume_rows(frame, prefix="Ticker")
        families = build_intraday_feature_families(list(micro.columns))

        latest = micro.iloc[-1]
        self.assertGreater(float(latest["TickerLast15mVolumePctSession"]), 0.0)
        self.assertGreater(float(latest["TickerVolumeAcceleration5v30"]), 0.0)
        self.assertGreaterEqual(float(latest["TickerUpVolumeShare15m"]), 0.0)
        self.assertIn("ticker_micro_volume", families)
        self.assertIn("ticker_price_volume_interaction", families)


if __name__ == "__main__":
    unittest.main()
