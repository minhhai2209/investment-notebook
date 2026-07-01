from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analysis.evaluate_curated_intraday_models import fit_current_predictions, summarise_metrics


class CuratedIntradayModelTest(unittest.TestCase):
    def _sample_row(
        self,
        ticker: str,
        trade_date: str,
        *,
        signal: float,
        target: float | None,
    ) -> dict[str, object]:
        timestamp = pd.Timestamp(f"{trade_date} 10:00:00")
        if target is None:
            low = close = high = np.nan
        else:
            low = target - 0.2
            close = target
            high = target + 0.2
        return {
            "TradeDate": pd.Timestamp(trade_date),
            "SnapshotDate": trade_date,
            "SnapshotTs": timestamp,
            "SnapshotTimeBucket": "AM_EARLY",
            "Ticker": ticker,
            "PrevClose": 100.0,
            "Base": 100.0 + signal,
            "SnapshotRetFromPrevClosePct": signal,
            "Signal": signal,
            "TargetLowRetPct": low,
            "TargetCloseRetPct": close,
            "TargetHighRetPct": high,
        }

    def test_current_prediction_uses_latest_row_per_ticker(self) -> None:
        rows = []
        for idx, trade_date in enumerate(pd.bdate_range("2026-01-01", periods=12)):
            date_text = trade_date.strftime("%Y-%m-%d")
            rows.append(self._sample_row("VIC", date_text, signal=float(idx), target=float(idx) * 0.1))
            rows.append(self._sample_row("AAA", date_text, signal=float(idx + 1), target=float(idx + 1) * 0.1))
        rows.append(self._sample_row("VIC", "2026-01-20", signal=20.0, target=None))
        rows.append(self._sample_row("AAA", "2026-01-21", signal=21.0, target=None))
        feature_sets = {"price_volume_core": ["Signal"]}

        current = fit_current_predictions(pd.DataFrame(rows), feature_sets)

        self.assertEqual(set(current["Ticker"]), {"AAA", "VIC"})
        vic = current[current["Ticker"] == "VIC"].iloc[0]
        self.assertEqual(str(pd.Timestamp(vic["TradeDate"]).date()), "2026-01-20")
        self.assertEqual(vic["Model"], "ridge")
        self.assertEqual(vic["FeatureSet"], "price_volume_core")

    def test_all_metrics_are_aggregated_once_per_bucket_model(self) -> None:
        rows = []
        for ticker in ["AAA", "VIC"]:
            for idx in range(3):
                actual = float(idx + 1)
                pred = actual - 0.1
                rows.append(
                    {
                        "Ticker": ticker,
                        "SnapshotTimeBucket": "AM_EARLY",
                        "FeatureSet": "price_volume_core",
                        "Model": "ridge",
                        "UsedFeatureCount": 1,
                        "TargetLowRetPct": actual - 0.5,
                        "TargetCloseRetPct": actual,
                        "TargetHighRetPct": actual + 0.5,
                        "PredTargetLowRetPct": pred - 0.5,
                        "PredTargetCloseRetPct": pred,
                        "PredTargetHighRetPct": pred + 0.5,
                        "ActualRangePct": 1.0,
                        "PredRangePct": 1.0,
                        "CloseAbsErrPct": 0.1,
                        "HighAbsErrPct": 0.1,
                        "LowAbsErrPct": 0.1,
                        "CloseDirHit": True,
                    }
                )

        metrics = summarise_metrics(pd.DataFrame(rows))
        all_rows = metrics[metrics["Scope"] == "ALL"]

        self.assertEqual(len(all_rows), 1)
        self.assertEqual(int(all_rows.iloc[0]["Rows"]), 6)


if __name__ == "__main__":
    unittest.main()
