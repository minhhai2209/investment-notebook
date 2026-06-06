from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_position_ml_review import build_position_ml_review


class BuildPositionMlReviewTest(unittest.TestCase):
    def test_build_position_ml_review_uses_forecasts_without_manual_action_rules(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            out_dir = root / "out"
            analysis_dir = out_dir / "analysis"
            candidates_dir = analysis_dir / "candidates"
            candidates_dir.mkdir(parents=True)

            pd.DataFrame([{"Ticker": "MBB", "Last": 26.0, "LotSize": 100}]).to_csv(out_dir / "universe.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Ticker": "MBB",
                        "Horizon": 1,
                        "ForecastWindow": "T+1",
                        "ForecastOpen": 26.1,
                        "ForecastHigh": 26.25,
                        "ForecastLow": 25.8,
                        "ForecastClose": 25.9,
                        "Model": "ridge",
                        "CloseMAEPct": 1.2,
                        "CloseDirHitPct": 60.0,
                    },
                    {
                        "Ticker": "MBB",
                        "Horizon": 3,
                        "ForecastWindow": "T+3",
                        "ForecastOpen": 26.0,
                        "ForecastHigh": 26.5,
                        "ForecastLow": 25.7,
                        "ForecastClose": 26.3,
                        "Model": "hist_gbm",
                        "CloseMAEPct": 1.6,
                        "CloseDirHitPct": 57.5,
                    },
                ]
            ).to_csv(analysis_dir / "ml_ohlc_multi_session.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Ticker": "MBB",
                        "Horizon": 3,
                        "ForecastWindow": "T+3",
                        "Model": "ridge",
                        "PredPeakRetPct": 2.6,
                        "PredPeakDay": 2.0,
                        "PredDrawdownPct": -0.4,
                        "PredCloseRetPct": 1.4,
                        "PredNetEdgePct": 2.2,
                        "EvalRows": 40,
                        "PeakRetMAEPct": 1.5,
                        "CloseMAEPct": 2.8,
                        "TradeScoreHitPct": 57.5,
                    }
                ]
            ).to_csv(analysis_dir / "ml_single_name_timing.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Ticker": "MBB",
                        "Decision": "chờ",
                        "ModelDecisionBasis": "per_ticker_ml_not_robust_enough",
                        "PreferredBuyZoneLow": 25.85,
                        "PreferredBuyZoneHigh": 26.4,
                        "ZoneStatus": "inside",
                        "NoChaseAbove": 26.4,
                        "InvalidationBelow": 25.54,
                        "OHLCMultiSessionSummary": "T+1 close -0.63%",
                    }
                ]
            ).to_csv(candidates_dir / "candidate_watchlist_full.csv", index=False)

            report = build_position_ml_review(
                ticker="MBB",
                quantity=40_000,
                avg_price=26.25,
                current_price=26.0,
                universe_csv=out_dir / "universe.csv",
                analysis_dir=analysis_dir,
                output_dir=analysis_dir / "positions",
            )

            self.assertEqual(report["CurrentPnLVND"], -10_000_000)
            self.assertEqual(report["Policy"], "ml_only_no_manual_trim_or_stop_rules")
            self.assertEqual(report["CandidateContext"]["InvalidationBelow"], 25.54)
            self.assertEqual(report["BestTimingModel"]["PredNetEdgePct"], 2.2)
            self.assertEqual(len(report["OHLCForecasts"]), 2)
            self.assertEqual(report["OHLCForecasts"][0]["ForecastClosePnLVND"], -14_000_000)
            self.assertEqual(report["OHLCForecasts"][1]["ForecastClosePnLVND"], 2_000_000)
            scenarios = report["AverageDownScenarios"]
            infeasible_current_t1 = next(
                item
                for item in scenarios
                if item["EntryPrice"] == 26.0 and item["ForecastWindow"] == "T+1" and item["Target"] == "ForecastClose"
            )
            self.assertFalse(infeasible_current_t1["Feasible"])
            zone_low_t3 = next(
                item
                for item in scenarios
                if item["EntryPrice"] == 25.85 and item["ForecastWindow"] == "T+3" and item["Target"] == "ForecastClose"
            )
            self.assertEqual(zone_low_t3["RequiredAdditionalQuantity"], 0)
            self.assertTrue((analysis_dir / "positions" / "mbb_position_ml_review.json").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
