from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_ticker_ml_deep_dive import build_deep_dive


class BuildTickerMlDeepDiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.analysis_dir = self.root / "out" / "analysis"
        (self.analysis_dir / "ml_cycle_forecast").mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / "ticker_playbooks_live").mkdir(parents=True, exist_ok=True)
        (self.root / "research" / "tickers" / "ABC").mkdir(parents=True, exist_ok=True)

    def test_build_deep_dive_flags_extended_negative_timing_name_as_no_buy_now(self) -> None:
        universe = pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "Sector": "Bất động sản",
                    "Last": 100.0,
                    "Ret5d": 8.0,
                    "Ret20dVsIndex": 15.0,
                    "Ret60dVsIndex": 20.0,
                    "RSI14": 74.5,
                    "DistSMA20Pct": 18.0,
                    "Pos52wPct": 0.9,
                    "ADTV20_shares": 1_000_000.0,
                    "TickSize": 0.1,
                    "LotSize": 100,
                }
            ]
        )
        universe_path = self.root / "out" / "universe.csv"
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(universe_path, index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "ForecastOpen": 100.5,
                    "ForecastHigh": 101.0,
                    "ForecastLow": 98.0,
                    "ForecastClose": 98.5,
                    "ForecastCloseRetPct": -1.5,
                    "ForecastCandleBias": "BEARISH",
                    "TickerShockState1D": 1.0,
                    "TickerImpulseState3D": 0.0,
                    "TickerWideRangeState": 1.0,
                    "TickerTrendRegimeState": 1.0,
                }
            ]
        ).to_csv(self.analysis_dir / "ml_ohlc_next_session.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "Horizon": 3,
                    "ForecastWindow": "T+3",
                    "EvalRows": 42,
                    "PeakRetMAEPct": 1.0,
                    "PeakDayMAE": 0.7,
                    "DrawdownMAEPct": 1.2,
                    "CloseMAEPct": 1.1,
                    "TradeScoreHitPct": 44.0,
                    "PredPeakRetPct": 1.0,
                    "PredDrawdownPct": -4.0,
                    "PredCloseRetPct": -1.0,
                    "PredPeakPrice": 101.0,
                    "PredDrawdownPrice": 96.0,
                    "PredClosePrice": 99.0,
                    "PredRewardRisk": 0.25,
                    "PredNetEdgePct": -2.0,
                },
                {
                    "Ticker": "ABC",
                    "Horizon": 10,
                    "ForecastWindow": "T+10",
                    "EvalRows": 42,
                    "PeakRetMAEPct": 1.8,
                    "PeakDayMAE": 1.6,
                    "DrawdownMAEPct": 1.9,
                    "CloseMAEPct": 1.7,
                    "TradeScoreHitPct": 39.0,
                    "PredPeakRetPct": 2.0,
                    "PredDrawdownPct": -8.0,
                    "PredCloseRetPct": -3.0,
                    "PredPeakPrice": 102.0,
                    "PredDrawdownPrice": 92.0,
                    "PredClosePrice": 97.0,
                    "PredRewardRisk": 0.25,
                    "PredNetEdgePct": -4.0,
                },
            ]
        ).to_csv(self.analysis_dir / "ml_single_name_timing.csv", index=False)

        range_rows = pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "Horizon": 1,
                    "ForecastWindow": "T+1",
                    "Low": 98.0,
                    "Mid": 100.0,
                    "High": 102.0,
                    "PredLowRetPct": -2.0,
                    "PredMidRetPct": 0.0,
                    "PredHighRetPct": 2.0,
                }
            ]
        )
        range_rows.to_csv(self.analysis_dir / "ml_range_predictions_recent_focus.csv", index=False)
        range_rows.to_csv(self.analysis_dir / "ml_range_predictions_full_2y.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "ForecastWindow": "2M",
                    "PredPeakRetPct": 10.0,
                    "PredPeakDays": 12.0,
                    "PredPeakPrice": 110.0,
                    "PredDrawdownPct": -12.0,
                    "PredDrawdownPrice": 88.0,
                }
            ]
        ).to_csv(self.analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "EntryScoreRank": 1,
                    "LimitPrice": 94.0,
                    "EntryAnchor": "cycle_drawdown",
                    "EntryVsLastPct": -6.0,
                    "NetRetToNextClosePct": 2.0,
                    "NetRetToNextHighPct": 5.0,
                    "BestTimingWindow": "T+3",
                    "BestTimingNetEdgePct": 1.0,
                    "BestTimingRewardRisk": 2.0,
                    "CycleNetEdgePct": 5.0,
                    "CycleRewardRisk": 3.0,
                    "FillScoreComposite": 40.0,
                    "EntryScore": 3.5,
                }
            ]
        ).to_csv(self.analysis_dir / "ml_entry_ladder_eval.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "ABC",
                    "StrategyFamily": "washout_reclaim",
                    "LatestSignal": False,
                    "TestScore": 10.0,
                    "AllScore": 12.0,
                    "RobustScore": 8.0,
                    "TestWinRatePct": 55.0,
                    "TestAvgReturnPct": 3.0,
                    "AllAvgReturnPct": 2.5,
                    "AllWorstDrawdownPct": -9.0,
                }
            ]
        ).to_csv(self.analysis_dir / "ticker_playbooks_live" / "ticker_playbook_best_configs.csv", index=False)

        state = {
            "Ticker": "ABC",
            "Archetype": "momentum_high_beta",
            "PreferredHoldWindow": "3-10 phiên",
            "DailySummary": "daily summary",
            "WeeklySummary": "weekly summary",
            "BestTimingWindow": "T+3",
            "BestTimingNetEdgePct": -2.0,
            "T10NetEdgePct": -4.0,
            "SuggestedNewCapitalPct": 12.5,
            "DeferredBuildPct": 7.5,
            "SessionBuyPlanSummary": "không có session buy plan riêng",
            "SessionBuyTranches": [],
            "PreferredBuyZoneLow": 92.0,
            "PreferredBuyZoneHigh": 94.0,
            "BullishConfirmAbove": 105.0,
            "DamageBelow": 88.0,
            "ExecutionBias": "unknown",
            "BurstExecutionBias": "wait_for_day2_confirmation",
            "ExecutionNote": "state note",
            "TrimAggression": "moderate",
            "MustSellFractionPct": 15.0,
        }
        (self.root / "research" / "tickers" / "ABC" / "state.json").write_text(
            json.dumps(state),
            encoding="utf-8",
        )

        report = build_deep_dive(
            ticker="ABC",
            budget_vnd=5_000_000_000,
            universe_csv=universe_path,
            analysis_dir=self.analysis_dir,
            research_dir=self.root / "research",
            output_dir=self.root / "out" / "deep_dive",
        )

        self.assertEqual(report["Verdict"], "NO_BUY_NOW")
        self.assertTrue(Path(report["OutputJSON"]).exists())
        self.assertTrue(Path(report["OutputMarkdown"]).exists())
        self.assertEqual(report["BudgetPlan"], [])
        self.assertIsNone(report["Snapshot"]["ForeignFlow5dBnVND"])
        self.assertIsNone(report["Snapshot"]["ForeignFlow20dBnVND"])
        self.assertEqual(report["Sizing"]["RecommendedDeployPctOfRefBudget"], 12.5)
        self.assertEqual(report["Sizing"]["NoChaseAbove"], 94.0)
        self.assertEqual(report["Sizing"]["InvalidationBelow"], 88.0)
        self.assertIn("42 mẫu", report["TimingRows"][0]["ValidationSummary"])
        self.assertEqual(report["OHLCStateSignals"]["Summary"], "shock up, wide-range expansion, hot trend regime")
        self.assertEqual(report["OHLCStateSignals"]["TickerShockState1D"], 1.0)
        self.assertIn("SpecializedOverlay", report)
        self.assertEqual(report["State"]["ExecutionNote"], "state note")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
