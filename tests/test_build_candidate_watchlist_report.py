from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_candidate_watchlist_report import build_candidate_watchlist


class BuildCandidateWatchlistReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.analysis_dir = self.root / "out" / "analysis"
        (self.analysis_dir / "ticker_playbooks_live").mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / "ml_cycle_forecast").mkdir(parents=True, exist_ok=True)
        self.research_dir = self.root / "research"
        self.output_dir = self.analysis_dir / "candidates"

    def _write_core_inputs(self) -> None:
        universe = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Sector": "Ngân hàng",
                    "Last": 49.8,
                    "RSI14": 57.0,
                    "DistSMA20Pct": 2.5,
                    "Ret20dVsIndex": 2.0,
                    "Ret60dVsIndex": 4.0,
                    "ADTV20_shares": 20_000_000.0,
                    "TickSize": 0.05,
                    "LotSize": 100,
                    "GridBelow_T1": 49.9,
                    "GridBelow_T2": 49.7,
                    "GridBelow_T3": 49.5,
                    "SectorBreadthPositive5dPct": 45.0,
                    "NetBuySellForeign_kVND_20d": 500_000_000.0,
                },
                {
                    "Ticker": "BBB",
                    "Sector": "Bất động sản",
                    "Last": 120.0,
                    "RSI14": 76.0,
                    "DistSMA20Pct": 22.0,
                    "Ret20dVsIndex": 18.0,
                    "Ret60dVsIndex": 12.0,
                    "ADTV20_shares": 200_000.0,
                    "TickSize": 0.1,
                    "LotSize": 100,
                    "GridBelow_T1": 119.8,
                    "GridBelow_T2": 119.5,
                    "GridBelow_T3": 119.0,
                    "SectorBreadthPositive5dPct": 95.0,
                    "NetBuySellForeign_kVND_20d": 1_000_000_000.0,
                },
            ]
        )
        out_dir = self.root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        universe.to_csv(out_dir / "universe.csv", index=False)

        with (out_dir / "market_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "GeneratedAt": "2026-04-20T11:46:07+07:00",
                    "IndexRangePos20": 0.82,
                    "BreadthAboveSMA20Pct": 68.0,
                    "BreadthPositive5dPct": 48.0,
                },
                handle,
            )

        pd.DataFrame(
            [
                {"Sector": "Ngân hàng", "TickerCount": 5},
                {"Sector": "Bất động sản", "TickerCount": 4},
            ]
        ).to_csv(out_dir / "sector_summary.csv", index=False)

        pd.DataFrame(
            [
                {"Ticker": "AAA", "LatestSignal": True, "AllScore": 25.0, "RobustScore": 20.0},
                {"Ticker": "BBB", "LatestSignal": False, "AllScore": 24.0, "RobustScore": -5.0},
            ]
        ).to_csv(self.analysis_dir / "ticker_playbooks_live" / "ticker_playbook_best_configs.csv", index=False)

    def test_build_candidate_watchlist_core_keeps_grid_only_zone_as_wait(self) -> None:
        self._write_core_inputs()

        report = build_candidate_watchlist(
            mode="core",
            budget_vnd=5_000_000_000,
            universe_csv=self.root / "out" / "universe.csv",
            market_summary_json=self.root / "out" / "market_summary.json",
            sector_summary_csv=self.root / "out" / "sector_summary.csv",
            analysis_dir=self.analysis_dir,
            research_dir=self.research_dir,
            output_dir=self.output_dir,
        )

        rows = pd.DataFrame(report["Rows"])
        aaa = rows.loc[rows["Ticker"].eq("AAA")].iloc[0]
        bbb = rows.loc[rows["Ticker"].eq("BBB")].iloc[0]

        self.assertEqual(report["Mode"], "core")
        self.assertEqual(aaa["Decision"], "chờ")
        self.assertEqual(aaa["PreferredBuyZoneSource"], "grid")
        self.assertFalse(bool(aaa["AnchoredBuyZone"]))
        self.assertEqual(bbb["Decision"], "không_mua")
        self.assertTrue((self.output_dir / "candidate_watchlist_core.csv").exists())
        self.assertTrue((self.output_dir / "candidate_watchlist_core.md").exists())

    def test_build_candidate_watchlist_full_uses_research_zone_to_mark_wait(self) -> None:
        self._write_core_inputs()

        pd.DataFrame(
            [
                {"Ticker": "AAA", "ForecastCloseRetPct": 1.2, "ForecastCandleBias": "BULLISH"},
                {"Ticker": "BBB", "ForecastCloseRetPct": -2.0, "ForecastCandleBias": "BEARISH"},
            ]
        ).to_csv(self.analysis_dir / "ml_ohlc_next_session.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Horizon": 3,
                    "ForecastWindow": "T+3",
                    "PredNetEdgePct": 3.0,
                    "PredPeakRetPct": 4.5,
                    "PredPeakDay": 2.0,
                    "PredDrawdownPct": -1.2,
                    "PredCloseRetPct": 1.4,
                    "PredCapitalEfficiencyPctPerDay": 1.0,
                    "EvalRows": 48,
                    "PeakRetMAEPct": 1.1,
                    "DrawdownMAEPct": 0.8,
                    "CloseMAEPct": 0.9,
                    "TradeScoreHitPct": 62.0,
                },
                {
                    "Ticker": "AAA",
                    "Horizon": 10,
                    "ForecastWindow": "T+10",
                    "PredNetEdgePct": 1.0,
                    "PredPeakRetPct": 5.0,
                    "PredPeakDay": 5.0,
                    "PredDrawdownPct": -2.0,
                    "PredCloseRetPct": 2.0,
                    "PredCapitalEfficiencyPctPerDay": 0.5,
                    "EvalRows": 48,
                    "PeakRetMAEPct": 1.4,
                    "DrawdownMAEPct": 1.0,
                    "CloseMAEPct": 1.2,
                    "TradeScoreHitPct": 58.0,
                },
                {
                    "Ticker": "BBB",
                    "Horizon": 3,
                    "ForecastWindow": "T+3",
                    "PredNetEdgePct": -4.0,
                    "PredPeakRetPct": 1.5,
                    "PredPeakDay": 2.0,
                    "PredDrawdownPct": -5.0,
                    "PredCloseRetPct": -2.0,
                    "PredCapitalEfficiencyPctPerDay": -0.7,
                    "EvalRows": 36,
                    "PeakRetMAEPct": 1.6,
                    "DrawdownMAEPct": 1.9,
                    "CloseMAEPct": 1.8,
                    "TradeScoreHitPct": 41.0,
                },
                {
                    "Ticker": "BBB",
                    "Horizon": 10,
                    "ForecastWindow": "T+10",
                    "PredNetEdgePct": -2.0,
                    "PredPeakRetPct": 2.5,
                    "PredPeakDay": 6.0,
                    "PredDrawdownPct": -6.5,
                    "PredCloseRetPct": -1.0,
                    "PredCapitalEfficiencyPctPerDay": -0.2,
                    "EvalRows": 36,
                    "PeakRetMAEPct": 2.1,
                    "DrawdownMAEPct": 2.4,
                    "CloseMAEPct": 2.2,
                    "TradeScoreHitPct": 39.0,
                },
            ]
        ).to_csv(self.analysis_dir / "ml_single_name_timing.csv", index=False)

        pd.DataFrame(
            [
                {"Ticker": "AAA", "EntryScoreRank": 1, "LimitPrice": 48.5, "FillScoreComposite": 60.0, "EntryScore": 3.0},
                {"Ticker": "BBB", "EntryScoreRank": 1, "LimitPrice": 111.0, "FillScoreComposite": 35.0, "EntryScore": 2.0},
            ]
        ).to_csv(self.analysis_dir / "ml_entry_ladder_eval.csv", index=False)

        pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "ForecastWindow": "2M",
                    "PredPeakRetPct": 8.0,
                    "PredPeakDays": 24.0,
                    "PredDrawdownPct": -6.0,
                },
                {
                    "Ticker": "BBB",
                    "ForecastWindow": "2M",
                    "PredPeakRetPct": 5.0,
                    "PredPeakDays": 27.0,
                    "PredDrawdownPct": -12.0,
                },
            ]
        ).to_csv(self.analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv", index=False)

        (self.research_dir / "tickers" / "AAA").mkdir(parents=True, exist_ok=True)
        (self.research_dir / "tickers" / "BBB").mkdir(parents=True, exist_ok=True)
        (self.research_dir / "manifest.json").write_text(json.dumps({"GeneratedAt": "2026-04-20"}), encoding="utf-8")
        (self.research_dir / "tickers" / "AAA" / "state.json").write_text(
            json.dumps(
                {
                    "Ticker": "AAA",
                    "PreferredBuyZoneLow": 47.5,
                    "PreferredBuyZoneHigh": 48.5,
                    "BestTimingWindow": "T+3",
                    "BestTimingNetEdgePct": 3.0,
                    "T10NetEdgePct": 1.0,
                    "SuggestedNewCapitalPct": 4.0,
                    "DeferredBuildPct": 6.0,
                    "PersistentWeaknessBid": True,
                    "SessionBuyPlanSummary": "48.5: core ladder",
                    "SessionBuyTranches": [
                        {"Role": "shallow_core", "LimitPrice": 49.8, "AllocatedCapitalPctOfPortfolio": 2.0},
                        {"Role": "mid_core", "LimitPrice": 48.5, "AllocatedCapitalPctOfPortfolio": 1.5},
                        {"Role": "deep_core", "LimitPrice": 47.5, "AllocatedCapitalPctOfPortfolio": 1.5},
                    ],
                    "DamageBelow": 46.8,
                    "BullishConfirmAbove": 50.5,
                }
            ),
            encoding="utf-8",
        )
        (self.research_dir / "tickers" / "BBB" / "state.json").write_text(
            json.dumps(
                {
                    "Ticker": "BBB",
                    "PreferredBuyZoneLow": 108.0,
                    "PreferredBuyZoneHigh": 111.0,
                    "BestTimingWindow": "T+3",
                    "BestTimingNetEdgePct": -4.0,
                    "T10NetEdgePct": -2.0,
                    "SuggestedNewCapitalPct": 0.5,
                    "DeferredBuildPct": 9.5,
                    "PersistentWeaknessBid": False,
                    "SessionBuyPlanSummary": "111.0: deep only",
                    "SessionBuyTranches": [
                        {"Role": "shallow_core", "LimitPrice": 111.0, "AllocatedCapitalPctOfPortfolio": 0.3},
                        {"Role": "deep_core", "LimitPrice": 108.0, "AllocatedCapitalPctOfPortfolio": 0.2},
                    ],
                    "DamageBelow": 104.0,
                    "BullishConfirmAbove": 121.0,
                }
            ),
            encoding="utf-8",
        )

        report = build_candidate_watchlist(
            mode="full",
            budget_vnd=5_000_000_000,
            universe_csv=self.root / "out" / "universe.csv",
            market_summary_json=self.root / "out" / "market_summary.json",
            sector_summary_csv=self.root / "out" / "sector_summary.csv",
            analysis_dir=self.analysis_dir,
            research_dir=self.research_dir,
            output_dir=self.output_dir,
        )

        rows = pd.DataFrame(report["Rows"])
        aaa = rows.loc[rows["Ticker"].eq("AAA")].iloc[0]
        bbb = rows.loc[rows["Ticker"].eq("BBB")].iloc[0]

        self.assertEqual(report["Mode"], "full")
        self.assertEqual(aaa["Decision"], "chờ")
        self.assertEqual(aaa["PreferredBuyZoneLow"], 47.5)
        self.assertEqual(aaa["PreferredBuyZoneSource"], "state")
        self.assertTrue(bool(aaa["AnchoredBuyZone"]))
        self.assertEqual(aaa["ReferenceBuyPrice"], 47.5)
        self.assertEqual(aaa["BestTimingPeakRetPct"], 4.5)
        self.assertEqual(aaa["BestTimingPeakDay"], 2.0)
        self.assertEqual(aaa["BestCyclePeakRetPct"], 8.0)
        self.assertEqual(aaa["BestCyclePeakDays"], 24.0)
        self.assertIn("peak 4.50% trong ~2.0 phiên", aaa["TimingProfitSummary"])
        self.assertIn("peak 8.00% trong ~24.0 phiên", aaa["CycleProfitSummary"])
        self.assertEqual(aaa["RecommendedDeployPctOfRefBudget"], 4.0)
        self.assertEqual(aaa["InvalidationBelow"], 46.8)
        self.assertEqual(aaa["NoChaseAbove"], 48.5)
        self.assertEqual(aaa["BreakoutConfirmAbove"], 50.5)
        self.assertIn("48 mẫu", aaa["ValidationSummary"])
        self.assertEqual(aaa["ConservativePeakRetPct"], 3.4)
        self.assertEqual(aaa["ConservativeCloseRetPct"], 0.5)
        self.assertEqual(aaa["ReferenceBudgetFullPlanPct"], 100.0)
        self.assertEqual(aaa["ReferenceBudgetDeployNowPct"], 40.0)
        self.assertEqual(aaa["ReferenceBudgetDeployNowVND"], 1_996_980_000)
        self.assertEqual(bbb["Decision"], "không_mua")
        self.assertTrue((self.output_dir / "candidate_watchlist_full.json").exists())
        markdown = (self.output_dir / "candidate_watchlist_full.md").read_text(encoding="utf-8")
        self.assertIn("timing `T+3: peak 4.50% trong ~2.0 phiên, close 1.40%`", markdown)
        self.assertIn("cycle `2M: peak 8.00% trong ~24.0 phiên`", markdown)
        self.assertIn("verify `48 mẫu | hit 62.0% | peak MAE 1.10% | close MAE 0.90%`", markdown)
        self.assertIn("deploy `~4.0%` ref budget", markdown)
        self.assertIn("full-plan `100.0%` ref budget", markdown)
        self.assertIn("now `40.0%` ref budget", markdown)
        self.assertIn("no-chase `>48.5`", markdown)
        self.assertIn("invalid `<46.8`", markdown)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
