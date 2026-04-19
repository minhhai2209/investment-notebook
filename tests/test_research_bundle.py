from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ResearchBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_inputs(self) -> dict[str, Path]:
        out_dir = self.root / "out"
        analysis_dir = out_dir / "analysis"
        cycle_dir = analysis_dir / "ml_cycle_forecast"
        playbook_dir = analysis_dir / "ticker_playbooks_live"
        intraday_dir = out_dir / "data" / "intraday_5m"
        for path in [out_dir, analysis_dir, cycle_dir, playbook_dir, intraday_dir]:
            path.mkdir(parents=True, exist_ok=True)

        universe_df = pd.DataFrame(
            [
                {
                    "Ticker": "VNINDEX",
                    "EngineRunAt": "2026-04-12T15:35:00+07:00",
                    "Sector": "",
                    "Last": 1750.0,
                    "Ref": 1740.0,
                    "ChangePct": 0.57,
                    "SMA20": 1700.0,
                    "SMA50": 1680.0,
                    "SMA200": 1600.0,
                    "EMA20": 1705.0,
                    "RSI14": 58.0,
                    "ATR14Pct": 2.1,
                    "Vol20Pct": 1.9,
                    "Vol60Pct": 1.7,
                    "MACD": 10.0,
                    "MACDSignal": 9.5,
                    "Beta60_Index": 1.0,
                    "Corr60_Index": 1.0,
                    "Corr20_Index": 1.0,
                    "Beta20_Index": 1.0,
                    "Ret5d": 2.0,
                    "Ret20d": 3.0,
                    "Ret60d": 5.0,
                    "Ret20dVsIndex": 0.0,
                    "Ret60dVsIndex": 0.0,
                    "Ret20dVsSector": 0.0,
                    "Ret60dVsSector": 0.0,
                    "Pos52wPct": 0.7,
                    "High52w": 1800.0,
                    "Low52w": 1200.0,
                    "ADTV20_shares": 1_000_000.0,
                    "IntradayVol_shares": 1_100_000.0,
                    "IntradayValue_kVND": 1_925_000_000.0,
                    "IntradayPctADV20": 1.1,
                    "ValidBid1": 1749.0,
                    "ValidAsk1": 1750.0,
                    "GridBelow_T1": 1740.0,
                    "GridBelow_T2": 1735.0,
                    "GridBelow_T3": 1730.0,
                    "GridAbove_T1": 1755.0,
                    "GridAbove_T2": 1760.0,
                    "GridAbove_T3": 1765.0,
                    "DistSMA20Pct": 2.94,
                    "PE_fwd": 0.0,
                    "PB": 0.0,
                    "ROE": 0.0,
                    "PositionMarketValue_kVND": 0.0,
                    "PositionWeightPct": 0.0,
                    "EnginePortfolioMarketValue_kVND": 168000.0,
                },
                {
                    "Ticker": "NVL",
                    "EngineRunAt": "2026-04-12T15:35:00+07:00",
                    "Sector": "Bất động sản",
                    "Last": 16.8,
                    "Ref": 16.8,
                    "ChangePct": 0.0,
                    "SMA20": 14.21,
                    "SMA50": 12.93,
                    "SMA200": 14.62,
                    "EMA20": 14.36,
                    "RSI14": 75.33,
                    "ATR14Pct": 4.67,
                    "Vol20Pct": 3.86,
                    "Vol60Pct": 3.43,
                    "MACD": 0.91,
                    "MACDSignal": 0.63,
                    "Beta60_Index": 0.78,
                    "Corr60_Index": 0.40,
                    "Corr20_Index": 0.48,
                    "Beta20_Index": 1.01,
                    "Ret5d": 17.48,
                    "Ret20d": 32.81,
                    "Ret60d": 33.86,
                    "Ret20dVsIndex": 29.64,
                    "Ret60dVsIndex": 40.18,
                    "Ret20dVsSector": 0.0,
                    "Ret60dVsSector": 0.0,
                    "Pos52wPct": 0.78,
                    "High52w": 19.3,
                    "Low52w": 7.88,
                    "ADTV20_shares": 25_165_250.0,
                    "IntradayVol_shares": 27_600_600.0,
                    "IntradayValue_kVND": 468_030_025.0,
                    "IntradayPctADV20": 1.10,
                    "ValidBid1": 16.75,
                    "ValidAsk1": 16.85,
                    "GridBelow_T1": 16.75,
                    "GridBelow_T2": 16.70,
                    "GridBelow_T3": 16.65,
                    "GridAbove_T1": 16.85,
                    "GridAbove_T2": 16.90,
                    "GridAbove_T3": 16.95,
                    "DistSMA20Pct": 18.21,
                    "PE_fwd": 26.89,
                    "PB": 0.45,
                    "ROE": 3.00,
                    "PositionMarketValue_kVND": 168000.0,
                    "PositionWeightPct": 100.0,
                    "EnginePortfolioMarketValue_kVND": 168000.0,
                },
            ]
        )
        universe_path = out_dir / "universe.csv"
        universe_df.to_csv(universe_path, index=False)

        market_summary = {
            "BreadthAboveSMA20Pct": 100.0,
            "BreadthAboveSMA50Pct": 90.0,
            "BreadthPositive5dPct": 100.0,
            "NewHigh20Pct": 35.0,
            "AdvanceDeclineRatio": 4.0,
            "VNINDEX_ATR14PctRank": 80.0,
        }
        market_summary_path = out_dir / "market_summary.json"
        market_summary_path.write_text(json.dumps(market_summary), encoding="utf-8")

        sector_df = pd.DataFrame(
            [
                {
                    "Sector": "Bất động sản",
                    "SectorBreadthAboveSMA20Pct": 100.0,
                    "SectorBreadthAboveSMA50Pct": 100.0,
                    "SectorBreadthPositive5dPct": 100.0,
                    "SectorMedianRet20dVsIndex": 29.64,
                    "SectorMedianRet60dVsIndex": 40.18,
                }
            ]
        )
        sector_path = out_dir / "sector_summary.csv"
        sector_df.to_csv(sector_path, index=False)

        timing_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "ForecastWindow": "T+3",
                    "Horizon": 3,
                    "PredPeakRetPct": 3.87,
                    "PredPeakDay": 2.5,
                    "PredDrawdownPct": -3.62,
                    "PredCloseRetPct": 1.53,
                    "PredRewardRisk": 1.07,
                    "PredTradeScore": 0.25,
                    "PredNetEdgePct": 0.19,
                    "PredCapitalEfficiencyPctPerDay": 0.08,
                },
                {
                    "Ticker": "NVL",
                    "ForecastWindow": "T+5",
                    "Horizon": 5,
                    "PredPeakRetPct": 5.30,
                    "PredPeakDay": 3.8,
                    "PredDrawdownPct": -3.91,
                    "PredCloseRetPct": 1.76,
                    "PredRewardRisk": 1.35,
                    "PredTradeScore": 1.39,
                    "PredNetEdgePct": 1.33,
                    "PredCapitalEfficiencyPctPerDay": 0.35,
                },
                {
                    "Ticker": "NVL",
                    "ForecastWindow": "T+10",
                    "Horizon": 10,
                    "PredPeakRetPct": 4.23,
                    "PredPeakDay": 3.1,
                    "PredDrawdownPct": -9.75,
                    "PredCloseRetPct": -3.43,
                    "PredRewardRisk": 0.43,
                    "PredTradeScore": -5.52,
                    "PredNetEdgePct": -5.58,
                    "PredCapitalEfficiencyPctPerDay": -1.80,
                },
            ]
        )
        timing_df.to_csv(analysis_dir / "ml_single_name_timing.csv", index=False)

        ladder_df = pd.DataFrame(
            [
                {"Ticker": "NVL", "EntryScoreRank": 1, "PriceRank": 8, "LimitPrice": 15.65, "EntryScore": 6.7, "BestTimingWindow": "T+3", "BestTimingNetEdgePct": 11.45, "BestTimingCloseRetPct": 8.93, "BestTimingRewardRisk": 3.33, "CycleNetEdgePct": 21.10, "CycleRewardRisk": 7.48, "FillScoreComposite": 22.03},
                {"Ticker": "NVL", "EntryScoreRank": 2, "PriceRank": 7, "LimitPrice": 16.00, "EntryScore": 6.6, "BestTimingWindow": "T+3", "BestTimingNetEdgePct": 9.01, "BestTimingCloseRetPct": 6.55, "BestTimingRewardRisk": 7.59, "CycleNetEdgePct": 18.45, "CycleRewardRisk": 31.94, "FillScoreComposite": 37.74},
                {"Ticker": "NVL", "EntryScoreRank": 3, "PriceRank": 6, "LimitPrice": 16.05, "EntryScore": 6.5, "BestTimingWindow": "T+3", "BestTimingNetEdgePct": 8.67, "BestTimingCloseRetPct": 6.21, "BestTimingRewardRisk": 9.92, "CycleNetEdgePct": 18.08, "CycleRewardRisk": 68.14, "FillScoreComposite": 39.89},
            ]
        )
        ladder_df.to_csv(analysis_dir / "ml_entry_ladder_eval.csv", index=False)

        ohlc_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "ForecastOpen": 16.90,
                    "ForecastHigh": 17.16,
                    "ForecastLow": 16.67,
                    "ForecastClose": 16.90,
                    "ForecastCloseRetPct": 0.61,
                    "ForecastRangePct": 2.92,
                    "ForecastCandleBias": "BEARISH",
                }
            ]
        )
        ohlc_df.to_csv(analysis_dir / "ml_ohlc_next_session.csv", index=False)

        playbook_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "StrategyFamily": "breakout_followthrough",
                    "StrategyLabel": "breakout_followthrough/...",
                    "LatestSignal": False,
                    "LatestSignalDate": "2026-04-11",
                    "TestScore": 13.7,
                    "RobustScore": 16.8,
                    "TestAvgReturnPct": 1.98,
                    "TestWorstDrawdownPct": -9.27,
                }
            ]
        )
        playbook_df.to_csv(playbook_dir / "ticker_playbook_best_configs.csv", index=False)

        cycle_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "HorizonMonths": 1,
                    "ForecastWindow": "1M",
                    "Variant": "recent_focus",
                    "Model": "ridge",
                    "SelectionScore": 10.9,
                    "PredPeakRetPct": 12.87,
                    "PredPeakDays": 8.6,
                    "PredPeakPrice": 18.96,
                    "PredDrawdownPct": -4.21,
                    "PredDrawdownPrice": 16.09,
                }
            ]
        )
        cycle_df.to_csv(cycle_dir / "cycle_forecast_best_horizon_by_ticker.csv", index=False)

        range_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "ForecastWindow": "T+5",
                    "PredLowRetPct": -1.04,
                    "PredMidRetPct": 0.86,
                    "PredHighRetPct": 2.54,
                    "CloseMAEPct": 8.25,
                    "RangeMAEPct": 2.64,
                    "CloseDirHitPct": 73.33,
                }
            ]
        )
        range_df.to_csv(analysis_dir / "ml_range_predictions_full_2y.csv", index=False)
        range_df.to_csv(analysis_dir / "ml_range_predictions_recent_focus.csv", index=False)

        intraday_report_df = pd.DataFrame(
            [
                {
                    "Ticker": "NVL",
                    "SnapshotTimeBucket": "PM_LATE",
                    "PredLowRetPct": -0.46,
                    "PredMidRetPct": -0.31,
                    "PredHighRetPct": -0.03,
                    "SelectionScore": 1.2,
                }
            ]
        )
        intraday_report_df.to_csv(analysis_dir / "ml_intraday_rest_of_session.csv", index=False)

        intraday_nvl = pd.DataFrame(
            [
                {"t": 1775793300, "open": 16.95, "high": 17.45, "low": 16.90, "close": 17.30, "volume": 3_973_900, "date_vn": "2026-04-11"},
                {"t": 1775795100, "open": 17.10, "high": 17.10, "low": 16.70, "close": 16.70, "volume": 4_439_300, "date_vn": "2026-04-11"},
                {"t": 1775797800, "open": 16.90, "high": 17.00, "low": 16.85, "close": 17.00, "volume": 5_099_600, "date_vn": "2026-04-11"},
                {"t": 1775799600, "open": 16.85, "high": 16.85, "low": 16.75, "close": 16.80, "volume": 1_091_000, "date_vn": "2026-04-11"},
                {"t": 1775801700, "open": 16.80, "high": 17.15, "low": 16.60, "close": 16.85, "volume": 4_577_200, "date_vn": "2026-04-11"},
                {"t": 1775804400, "open": 16.85, "high": 16.90, "low": 16.80, "close": 16.90, "volume": 4_663_500, "date_vn": "2026-04-11"},
                {"t": 1775805900, "open": 16.80, "high": 16.80, "low": 16.70, "close": 16.70, "volume": 1_820_400, "date_vn": "2026-04-11"},
                {"t": 1775807100, "open": 16.80, "high": 16.80, "low": 16.80, "close": 16.80, "volume": 1_940_700, "date_vn": "2026-04-11"},
            ]
        )
        intraday_nvl.to_csv(intraday_dir / "NVL_5m.csv", index=False)

        intraday_index = pd.DataFrame(
            [
                {"t": 1775793300, "open": 1736.68, "high": 1756.32, "low": 1736.68, "close": 1756.32, "volume": 30_718_720, "date_vn": "2026-04-11"},
                {"t": 1775795100, "open": 1756.00, "high": 1761.02, "low": 1755.80, "close": 1761.02, "volume": 135_402_173, "date_vn": "2026-04-11"},
                {"t": 1775797800, "open": 1757.00, "high": 1758.00, "low": 1755.33, "close": 1755.33, "volume": 134_558_879, "date_vn": "2026-04-11"},
                {"t": 1775799600, "open": 1755.40, "high": 1757.68, "low": 1755.30, "close": 1757.68, "volume": 77_386_439, "date_vn": "2026-04-11"},
                {"t": 1775801700, "open": 1756.00, "high": 1756.10, "low": 1754.69, "close": 1754.69, "volume": 144_619_853, "date_vn": "2026-04-11"},
                {"t": 1775804400, "open": 1756.70, "high": 1756.96, "low": 1755.50, "close": 1755.60, "volume": 150_677_297, "date_vn": "2026-04-11"},
                {"t": 1775805900, "open": 1752.11, "high": 1752.22, "low": 1748.96, "close": 1749.77, "volume": 119_005_855, "date_vn": "2026-04-11"},
                {"t": 1775807100, "open": 1749.73, "high": 1750.00, "low": 1749.73, "close": 1750.00, "volume": 63_156_063, "date_vn": "2026-04-11"},
            ]
        )
        intraday_index.to_csv(intraday_dir / "VNINDEX_5m.csv", index=False)

        return {
            "universe_csv": universe_path,
            "market_summary_json": market_summary_path,
            "sector_summary_csv": sector_path,
            "analysis_dir": analysis_dir,
            "intraday_dir": intraday_dir,
            "research_dir": self.root / "research",
            "total_capital_kvnd": 500_000,
        }

    def test_build_research_bundle_creates_structured_artifacts(self) -> None:
        from scripts.research.build_research_bundle import build_research_bundle

        paths = self._write_inputs()
        manifest = build_research_bundle(**paths)

        self.assertEqual(manifest["UniverseTickers"], ["NVL"])
        self.assertEqual(manifest["TotalCapitalKVND"], 500_000)
        self.assertEqual(manifest["Tickers"]["NVL"]["Archetype"], "special_situation")
        self.assertIn("PortfolioAllocator", manifest)
        self.assertGreater(manifest["PortfolioAllocator"]["TargetInvestedPct"], 0)
        self.assertTrue(manifest["PortfolioAllocator"]["SingleNameMode"])
        self.assertIn("NVL", manifest["PortfolioAllocator"]["Tickers"])
        self.assertAlmostEqual(manifest["PortfolioAllocator"]["CurrentEquityValueKVND"], 168000.0)
        self.assertAlmostEqual(manifest["PortfolioAllocator"]["CurrentInvestedPct"], 33.6)
        self.assertAlmostEqual(manifest["PortfolioAllocator"]["DeployableGapPct"], 31.4)
        self.assertAlmostEqual(manifest["PortfolioAllocator"]["SessionBuildCapPct"], 12.56)
        self.assertAlmostEqual(manifest["PortfolioAllocator"]["DeferredBuildPct"], 18.84)

        research_dir = paths["research_dir"]
        self.assertTrue((research_dir / "manifest.json").exists())
        self.assertTrue((research_dir / "tickers" / "NVL" / "profile.md").exists())
        self.assertTrue((research_dir / "tickers" / "NVL" / "weekly" / "2026-W15.md").exists())
        self.assertTrue((research_dir / "tickers" / "NVL" / "daily" / "2026-04-12.md").exists())
        state = json.loads((research_dir / "tickers" / "NVL" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["Archetype"], "special_situation")
        self.assertFalse(state["DefaultAddOnStrengthAllowed"])
        self.assertFalse(state["AddOnStrengthAllowed"])
        self.assertTrue(state["AddOnWeaknessAllowed"])
        self.assertTrue(state["PortfolioSingleNameMode"])
        self.assertAlmostEqual(state["PortfolioCurrentInvestedPct"], 33.6)
        self.assertAlmostEqual(state["PortfolioDeployableGapPct"], 31.4)
        self.assertAlmostEqual(state["PortfolioSessionBuildCapPct"], 12.56)
        self.assertAlmostEqual(state["PortfolioDeferredBuildPct"], 18.84)
        self.assertAlmostEqual(state["CurrentWeightPct"], 33.6)
        self.assertAlmostEqual(state["PreferredBuyZoneLow"], 15.65)
        self.assertIn("TargetWeightMinPct", state)
        self.assertIn("TargetWeightMaxPct", state)
        self.assertIn("SuggestedNewCapitalPct", state)
        self.assertIn("StrengthReservePct", state)
        self.assertTrue(state["OpeningSqueezeFailure"])
        self.assertEqual(state["BurstExecutionBias"], "failed_followthrough")
        self.assertEqual(state["TrimAggression"], "high")
        self.assertEqual(state["UrgentTrimMode"], "none")
        self.assertEqual(state["MustSellFractionPct"], 0.0)
        self.assertGreaterEqual(state["TargetWeightMinPct"], 65.0)
        self.assertGreaterEqual(state["TargetWeightMaxPct"], state["TargetWeightMinPct"])
        self.assertAlmostEqual(state["SuggestedNewCapitalPct"], 12.56)
        self.assertAlmostEqual(state["DeferredBuildPct"], 18.84)
        self.assertEqual(len(state["SessionBuyTranches"]), 3)
        self.assertAlmostEqual(sum(item["BudgetSharePctOfSession"] for item in state["SessionBuyTranches"]), 100.0, places=2)
        self.assertEqual(state["SessionBuyTranches"][0]["Role"], "shallow_core")
        self.assertEqual(state["SessionBuyTranches"][-1]["Role"], "deep_core")
        self.assertGreater(state["SessionBuyTranches"][-1]["BudgetSharePctOfSession"], state["SessionBuyTranches"][0]["BudgetSharePctOfSession"])
        self.assertIn("timing tốt nhất là T+5", state["WeeklySummary"])
        daily_note = (research_dir / "tickers" / "NVL" / "daily" / "2026-04-12.md").read_text(encoding="utf-8")
        self.assertIn("Hướng dẫn ra lệnh hôm nay", daily_note)
        self.assertIn("T+10 đã âm", daily_note)
        self.assertIn("Kỷ luật size", daily_note)
        self.assertIn("Execution bias", daily_note)
        self.assertIn("Session buy plan", daily_note)
        self.assertIn("SELL", daily_note)
        ticker_manifest = manifest["Tickers"]["NVL"]
        self.assertFalse(ticker_manifest["DefaultAllowAddOnStrength"])
        self.assertFalse(ticker_manifest["AllowAddOnStrength"])
        self.assertTrue(ticker_manifest["AllowAddOnWeakness"])
        self.assertTrue(ticker_manifest["PortfolioSingleNameMode"])
        self.assertAlmostEqual(ticker_manifest["SuggestedNewCapitalPct"], 12.56)
        self.assertAlmostEqual(ticker_manifest["DeferredBuildPct"], 18.84)
        self.assertEqual(ticker_manifest["BurstExecutionBias"], "failed_followthrough")
        self.assertEqual(ticker_manifest["UrgentTrimMode"], "none")
        self.assertEqual(ticker_manifest["MustSellFractionPct"], 0.0)

    def test_build_research_bundle_fails_on_missing_required_universe_columns(self) -> None:
        from scripts.research.build_research_bundle import build_research_bundle

        paths = self._write_inputs()
        broken_universe = pd.read_csv(paths["universe_csv"]).drop(columns=["DistSMA20Pct"])
        broken_universe.to_csv(paths["universe_csv"], index=False)

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_research_bundle(**paths)

    def test_build_research_bundle_prunes_stale_ticker_directories(self) -> None:
        from scripts.research.build_research_bundle import build_research_bundle

        paths = self._write_inputs()
        stale_dir = paths["research_dir"] / "tickers" / "HPG"
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "profile.md").write_text("stale", encoding="utf-8")

        build_research_bundle(**paths)

        self.assertFalse(stale_dir.exists())
        self.assertTrue((paths["research_dir"] / "tickers" / "NVL").exists())

    def test_build_research_bundle_rewrites_stale_markdown_artifacts(self) -> None:
        from scripts.research.build_research_bundle import build_research_bundle

        paths = self._write_inputs()
        ticker_dir = paths["research_dir"] / "tickers" / "NVL"
        weekly_path = ticker_dir / "weekly" / "2026-W15.md"
        daily_path = ticker_dir / "daily" / "2026-04-12.md"
        weekly_path.parent.mkdir(parents=True, exist_ok=True)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        weekly_path.write_text("# Weekly research NVL - 2026-W15\n\nSnapshot gốc: `2026-04-01T15:35:00+07:00`\n", encoding="utf-8")
        daily_path.write_text("# Daily research NVL\n\nSnapshot gốc: `2026-04-01T15:35:00+07:00`\n", encoding="utf-8")

        build_research_bundle(**paths)

        weekly_text = weekly_path.read_text(encoding="utf-8")
        daily_text = daily_path.read_text(encoding="utf-8")
        self.assertIn("Snapshot gốc: `2026-04-12T15:35:00+07:00`", weekly_text)
        self.assertIn("Snapshot gốc: `2026-04-12T15:35:00+07:00`", daily_text)
        self.assertIn("nguồn breadth", weekly_text)

    def test_human_notes_can_force_persistent_weakness_bid_and_wider_band(self) -> None:
        from scripts.research.build_research_bundle import build_research_bundle

        paths = self._write_inputs()
        daily_path = paths["intraday_dir"].parent / "NVL_daily.csv"
        rows = []
        close = 10.0
        for idx in range(30):
            if idx == 26:
                close = 14.5
            elif idx == 27:
                close = 15.5
            elif idx == 28:
                close = 16.7
            elif idx == 29:
                close = 16.8
            else:
                close += 0.05
            rows.append(
                {
                    "t": 1773000000 + idx * 86400,
                    "open": round(close * 0.99, 2),
                    "high": round(close * 1.02, 2),
                    "low": round(close * 0.97, 2),
                    "close": round(close, 2),
                    "volume": 10_000_000 + idx * 1000,
                    "date_vn": f"2026-03-{idx + 1:02d}" if idx < 30 else f"2026-04-{idx - 29:02d}",
                }
            )
        pd.DataFrame(rows).to_csv(daily_path, index=False)

        human_notes_path = self.root / "human_notes.md"
        human_notes_path.write_text(
            "# Human Notes\n\n- `NVL`: target giá `34` trong năm 2026 theo tính toán và thông tin riêng của tôi.\n",
            encoding="utf-8",
        )
        paths["total_capital_kvnd"] = 2_000_000

        manifest = build_research_bundle(**paths, human_notes_path=human_notes_path)

        state = json.loads((paths["research_dir"] / "tickers" / "NVL" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["HumanTargetPrice"], 34.0)
        self.assertEqual(state["HumanTargetYear"], 2026)
        self.assertTrue(state["PersistentWeaknessBid"])
        self.assertTrue(state["AddOnWeaknessAllowed"])
        self.assertTrue(state["PortfolioSingleNameMode"])
        self.assertGreaterEqual(state["TargetWeightMinPct"], 65.0)
        self.assertGreaterEqual(state["TargetWeightMaxPct"], 90.0)
        self.assertAlmostEqual(state["PortfolioSessionBuildCapPct"], 22.5)
        self.assertAlmostEqual(state["SuggestedNewCapitalPct"], 22.5)
        self.assertGreater(state["DeferredBuildPct"], 30.0)
        self.assertIsNotNone(state["BurstSampleCount"])
        self.assertEqual(state["BurstExecutionBias"], "failed_day2_followthrough")
        self.assertEqual(state["TrimAggression"], "high")
        self.assertEqual(state["UrgentTrimMode"], "frontload_sell")
        self.assertGreaterEqual(state["MustSellFractionPct"], 30.0)
        self.assertIn("Human note", (paths["research_dir"] / "tickers" / "NVL" / "daily" / "2026-04-12.md").read_text(encoding="utf-8"))
        self.assertEqual(manifest["Tickers"]["NVL"]["HumanTargetPrice"], 34.0)
        self.assertTrue(manifest["Tickers"]["NVL"]["PersistentWeaknessBid"])
        self.assertTrue(manifest["Tickers"]["NVL"]["PortfolioSingleNameMode"])
        self.assertAlmostEqual(manifest["Tickers"]["NVL"]["SuggestedNewCapitalPct"], 22.5)
        self.assertGreater(manifest["Tickers"]["NVL"]["DeferredBuildPct"], 30.0)
        self.assertEqual(manifest["Tickers"]["NVL"]["UrgentTrimMode"], "frontload_sell")
        tranches = manifest["Tickers"]["NVL"]["SessionBuyTranches"]
        self.assertEqual(len(tranches), 4)
        self.assertAlmostEqual(sum(item["BudgetSharePctOfSession"] for item in tranches), 100.0, places=2)
        self.assertEqual(tranches[0]["Role"], "continuation_reserve")
        self.assertNotIn("bridge", [item["Role"] for item in tranches])
        self.assertEqual(tranches[-1]["Role"], "deep_core")

    def test_allocate_global_buy_tranches_uses_ranked_tranches_not_fixed_ticker_quotas(self) -> None:
        from scripts.research.build_research_bundle import _allocate_global_buy_tranches

        ticker_context = {
            "NVL": {
                "row": {"Last": 16.8},
                "ticker_ladder": pd.DataFrame(
                    [
                        {"EntryScoreRank": 1, "PriceRank": 8, "LimitPrice": 15.65, "EntryScore": 6.7, "FillScoreComposite": 22.03},
                        {"EntryScoreRank": 2, "PriceRank": 7, "LimitPrice": 16.00, "EntryScore": 6.6, "FillScoreComposite": 37.74},
                        {"EntryScoreRank": 3, "PriceRank": 6, "LimitPrice": 16.05, "EntryScore": 6.5, "FillScoreComposite": 39.89},
                        {"EntryScoreRank": 4, "PriceRank": 5, "LimitPrice": 16.50, "EntryScore": 4.7, "FillScoreComposite": 64.37},
                    ]
                ),
                "burst_summary": {"latest_signal_age": 1},
                "execution_guidance": {"burst_execution_bias": "failed_day2_followthrough", "trim_aggression": "high"},
                "conviction_score": 1.15,
                "timing_summary": {"best_net_edge": 1.3, "t10_net_edge": -5.5},
                "tape_summary": {"execution_bias": "neutral"},
            },
            "MBB": {
                "row": {"Last": 26.8},
                "ticker_ladder": pd.DataFrame(
                    [
                        {"EntryScoreRank": 1, "PriceRank": 5, "LimitPrice": 25.60, "EntryScore": 6.8, "FillScoreComposite": 58.0},
                        {"EntryScoreRank": 2, "PriceRank": 4, "LimitPrice": 25.65, "EntryScore": 6.7, "FillScoreComposite": 57.0},
                        {"EntryScoreRank": 3, "PriceRank": 3, "LimitPrice": 26.00, "EntryScore": 6.2, "FillScoreComposite": 48.0},
                        {"EntryScoreRank": 4, "PriceRank": 2, "LimitPrice": 26.55, "EntryScore": 5.6, "FillScoreComposite": 69.0},
                    ]
                ),
                "burst_summary": {"latest_signal_age": None},
                "execution_guidance": {"burst_execution_bias": "normal_tactical_management", "trim_aggression": "moderate"},
                "conviction_score": 1.35,
                "timing_summary": {"best_net_edge": 2.2, "t10_net_edge": 1.4},
                "tape_summary": {"execution_bias": "absorption"},
            },
        }
        ticker_targets = {
            "NVL": {
                "GapToMinWeightPct": 18.0,
                "GapToMaxWeightPct": 32.0,
                "AddOnWeaknessAllowed": True,
                "AddOnStrengthAllowed": False,
                "PersistentWeaknessBid": True,
            },
            "MBB": {
                "GapToMinWeightPct": 10.0,
                "GapToMaxWeightPct": 20.0,
                "AddOnWeaknessAllowed": True,
                "AddOnStrengthAllowed": True,
                "PersistentWeaknessBid": False,
            },
        }

        result = _allocate_global_buy_tranches(
            tickers=["NVL", "MBB"],
            ticker_context=ticker_context,
            ticker_targets=ticker_targets,
            session_budget_pct=20.0,
            strength_share=0.35,
        )

        ranked = result["RankedTranches"]
        self.assertTrue(ranked)
        self.assertLessEqual(sum(item["AllocatedCapitalPctOfPortfolio"] for item in ranked), 20.01)
        self.assertTrue(any(item["Ticker"] == "NVL" and item["Role"] == "deep_core" and item["AllocatedCapitalPctOfPortfolio"] >= 1.0 for item in ranked))
        self.assertTrue(any(item["Ticker"] == "MBB" for item in ranked))
        self.assertGreater(len(result["PerTickerTranches"]["NVL"]), 0)
        self.assertGreater(len(result["PerTickerTranches"]["MBB"]), 0)


if __name__ == "__main__":
    unittest.main()
