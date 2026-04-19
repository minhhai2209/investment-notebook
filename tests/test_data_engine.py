from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.engine.data_engine import DataEngine, EngineConfig, MarketDataService


class FakeMarketDataService(MarketDataService):
    def __init__(self, history: pd.DataFrame, intraday: pd.DataFrame) -> None:
        self._history = history
        self._intraday = intraday

    def load_history(self, tickers):
        return self._history

    def load_intraday(self, tickers):
        return self._intraday


class DataEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def _write_config(
        self,
        industry_rows: list[dict[str, object]] | None = None,
        portfolio_csv_text: str | None = None,
        universe_tickers: list[str] | None = None,
        core_tickers: list[str] | None = None,
        preferred_tickers: list[str] | None = None,
    ) -> Path:
        config_path = self.base / "config.yaml"
        industry_csv = self.base / "industry.csv"
        industry_df = pd.DataFrame(
            industry_rows
            or [
                {"Ticker": "AAA", "Sector": "Tech"},
                {"Ticker": "BBB", "Sector": "Finance"},
            ]
        )
        industry_df.to_csv(industry_csv, index=False)
        portfolio_dir = self.base / "pf"
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        cafef_cache_dir = self.base / "cafef_cache"
        (portfolio_dir / "portfolio.csv").write_text(
            portfolio_csv_text or "Ticker,Quantity,AvgPrice\nAAA,10,12\nBBB,5,20\n",
            encoding="utf-8",
        )
        config_path.write_text(
            """
            universe:
              csv: {industry_csv}
              include_indices: true
{core_tickers_block}
{preferred_tickers_block}
{universe_tickers_block}
            technical_indicators:
              moving_averages: [2]
              rsi_periods: [2]
              atr_periods: [2]
              macd:
                fast: 2
                slow: 3
                signal: 2
            portfolio:
              directory: {portfolio_dir}
            output:
              base_dir: {out_dir}
              presets_dir: .
              portfolios_dir: .
              diagnostics_dir: .
            execution:
              aggressiveness: med
              max_order_pct_adv: 0.1
              slice_adv_ratio: 0.25
              min_lot: 100
              max_qty_per_order: 500000
            data:
              history_cache: {cache_dir}
              history_min_days: 1
              intraday_window_minutes: 60
              cafef_flow_enabled: false
              cafef_flow_cache: {cafef_cache}
              cafef_flow_max_age_hours: 12
              vietstock_overview_enabled: false
              vietstock_overview_cache: {cache_dir}/vietstock_overview
              vietstock_overview_max_age_hours: 24
            """.format(
                industry_csv=industry_csv,
                core_tickers_block=(
                    "              core_tickers:\n"
                    + "".join(f"                - {ticker}\n" for ticker in core_tickers)
                    if core_tickers
                    else ""
                ),
                preferred_tickers_block=(
                    "              preferred_tickers:\n"
                    + "".join(f"                - {ticker}\n" for ticker in preferred_tickers)
                    if preferred_tickers
                    else ""
                ),
                universe_tickers_block=(
                    "              tickers:\n"
                    + "".join(f"                - {ticker}\n" for ticker in universe_tickers)
                    if universe_tickers
                    else ""
                ),
                portfolio_dir=portfolio_dir,
                out_dir=self.base / "out",
                cache_dir=self.base / "cache",
                cafef_cache=cafef_cache_dir,
            ),
            encoding="utf-8",
        )
        return config_path

    def test_engine_generates_outputs(self):
        history_df = pd.DataFrame(
            {
                "Date": ["2024-07-01", "2024-07-02", "2024-07-01", "2024-07-02"],
                "Ticker": ["AAA", "AAA", "BBB", "BBB"],
                "Open": [10, 11, 19, 21],
                "High": [11, 12, 21, 22],
                "Low": [9, 10, 18, 19],
                "Close": [11, 12, 20, 21],
                "Volume": [1000, 1200, 800, 900],
                "t": [1, 2, 1, 2],
            }
        )
        intraday_df = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "Ts": [3, 3],
                "Price": [12.5, 21.5],
                "RSI14": [55, 60],
                "TimeVN": ["2024-07-02 14:30:00", "2024-07-02 14:30:00"],
            }
        )
        config_path = self._write_config()
        config = EngineConfig.from_yaml(config_path)
        config.cafef_flow_enabled = True
        config.cafef_flow_cache_dir = self.base / "cafef_cache"
        config.cafef_flow_cache_dir.mkdir(parents=True, exist_ok=True)
        engine = DataEngine(
            config,
            FakeMarketDataService(history_df, intraday_df),
            vn30_fetcher=lambda: set(),
        )
        fake_flow_df = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "ForeignFlowDate": "2024-07-02",
                    "ForeignRoomRemaining_shares": 1000.0,
                    "ForeignHoldingPct": 25.0,
                    "NetBuySellForeign_shares_5d": 100.0,
                    "NetBuySellForeign_kVND_5d": 50.0,
                    "NetBuySellProprietary_shares_1d": -20.0,
                },
                {
                    "Ticker": "BBB",
                    "ForeignFlowDate": "2024-07-02",
                    "ForeignRoomRemaining_shares": 2000.0,
                    "ForeignHoldingPct": 30.0,
                    "NetBuySellForeign_shares_5d": -80.0,
                    "NetBuySellForeign_kVND_5d": -40.0,
                    "NetBuySellProprietary_shares_1d": 10.0,
                },
            ]
        )
        with mock.patch(
            "scripts.engine.data_engine.build_flow_feature_frame", return_value=fake_flow_df
        ):
            summary = engine.run()

        out_dir = config.output_base_dir
        universe_path = out_dir / "universe.csv"
        positions_path = out_dir / "positions.csv"
        market_summary_path = out_dir / "market_summary.json"
        sector_summary_path = out_dir / "sector_summary.csv"
        for path in [
            universe_path,
            positions_path,
            market_summary_path,
            sector_summary_path,
        ]:
            self.assertTrue(path.exists(), f"Missing output: {path}")

        universe_df = pd.read_csv(universe_path)
        market_summary = json.loads(market_summary_path.read_text(encoding="utf-8"))
        sector_summary_df = pd.read_csv(sector_summary_path)
        self.assertIn("Ceil", universe_df.columns)
        self.assertIn("LotSize", universe_df.columns)
        self.assertIn("GridBelow_T1", universe_df.columns)
        self.assertIn("Sector", universe_df.columns)
        self.assertIn("EngineRunAt", universe_df.columns)
        self.assertIn("PositionQuantity", universe_df.columns)
        self.assertIn("PositionPNLPct", universe_df.columns)
        self.assertNotIn("CurrentQty", universe_df.columns)
        self.assertNotIn("LiquidityScore_kVND", universe_df.columns)
        self.assertIn("OneLotATR_kVND", universe_df.columns)
        self.assertIn("SlippageOneTickPct", universe_df.columns)
        self.assertIn("IntradayVol_shares", universe_df.columns)
        self.assertIn("IntradayValue_kVND", universe_df.columns)
        self.assertIn("IntradayPctADV20", universe_df.columns)
        self.assertIn("NetBuySellForeign_shares_5d", universe_df.columns)
        self.assertIn("NetBuySellForeign_kVND_5d", universe_df.columns)
        self.assertIn("NetBuySellProprietary_shares_1d", universe_df.columns)
        self.assertIn("ForeignRoomRemaining_shares", universe_df.columns)
        self.assertIn("ForeignHoldingPct", universe_df.columns)
        self.assertIn("Pos52wPct", universe_df.columns)
        self.assertIn("ADTV20Rank", universe_df.columns)
        self.assertIn("PositionPctADV20", universe_df.columns)
        self.assertIn("PositionATR_kVND", universe_df.columns)
        self.assertIn("EnginePortfolioMarketValue_kVND", universe_df.columns)
        self.assertIn("PositionWeightPct", universe_df.columns)
        self.assertIn("Ret20dVsIndex", universe_df.columns)
        self.assertIn("Ret60dVsIndex", universe_df.columns)
        self.assertIn("RelStrength20Rank", universe_df.columns)
        self.assertIn("RelStrength60Rank", universe_df.columns)
        self.assertIn("SectorBreadthAboveSMA20Pct", universe_df.columns)
        self.assertIn("SectorBreadthAboveSMA50Pct", universe_df.columns)
        self.assertIn("SectorBreadthPositive5dPct", universe_df.columns)
        self.assertIn("SectorADTVRank", universe_df.columns)
        self.assertIn("Ret20dVsSector", universe_df.columns)
        self.assertIn("Ret60dVsSector", universe_df.columns)
        self.assertNotIn("OneLotNotional_kVND", universe_df.columns)
        self.assertNotIn("AboveSMA20", universe_df.columns)
        self.assertNotIn("IndexRegimeTag", universe_df.columns)
        self.assertNotIn("SectorRegimeTag", universe_df.columns)
        self.assertNotIn("SectorLeadershipTag", universe_df.columns)
        self.assertNotIn("SectorLeadershipScore", universe_df.columns)
        self.assertNotIn("SectorConfidence", universe_df.columns)
        self.assertNotIn("MarketActionBias", universe_df.columns)
        self.assertNotIn("MarketActionConfidence", universe_df.columns)
        self.assertNotIn("IndexRangeState60", universe_df.columns)
        self.assertNotIn("BreadthState", universe_df.columns)
        self.assertNotIn("RelativeStrengthScore", universe_df.columns)
        self.assertNotIn("TrendScore", universe_df.columns)
        self.assertNotIn("MeanRevScore", universe_df.columns)
        self.assertNotIn("FlowScore", universe_df.columns)
        self.assertNotIn("BuyScore", universe_df.columns)
        self.assertNotIn("SellScore", universe_df.columns)
        self.assertNotIn("Confidence", universe_df.columns)
        self.assertNotIn("OrderBias", universe_df.columns)
        self.assertNotIn("SetupTag", universe_df.columns)
        self.assertNotIn("PresetFitMomentum", universe_df.columns)
        self.assertNotIn("PresetFitMeanRev", universe_df.columns)
        self.assertNotIn("PresetFitBalanced", universe_df.columns)
        self.assertNotIn("RiskGuards", universe_df.columns)
        self.assertIn("IsVN30", universe_df.columns)
        self.assertEqual(len(universe_df), 2)
        self.assertTrue((universe_df["IsVN30"] == 0).all())
        aaa_market = universe_df.loc[universe_df["Ticker"] == "AAA"].iloc[0]
        self.assertEqual(aaa_market["LotSize"], config.min_lot)
        self.assertAlmostEqual(
            aaa_market["SlippageOneTickPct"],
            (aaa_market["TickSize"] / aaa_market["Last"]) * 100.0,
        )
        if not pd.isna(aaa_market["ADTV20_shares"]):
            self.assertAlmostEqual(
                aaa_market["PositionPctADV20"],
                (aaa_market["PositionQuantity"] / aaa_market["ADTV20_shares"]) * 100.0,
            )
        total_nav = universe_df["EnginePortfolioMarketValue_kVND"].iloc[0]
        self.assertGreaterEqual(total_nav, 0.0)
        if total_nav:
            self.assertAlmostEqual(
                aaa_market["PositionWeightPct"],
                (aaa_market["PositionMarketValue_kVND"] / total_nav) * 100.0,
            )
        self.assertAlmostEqual(aaa_market["Last"], 12.5)
        self.assertAlmostEqual(aaa_market["Ref"], 12.0)
        self.assertAlmostEqual(
            aaa_market["ChangePct"],
            ((aaa_market["Last"] / aaa_market["Ref"]) - 1.0) * 100.0,
        )
        self.assertAlmostEqual(aaa_market["DistRefPct"], aaa_market["ChangePct"])
        self.assertAlmostEqual(aaa_market["Ceil"], 12.8)
        self.assertAlmostEqual(aaa_market["Floor"], 11.2)
        self.assertAlmostEqual(aaa_market["PositionQuantity"], 10.0)
        self.assertEqual(
            universe_df.loc[universe_df["Ticker"] == "AAA", "EngineRunAt"].iloc[0],
            universe_df["EngineRunAt"].iloc[0],
        )

        self.assertNotIn("ADV20", universe_df.columns)
        self.assertNotIn("ADTV20_kVND", universe_df.columns)
        self.assertNotIn("OneLotNotional_kVND", universe_df.columns)
        self.assertNotIn("BandDistance", universe_df.columns)
        self.assertIn("IndexRangePos60", market_summary)
        self.assertIn("BreadthAboveSMA20Pct", market_summary)
        self.assertIn("MarketCoMovement20Pct", market_summary)
        self.assertIn("GeneratedAt", market_summary)
        self.assertIn("SectorBreadthAboveSMA20Pct", sector_summary_df.columns)
        self.assertIn("SectorMedianForeignFlow5d_kVND", sector_summary_df.columns)
        self.assertIn("SectorMedianProprietaryFlow20d_kVND", sector_summary_df.columns)
        self.assertNotIn("IndexRegimeTag", market_summary)
        self.assertNotIn("IndexRangeState60", market_summary)
        self.assertNotIn("BreadthState", market_summary)
        self.assertNotIn("MarketActionBias", market_summary)
        self.assertNotIn("MarketActionConfidence", market_summary)
        self.assertNotIn("SectorRegimeTag", sector_summary_df.columns)
        self.assertNotIn("SectorLeadershipTag", sector_summary_df.columns)
        self.assertNotIn("SectorLeadershipScore", sector_summary_df.columns)
        self.assertNotIn("SectorConfidence", sector_summary_df.columns)

        for required in ["universe.csv"]:
            self.assertTrue((out_dir / required).exists())
        # sanity check merged holdings columns align with prompt contract
        self.assertIn("PositionMarketValue_kVND", universe_df.columns)

        self.assertGreater(summary["tickers"], 0)
        self.assertEqual(summary["market_summary"], str(market_summary_path))
        self.assertEqual(summary["sector_summary"], str(sector_summary_path))

        positions_df = pd.read_csv(positions_path)
        self.assertIn("Ticker", positions_df.columns)
        self.assertIn("Quantity", positions_df.columns)
        self.assertFalse(positions_df.empty)

    def test_industry_filter_limits_universe(self):
        history_df = pd.DataFrame(
            {
                "Date": ["2024-07-01", "2024-07-02", "2024-07-01", "2024-07-02"],
                "Ticker": ["AAA", "AAA", "BBB", "BBB"],
                "Open": [10, 11, 19, 21],
                "High": [11, 12, 21, 22],
                "Low": [9, 10, 18, 19],
                "Close": [11, 12, 20, 21],
                "Volume": [1000, 1200, 800, 900],
                "t": [1, 2, 1, 2],
            }
        )
        intraday_df = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "Ts": [3, 3],
                "Price": [12.5, 21.5],
                "RSI14": [55, 60],
                "TimeVN": ["2024-07-02 14:30:00", "2024-07-02 14:30:00"],
            }
        )
        with mock.patch.dict("os.environ", {"INDUSTRY_TICKER_FILTER": "AAA"}, clear=False):
            config_path = self._write_config()
            config = EngineConfig.from_yaml(config_path)
            engine = DataEngine(
                config,
                FakeMarketDataService(history_df, intraday_df),
                vn30_fetcher=lambda: set(),
            )
            summary = engine.run()

        out_dir = config.output_base_dir
        universe_path = out_dir / "universe.csv"
        positions_path = out_dir / "positions.csv"
        self.assertTrue(universe_path.exists())
        self.assertTrue(positions_path.exists())

        universe_df = pd.read_csv(universe_path)
        self.assertEqual(sorted(universe_df["Ticker"].unique().tolist()), ["AAA"])
        positions_df = pd.read_csv(positions_path)
        self.assertEqual(sorted(positions_df["Ticker"].unique().tolist()), ["AAA"])
        self.assertEqual(summary["tickers"], 1)

    def test_market_summary_uses_benchmark_breadth_for_tiny_universe(self):
        dates = pd.date_range("2024-06-01", periods=25, freq="D")
        history_rows = []
        for idx, date in enumerate(dates):
            aaa_close = 30.0 - idx + (1.2 if idx % 3 == 0 else 0.0)
            bbb_close = 10.0 + idx - (1.2 if idx % 4 == 0 else 0.0)
            index_close = 1200.0 + (idx * 2.0) - (2.5 if idx % 6 == 0 else 0.0)
            history_rows.extend(
                [
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Ticker": "AAA",
                        "Open": aaa_close + 0.1,
                        "High": aaa_close + 0.5,
                        "Low": aaa_close - 0.5,
                        "Close": aaa_close,
                        "Volume": 1000 + idx,
                        "t": idx + 1,
                    },
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Ticker": "BBB",
                        "Open": bbb_close - 0.1,
                        "High": bbb_close + 0.5,
                        "Low": bbb_close - 0.5,
                        "Close": bbb_close,
                        "Volume": 1200 + idx,
                        "t": idx + 1,
                    },
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Ticker": "VNINDEX",
                        "Open": index_close - 1.0,
                        "High": index_close + 2.0,
                        "Low": index_close - 2.0,
                        "Close": index_close,
                        "Volume": 100_000 + idx,
                        "t": idx + 1,
                    },
                ]
            )
        history_df = pd.DataFrame(history_rows)
        intraday_df = pd.DataFrame(
            {
                "Ticker": ["AAA", "VNINDEX"],
                "Ts": [999, 999],
                "Price": [6.5, 1249.0],
                "RSI14": [35, 60],
                "TimeVN": ["2024-06-25 14:30:00", "2024-06-25 14:30:00"],
            }
        )
        config_path = self._write_config(
            industry_rows=[
                {"Ticker": "AAA", "Sector": "Tech"},
                {"Ticker": "BBB", "Sector": "Finance"},
            ],
            portfolio_csv_text="Ticker,Quantity,AvgPrice\nAAA,10,12\n",
            universe_tickers=["AAA"],
        )
        config = EngineConfig.from_yaml(config_path)
        engine = DataEngine(
            config,
            FakeMarketDataService(history_df, intraday_df),
            vn30_fetcher=lambda: {"BBB"},
        )

        engine.run()

        market_summary = json.loads((config.output_base_dir / "market_summary.json").read_text(encoding="utf-8"))
        universe_df = pd.read_csv(config.output_base_dir / "universe.csv")
        asset_tickers = sorted(
            ticker for ticker in universe_df["Ticker"].astype(str).str.upper().unique().tolist() if ticker != "VNINDEX"
        )
        self.assertEqual(asset_tickers, ["AAA"])
        self.assertEqual(market_summary["UniverseTickerCount"], 1)
        self.assertEqual(market_summary["BreadthUniverseTickerCount"], 1)
        self.assertEqual(market_summary["BreadthSource"], "benchmark_basket")
        self.assertEqual(market_summary["BreadthAboveSMA20Pct"], 100.0)

    def test_config_core_tickers_strictly_filter_universe_and_portfolio(self):
        history_df = pd.DataFrame(
            {
                "Date": [
                    "2024-07-01",
                    "2024-07-02",
                    "2024-07-01",
                    "2024-07-02",
                    "2024-07-01",
                    "2024-07-02",
                    "2024-07-01",
                    "2024-07-02",
                ],
                "Ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "DDD"],
                "Open": [10, 11, 19, 21, 29, 30, 39, 40],
                "High": [11, 12, 21, 22, 31, 32, 41, 42],
                "Low": [9, 10, 18, 19, 28, 29, 38, 39],
                "Close": [11, 12, 20, 21, 30, 31, 40, 41],
                "Volume": [1000, 1200, 800, 900, 700, 750, 600, 650],
                "t": [1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        intraday_df = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB", "CCC", "DDD"],
                "Ts": [3, 3, 3, 3],
                "Price": [12.5, 21.5, 31.5, 41.5],
                "RSI14": [55, 60, 58, 57],
                "TimeVN": [
                    "2024-07-02 14:30:00",
                    "2024-07-02 14:30:00",
                    "2024-07-02 14:30:00",
                    "2024-07-02 14:30:00",
                ],
            }
        )
        config_path = self._write_config(
            industry_rows=[
                {"Ticker": "AAA", "Sector": "Tech"},
                {"Ticker": "BBB", "Sector": "Finance"},
                {"Ticker": "CCC", "Sector": "Utilities"},
                {"Ticker": "DDD", "Sector": "Materials"},
            ],
            portfolio_csv_text="Ticker,Quantity,AvgPrice\nBBB,5,20\n",
            core_tickers=["AAA"],
            preferred_tickers=["CCC"],
        )

        config = EngineConfig.from_yaml(config_path)
        self.assertEqual(config.industry_ticker_filter, ["AAA"])
        self.assertEqual(
            config.industry_ticker_filter_source,
            "config universe.core_tickers",
        )
        engine = DataEngine(
            config,
            FakeMarketDataService(history_df, intraday_df),
            vn30_fetcher=lambda: set(),
        )
        summary = engine.run()

        out_dir = config.output_base_dir
        universe_df = pd.read_csv(out_dir / "universe.csv")
        positions_df = pd.read_csv(out_dir / "positions.csv")

        self.assertEqual(sorted(universe_df["Ticker"].unique().tolist()), ["AAA"])
        self.assertTrue(positions_df.empty)
        self.assertEqual(summary["tickers"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
