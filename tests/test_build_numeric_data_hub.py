from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_hub.build_numeric_data_hub import build_data_hub, enrich_daily_frame


class NumericDataHubTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.daily_dir = self.base / "daily_cache"
        self.intraday_dir = self.base / "intraday_cache"
        self.depth_dir = self.base / "depth_cache"
        self.out_dir = self.base / "hub"
        self.daily_dir.mkdir()
        self.intraday_dir.mkdir()
        self.depth_dir.mkdir()

    def _write_config(self) -> Path:
        config = self.base / "data_hub.yaml"
        config.write_text(
            f"""
tickers:
  - VIC
  - VNINDEX
paths:
  daily_cache: {self.daily_dir}
  intraday_cache: {self.intraday_dir}
  depth_cache: {self.depth_dir}
  cafef_cache: {self.base / "cafef"}
  vietstock_overview_cache: {self.base / "overview"}
  vietstock_bctt_cache: {self.base / "bctt"}
  macro_cache: {self.base / "macro"}
  sector_map: {self.base / "industry_map.csv"}
  output_dir: {self.out_dir}
refresh:
  daily: false
  intraday: false
  depth: false
  cafef_flows: false
  vietstock_overview: false
  vietstock_bctt: false
  macro: false
windows:
  recent_daily_rows: 20
  recent_intraday_rows: 10
""",
            encoding="utf-8",
        )
        return config

    def _write_daily(self, ticker: str) -> None:
        rows = []
        start = pd.Timestamp("2025-01-01")
        for i in range(260):
            close = 100.0 + i
            rows.append(
                {
                    "date_vn": (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": close - 1.0,
                    "high": close + 2.0,
                    "low": close - 2.0,
                    "close": close,
                    "volume": 1_000_000 + i,
                }
            )
        pd.DataFrame(rows).to_csv(self.daily_dir / f"{ticker}_daily.csv", index=False)

    def _write_intraday(self, ticker: str) -> None:
        base_ts = 1_735_686_000
        rows = []
        for i in range(12):
            rows.append(
                {
                    "t": base_ts + i * 60,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 10_000 + i,
                }
            )
        pd.DataFrame(rows).to_csv(self.intraday_dir / f"{ticker}_1m.csv", index=False)

    def test_builds_chatgpt_browseable_numeric_outputs(self) -> None:
        self._write_daily("VIC")
        self._write_daily("VNINDEX")
        self._write_intraday("VIC")
        manifest = build_data_hub(self._write_config())

        self.assertEqual(manifest["purpose"], "Numeric-only market data hub for fast ChatGPT browsing. No news, no recommendations, no model forecasts.")
        for filename in ["manifest.json", "README.md", "source_status.csv", "latest_metrics.csv", "api_catalog.csv", "calculation_catalog.csv", "tickers.csv"]:
            self.assertTrue((self.out_dir / filename).exists(), filename)
        self.assertTrue((self.out_dir / "daily" / "VIC.csv").exists())
        self.assertTrue((self.out_dir / "intraday" / "VIC.csv").exists())
        self.assertTrue((self.out_dir / "intraday" / "minute_profile" / "VIC.csv").exists())
        self.assertTrue((self.out_dir / "market" / "breadth_daily.csv").exists())
        self.assertTrue((self.out_dir / "market" / "cross_section_latest.csv").exists())

        latest = pd.read_csv(self.out_dir / "latest_metrics.csv")
        vic = latest[latest["Ticker"].eq("VIC")].iloc[0]
        self.assertIn("Ret1dPct", latest.columns)
        self.assertIn("RSI14", latest.columns)
        self.assertIn("IntradayVWAP", latest.columns)
        self.assertIn("DailyTradedValue", latest.columns)
        self.assertIn("ExcessRet20dVsVNINDEXPct", latest.columns)
        self.assertIn("IntradayAvgVolumePerMinute", latest.columns)
        self.assertGreater(float(vic["LastClose"]), 0.0)

        saved_manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["read_order_for_chatgpt"][0], "manifest.json")
        self.assertIn("calculation_catalog", saved_manifest["files"])
        self.assertIn("source_status", saved_manifest["files"])

    def test_enrich_daily_frame_adds_feature_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=30),
                "open": range(100, 130),
                "high": range(102, 132),
                "low": range(98, 128),
                "close": range(101, 131),
                "volume": range(1000, 1030),
            }
        )
        enriched = enrich_daily_frame(frame)
        self.assertIn("ret_20d_pct", enriched.columns)
        self.assertIn("atr14_pct", enriched.columns)
        self.assertIn("volume_ratio_20", enriched.columns)
        self.assertIn("traded_value", enriched.columns)
        self.assertIn("realized_vol_20d_pct", enriched.columns)
        self.assertIn("drawdown_60d_pct", enriched.columns)


if __name__ == "__main__":
    unittest.main()
