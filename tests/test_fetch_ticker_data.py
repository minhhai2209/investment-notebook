from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.data_fetching.fetch_ticker_data import VN_TZ, ensure_intraday_cache, ensure_ohlc_cache


def _payload_from_ts(ts_values):
    return {
        "s": "ok",
        "t": ts_values,
        "o": [10.0 + idx for idx, _ in enumerate(ts_values)],
        "h": [10.5 + idx for idx, _ in enumerate(ts_values)],
        "l": [9.5 + idx for idx, _ in enumerate(ts_values)],
        "c": [10.2 + idx for idx, _ in enumerate(ts_values)],
        "v": [1000 + idx for idx, _ in enumerate(ts_values)],
    }


class FetchTickerDataTest(unittest.TestCase):
    def test_missing_cache_seeds_full_window_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            now_ts = int(datetime.now(VN_TZ).timestamp())
            seeded = _payload_from_ts([now_ts - 86400, now_ts])
            with patch("scripts.data_fetching.fetch_ticker_data.fetch_history", return_value=seeded) as mocked_fetch:
                ensure_ohlc_cache("AAA", outdir=str(outdir), min_days=30, recent_reconcile_days=3)

            self.assertEqual(mocked_fetch.call_count, 1)
            cached = pd.read_csv(outdir / "AAA_daily.csv")
            self.assertEqual(int(cached.shape[0]), 2)

    def test_existing_cache_backfills_and_reconciles_recent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            cache_path = outdir / "AAA_daily.csv"
            outdir.mkdir(parents=True, exist_ok=True)

            now = datetime.now(VN_TZ)
            ts_existing = [
                int((now - timedelta(days=2)).timestamp()),
                int((now - timedelta(days=1)).timestamp()),
            ]
            pd.DataFrame(
                {
                    "t": ts_existing,
                    "open": [10.0, 11.0],
                    "high": [10.5, 11.5],
                    "low": [9.5, 10.5],
                    "close": [10.2, 11.2],
                    "volume": [1000, 1100],
                    "date_vn": [
                        (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                        (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                    ],
                }
            ).to_csv(cache_path, index=False)

            ts_backfill = [
                int((now - timedelta(days=30)).timestamp()),
                int((now - timedelta(days=29)).timestamp()),
            ]
            ts_recent = [
                int((now - timedelta(days=1)).timestamp()),
                int(now.timestamp()),
            ]
            with patch(
                "scripts.data_fetching.fetch_ticker_data.fetch_history",
                side_effect=[_payload_from_ts(ts_backfill), _payload_from_ts(ts_recent)],
            ) as mocked_fetch:
                ensure_ohlc_cache("AAA", outdir=str(outdir), min_days=30, recent_reconcile_days=3)

            self.assertEqual(mocked_fetch.call_count, 2)
            cached = pd.read_csv(cache_path)
            self.assertEqual(int(cached.shape[0]), 5)
            self.assertEqual(int(cached["t"].min()), min(ts_backfill))
            self.assertEqual(int(cached["t"].max()), max(ts_recent))

    def test_intraday_cache_uses_resolution_specific_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            now_ts = int(datetime.now(VN_TZ).timestamp())
            seeded = _payload_from_ts([now_ts - 300, now_ts])
            with patch("scripts.data_fetching.fetch_ticker_data.fetch_history", return_value=seeded) as mocked_fetch:
                ensure_intraday_cache("AAA", outdir=str(outdir), min_days=5, resolution="5")

            self.assertEqual(mocked_fetch.call_count, 1)
            self.assertTrue((outdir / "AAA_5m.csv").exists())


if __name__ == "__main__":
    unittest.main()
