from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.data_fetching.vndirect_price_depth import (
    _decode_stock_snapshot,
    _decode_priceboard_payload,
    append_depth_snapshot_cache,
    merge_depth_into_intraday_cache,
)


def _encode_priceboard_payload(parts: list[str]) -> str:
    raw = "|".join(parts)
    return "".join(chr(ord(char) - (index % 5)) for index, char in enumerate(raw))


class VndirectPriceDepthTest(unittest.TestCase):
    def test_decode_priceboard_payload_reverses_vndirect_shift(self) -> None:
        encoded = _encode_priceboard_payload(["SFU", "VIC", "ST", "10"])

        self.assertEqual(_decode_priceboard_payload(encoded), ["SFU", "VIC", "ST", "10"])

    def test_decode_stock_snapshot_extracts_top_depth(self) -> None:
        values = [
            "VIC",
            "ST",
            "10",
            "196.0",
            "182.3",
            "209.7",
            "195.0",
            "194.0",
            "193.9",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "50.0",
            "20.0",
            "10.0",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "195.5",
            "195.6",
            "195.7",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "970.0",
            "40.0",
            "600.0",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
        encoded = _encode_priceboard_payload(["SFU", *values])

        decoded = _decode_stock_snapshot(encoded)

        self.assertEqual(decoded["code"], "VIC")
        self.assertEqual(decoded["bidPrice01"], "195.0")
        self.assertEqual(decoded["bidQtty01"], "50.0")
        self.assertEqual(decoded["offerPrice01"], "195.5")
        self.assertEqual(decoded["offerQtty01"], "970.0")

    def test_merge_depth_into_intraday_cache_writes_loader_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "VIC_1m.csv"
            ts = pd.Timestamp("2026-06-08T07:30:00Z")
            pd.DataFrame(
                {
                    "t": [int(ts.timestamp())],
                    "open": [195.0],
                    "high": [195.5],
                    "low": [194.9],
                    "close": [195.2],
                    "volume": [1000],
                }
            ).to_csv(cache_path, index=False)
            depth = pd.DataFrame(
                [
                    {
                        "Ticker": "VIC",
                        "FetchedAt": "2026-06-08T14:31:00+07:00",
                        "BestBid1": 195.0,
                        "BestAsk1": 195.5,
                        "BidVolume1": 50.0,
                        "AskVolume1": 970.0,
                        "BidVolume2": 20.0,
                        "AskVolume2": 40.0,
                        "BidVolume3": 10.0,
                        "AskVolume3": 600.0,
                    }
                ]
            )

            merge_depth_into_intraday_cache(depth, root, "1")

            merged = pd.read_csv(cache_path)
            self.assertAlmostEqual(float(merged["best_bid"].iloc[0]), 195.0)
            self.assertAlmostEqual(float(merged["best_ask"].iloc[0]), 195.5)
            self.assertAlmostEqual(float(merged["bid_volume_1"].iloc[0]), 50.0)
            self.assertAlmostEqual(float(merged["ask_volume_1"].iloc[0]), 970.0)

    def test_depth_cache_trims_snapshots_outside_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pd.DataFrame(
                [{"Ticker": "VIC", "FetchedAt": (now - timedelta(days=40)).isoformat(), "BestBid1": 190.0}]
            ).to_csv(root / "VIC_depth.csv", index=False)
            incoming = pd.DataFrame(
                [{"Ticker": "VIC", "FetchedAt": now.isoformat(), "BestBid1": 195.0}]
            )

            append_depth_snapshot_cache(incoming, root, retention_days=30)

            cached = pd.read_csv(root / "VIC_depth.csv")
            self.assertEqual(len(cached), 1)
            self.assertEqual(float(cached["BestBid1"].iloc[0]), 195.0)


if __name__ == "__main__":
    unittest.main()
