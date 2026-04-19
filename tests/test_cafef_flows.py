from __future__ import annotations

import unittest

import pandas as pd

from scripts.data_fetching.cafef_flows import summarize_flow_metrics


FOREIGN_ROWS = [
    {
        "Date": "2025-11-26",
        "NetShares": -1_340_900,
        "NetValue_billion": -36.1,
        "BuyShares": 3_263_200,
        "BuyValue_billion": 88.27,
        "SellShares": 4_604_100,
        "SellValue_billion": 124.37,
        "RoomRemainingShares": 2_271_264_664,
        "ForeignHoldingPct": 19.41,
    },
    {
        "Date": "2025-11-25",
        "NetShares": 2_628_600,
        "NetValue_billion": 71.54,
        "BuyShares": 6_279_300,
        "BuyValue_billion": 170.32,
        "SellShares": 3_650_700,
        "SellValue_billion": 98.78,
        "RoomRemainingShares": 2_270_000_000,
        "ForeignHoldingPct": 19.30,
    },
]


PROPRIETARY_ROWS = [
    {
        "Date": "2025-11-25",
        "BuyShares": 1_760_200,
        "BuyValue_billion": 47.73,
        "SellShares": 1_524_845,
        "SellValue_billion": 41.29,
        "NetShares": 235_355,
        "NetValue_billion": 6.44,
    },
    {
        "Date": "2025-11-24",
        "BuyShares": 3_590_430,
        "BuyValue_billion": 98.13,
        "SellShares": 1_291_590,
        "SellValue_billion": 35.26,
        "NetShares": 2_298_840,
        "NetValue_billion": 62.87,
    },
]


class CafeFFlowParsingTest(unittest.TestCase):
    def test_summarize_flow_metrics(self) -> None:
        foreign_df = pd.DataFrame(FOREIGN_ROWS)
        proprietary_df = pd.DataFrame(PROPRIETARY_ROWS)
        metrics = summarize_flow_metrics("HPG", foreign_df, proprietary_df, horizons=(1, 2))
        self.assertEqual(metrics["Ticker"], "HPG")
        self.assertAlmostEqual(metrics["NetBuySellForeign_shares_1d"], -1340900)
        expected_foreign_kvnd = (-36.1 + 71.54) * 1_000_000
        self.assertAlmostEqual(metrics["NetBuySellForeign_kVND_2d"], expected_foreign_kvnd)
        self.assertAlmostEqual(metrics["ForeignHoldingPct"], 19.41)
        expected_prop_kvnd = (6.44 + 62.87) * 1_000_000
        self.assertAlmostEqual(metrics["NetBuySellProprietary_kVND_2d"], expected_prop_kvnd)
        self.assertAlmostEqual(metrics["NetBuySellProprietary_shares_1d"], 235_355)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
