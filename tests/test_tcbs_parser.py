from __future__ import annotations

import unittest

from scripts.parsers.tcbs import parse_tcbs_table


class TcbsParserTest(unittest.TestCase):
    def test_parse_minimal_headers_and_rows(self):
        headers = [
            "Mã",
            "SL Tổng",
            "Được GD",
            "Giá vốn",
            "Thị giá",
        ]
        rows = [
            ["HT1\nPRIORITY_HIGH", "1,600", "1,600", "19.21", "17.65"],
            ["LPB\nPRIORITY_HIGH", "1,000", "1,000", "51.24", "51.90"],
            ["VNM", "900", "900", "58.78", "57.00"],
        ]
        df = parse_tcbs_table(headers, rows)
        self.assertEqual(list(df.columns), ["Ticker", "Quantity", "AvgPrice"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df.loc[0, "Ticker"], "HT1")
        self.assertEqual(int(df.loc[0, "Quantity"]), 1600)
        self.assertAlmostEqual(float(df.loc[0, "AvgPrice"]), 19.21)

    def test_parse_with_fallback_quantity_column(self):
        # Missing 'SL Tổng' -> should fallback to 'Được GD'
        headers = ["Mã", "Được GD", "Giá vốn"]
        rows = [["AAA", "2,000", "12.34"]]
        df = parse_tcbs_table(headers, rows)
        self.assertEqual(int(df.loc[0, "Quantity"]), 2000)
        self.assertAlmostEqual(float(df.loc[0, "AvgPrice"]), 12.34)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
