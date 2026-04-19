from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.data_fetching.macro_factor_cache import (
    load_macro_factor_matrix,
    parse_fred_csv_text,
    parse_stooq_csv_text,
)


class MacroFactorCacheTest(unittest.TestCase):
    def test_parse_fred_csv_text_normalises_columns(self) -> None:
        text = "\n".join(
            [
                "observation_date,SP500",
                "2026-03-19,5662.89",
                "2026-03-20,5670.12",
            ]
        )
        frame = parse_fred_csv_text(text, "SP500")
        self.assertEqual(list(frame.columns), ["Date", "Value"])
        self.assertEqual(str(frame.iloc[-1]["Date"].date()), "2026-03-20")
        self.assertAlmostEqual(float(frame.iloc[-1]["Value"]), 5670.12, places=6)

    def test_parse_stooq_csv_text_normalises_columns(self) -> None:
        text = "\n".join(
            [
                "Date,Open,High,Low,Close",
                "2026-03-19,3020,3030,3010,3025",
                "2026-03-20,3025,3040,3020,3038",
            ]
        )
        frame = parse_stooq_csv_text(text, "Close")
        self.assertEqual(str(frame.iloc[-1]["Date"].date()), "2026-03-20")
        self.assertAlmostEqual(float(frame.iloc[-1]["Value"]), 3038.0, places=6)

    def test_load_macro_factor_matrix_combines_cached_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            (cache_dir / "SP500.csv").write_text("Date,Value\n2026-03-19,5662.89\n2026-03-20,5670.12\n", encoding="utf-8")
            (cache_dir / "VIX.csv").write_text("Date,Value\n2026-03-19,18.1\n2026-03-20,17.4\n", encoding="utf-8")
            matrix = load_macro_factor_matrix(cache_dir)
            self.assertEqual(list(matrix.columns), ["SP500", "VIX"])
            self.assertAlmostEqual(float(matrix.loc["2026-03-20", "VIX"]), 17.4, places=6)


if __name__ == "__main__":
    unittest.main()
