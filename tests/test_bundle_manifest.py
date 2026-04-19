from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.codex.build_bundle_manifest import (
    CYCLE_BEST_COLUMNS,
    CYCLE_MATRIX_COLUMNS,
    ENTRY_LADDER_COLUMNS,
    INTRADAY_COLUMNS,
    MARKET_SUMMARY_KEYS,
    OHLC_COLUMNS,
    PLAYBOOK_COLUMNS,
    RANGE_COLUMNS,
    SECTOR_SUMMARY_COLUMNS,
    STRATEGY_BUCKET_COLUMNS,
    TIMING_COLUMNS,
    UNIVERSE_COLUMNS,
    build_bundle_manifest,
)


class BundleManifestTest(unittest.TestCase):
    def _write_csv(self, path: Path, columns: list[str], rows: list[dict[str, object]] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows or [], columns=columns)
        frame.to_csv(path, index=False)

    def test_build_bundle_manifest_reports_working_universe_and_optional_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            manifest_path = bundle_dir / "bundle_manifest.json"

            self._write_csv(
                bundle_dir / "universe.csv",
                UNIVERSE_COLUMNS,
                rows=[
                    {"Ticker": "VNINDEX"},
                    {"Ticker": "AAA"},
                    {"Ticker": "BBB"},
                ],
            )
            (bundle_dir / "total_capital_kVND.txt").write_text("5000000\n", encoding="utf-8")
            (bundle_dir / "market_summary.json").write_text(
                json.dumps({key: 0 for key in MARKET_SUMMARY_KEYS}),
                encoding="utf-8",
            )
            self._write_csv(bundle_dir / "sector_summary.csv", SECTOR_SUMMARY_COLUMNS)
            self._write_csv(bundle_dir / "ml_range_predictions_full_2y.csv", RANGE_COLUMNS)
            self._write_csv(bundle_dir / "ml_range_predictions_recent_focus.csv", RANGE_COLUMNS)
            self._write_csv(bundle_dir / "ml_cycle_forecast_ticker_matrix.csv", CYCLE_MATRIX_COLUMNS)
            self._write_csv(bundle_dir / "ml_cycle_forecast_best_horizon.csv", CYCLE_BEST_COLUMNS)
            self._write_csv(bundle_dir / "ticker_playbook_best_configs.csv", PLAYBOOK_COLUMNS)
            self._write_csv(bundle_dir / "ml_ohlc_next_session.csv", OHLC_COLUMNS)
            self._write_csv(bundle_dir / "ml_intraday_rest_of_session.csv", INTRADAY_COLUMNS)
            self._write_csv(bundle_dir / "ml_single_name_timing.csv", TIMING_COLUMNS)
            self._write_csv(bundle_dir / "ml_entry_ladder_eval.csv", ENTRY_LADDER_COLUMNS)
            self._write_csv(bundle_dir / "strategy_buckets.csv", STRATEGY_BUCKET_COLUMNS)
            (bundle_dir / "human_notes.md").write_text("# Notes\n", encoding="utf-8")
            (bundle_dir / "research").mkdir()
            (bundle_dir / "research" / "manifest.json").write_text(
                json.dumps({"SchemaVersion": 1, "UniverseTickers": ["AAA", "BBB"], "Tickers": {}}),
                encoding="utf-8",
            )

            manifest = build_bundle_manifest(bundle_dir=bundle_dir, output_path=manifest_path)

            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["WorkingUniverseTickers"], ["AAA", "BBB"])
            self.assertTrue(manifest["Files"]["ml_single_name_timing.csv"]["Exists"])
            self.assertTrue(manifest["Files"]["research/manifest.json"]["Exists"])

    def test_build_bundle_manifest_fails_when_required_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Missing required bundle file"):
                build_bundle_manifest(bundle_dir=bundle_dir, output_path=bundle_dir / "bundle_manifest.json")


if __name__ == "__main__":
    unittest.main()
