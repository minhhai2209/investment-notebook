from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_momentum_continuation_report import build_momentum_continuation_report


class BuildMomentumContinuationReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.analysis = self.out / "analysis"
        self.candidates_dir = self.analysis / "candidates"
        self.output_dir = self.analysis / "momentum"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def _write_inputs(self) -> None:
        (self.candidates_dir / "candidate_watchlist_full.json").write_text(
            json.dumps(
                {
                    "SnapshotDate": "2026-04-24T10:00:00+07:00",
                    "Rows": [
                        {
                            "Ticker": "AAA",
                            "Decision": "mua_ngay",
                            "CurrentPrice": 50.0,
                            "PreferredBuyZoneLow": 49.0,
                            "PreferredBuyZoneHigh": 50.5,
                            "ZoneStatus": "inside",
                            "ZoneGapPct": 0.0,
                            "NoChaseAbove": 50.5,
                            "InvalidationBelow": 48.0,
                            "BestTimingWindow": "T+3",
                            "BestTimingNetEdgePct": 4.0,
                            "T10NetEdgePct": 2.0,
                            "ForecastCloseRetPctT1": 0.5,
                            "ForecastCandleBias": "BULLISH",
                            "ForecastConsistencyStatus": "conflict",
                            "ForecastConsistencySummary": "timing peak exceeds OHLC path",
                            "ForecastConsistencyGapPct": 12.5,
                            "ReferenceBudgetPlanSummary": "50.0: starter, 49.0: main",
                            "SessionBuyPlanSummary": "50.0: starter",
                            "SpecializedOverlayScore": 2,
                            "SpecializedRegime": "trend_persistence_pullback_add",
                            "SpecializedActionBias": "giu_trend_add_co_chon_loc",
                        },
                        {
                            "Ticker": "BBB",
                            "Decision": "không_mua",
                            "CurrentPrice": 120.0,
                            "PreferredBuyZoneLow": 105.0,
                            "PreferredBuyZoneHigh": 110.0,
                            "ZoneStatus": "above",
                            "ZoneGapPct": 9.0,
                            "NoChaseAbove": 110.0,
                            "InvalidationBelow": 98.0,
                            "BestTimingWindow": "T+3",
                            "BestTimingNetEdgePct": 5.0,
                            "T10NetEdgePct": -4.0,
                            "ForecastCloseRetPctT1": -1.0,
                            "ForecastCandleBias": "BEARISH",
                            "ReferenceBudgetPlanSummary": "110.0: shallow, 105.0: deep",
                            "SessionBuyPlanSummary": "110.0: shallow",
                            "SpecializedOverlayScore": -6,
                            "SpecializedRegime": "fresh_burst_distribution",
                            "SpecializedActionBias": "khong_duoi_burst_cho_deep_pullback",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Last": 50.0,
                    "ChangePct": 0.2,
                    "RSI14": 55.0,
                    "DistSMA20Pct": 1.0,
                    "Ret5d": 1.5,
                    "Ret20d": 5.0,
                    "Ret20dVsIndex": 1.0,
                    "Ret60dVsIndex": 3.0,
                },
                {
                    "Ticker": "BBB",
                    "Last": 120.0,
                    "ChangePct": -3.0,
                    "RSI14": 84.0,
                    "DistSMA20Pct": 22.0,
                    "Ret5d": 12.0,
                    "Ret20d": 40.0,
                    "Ret20dVsIndex": 30.0,
                    "Ret60dVsIndex": 50.0,
                },
            ]
        ).to_csv(self.out / "universe.csv", index=False)

        (self.out / "market_summary.json").write_text(
            json.dumps(
                {
                    "GeneratedAt": "2026-04-24T10:00:00+07:00",
                    "BreadthPositive1dPct": 25.0,
                    "AdvanceDeclineRatio": 0.4,
                    "IndexRangePos20": 0.94,
                }
            ),
            encoding="utf-8",
        )
        events_path = self.root / "events.json"
        events_path.write_text(
            json.dumps(
                {
                    "Events": [
                        {
                            "Date": "2026-04-30",
                            "Label": "Nghi le 30/4",
                            "MarketClosed": True,
                            "Impact": "high",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.events_path = events_path

    def test_builds_direction_and_urgency_report(self) -> None:
        self._write_inputs()

        report = build_momentum_continuation_report(
            candidates_json=self.candidates_dir / "candidate_watchlist_full.json",
            universe_csv=self.out / "universe.csv",
            market_summary_json=self.out / "market_summary.json",
            events_json=self.events_path,
            output_dir=self.output_dir,
            tickers=["AAA", "BBB"],
        )

        rows = {row["Ticker"]: row for row in report["Rows"]}
        self.assertEqual(report["EventRisk"], "holiday_gap")
        self.assertEqual(rows["AAA"]["DirectionCall"], "model conflict / chờ xác nhận")
        self.assertEqual(rows["AAA"]["Urgency"], "model action: timing/OHLC conflict, wait for path confirmation")
        self.assertTrue(any("forecast conflict" in reason for reason in rows["AAA"]["ReasonBullets"]))
        self.assertEqual(rows["AAA"]["ForecastConsistencyStatus"], "conflict")
        self.assertEqual(rows["BBB"]["DirectionCall"], "giảm hoặc rủi ro giảm")
        self.assertEqual(rows["BBB"]["Urgency"], "model action: no new buy")
        self.assertTrue((self.output_dir / "momentum_continuation.json").exists())
        self.assertTrue((self.output_dir / "momentum_continuation.md").exists())

    def test_empty_ticker_list_uses_candidate_watchlist_rows(self) -> None:
        self._write_inputs()

        report = build_momentum_continuation_report(
            candidates_json=self.candidates_dir / "candidate_watchlist_full.json",
            universe_csv=self.out / "universe.csv",
            market_summary_json=self.out / "market_summary.json",
            events_json=self.events_path,
            output_dir=self.output_dir,
            tickers=[],
        )

        self.assertEqual(sorted(row["Ticker"] for row in report["Rows"]), ["AAA", "BBB"])
        self.assertEqual(report["Tickers"], ["AAA", "BBB"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
