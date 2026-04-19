from __future__ import annotations

import unittest

import pandas as pd

from scripts.analysis.build_intraday_rest_of_session_report import (
    _prepare_current_snapshot_row,
    classify_snapshot_bucket,
    select_current_intraday_rest_of_session_forecasts,
    summarise_intraday_rest_of_session_metrics,
    summarise_intraday_snapshots,
)


def _intraday_rows(day: str, prices: list[float], volumes: list[int], times: list[str]) -> pd.DataFrame:
    timestamps = pd.to_datetime([f"{day} {value}+07:00" for value in times])
    frame = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Open": prices,
            "High": [price + 0.1 for price in prices],
            "Low": [price - 0.1 for price in prices],
            "Close": prices,
            "Volume": volumes,
        }
    )
    frame["TradeDate"] = frame["Timestamp"].dt.normalize()
    frame["TradeTime"] = frame["Timestamp"].dt.time
    return frame


class BuildIntradayRestOfSessionReportTest(unittest.TestCase):
    def test_classify_snapshot_bucket_handles_supported_windows(self) -> None:
        self.assertEqual(classify_snapshot_bucket("2026-03-30T09:50:00+07:00"), "AM_EARLY")
        self.assertEqual(classify_snapshot_bucket("2026-03-30T11:45:00+07:00"), "LUNCH_BREAK")
        self.assertEqual(classify_snapshot_bucket("2026-03-30T13:50:00+07:00"), "PM_LATE")
        self.assertIsNone(classify_snapshot_bucket("2026-03-30T09:15:00+07:00"))
        self.assertIsNone(classify_snapshot_bucket("2026-03-30T14:28:00+07:00"))

    def test_summarise_intraday_snapshots_keeps_partial_latest_row_without_targets(self) -> None:
        baseline_day = _intraday_rows(
            "2026-03-27",
            prices=[9.8, 9.9, 10.0],
            volumes=[90, 100, 110],
            times=["09:00:00", "11:30:00", "14:30:00"],
        )
        previous_day = _intraday_rows(
            "2026-03-28",
            prices=[10.0, 10.1, 10.3, 10.4, 10.45],
            volumes=[100, 120, 140, 160, 180],
            times=["09:00:00", "09:45:00", "11:30:00", "13:45:00", "14:30:00"],
        )
        latest_partial = _intraday_rows(
            "2026-03-29",
            prices=[10.2, 10.3, 10.5],
            volumes=[110, 120, 130],
            times=["09:00:00", "13:30:00", "13:50:00"],
        )
        summary = summarise_intraday_snapshots(
            pd.concat([baseline_day, previous_day, latest_partial], ignore_index=True),
            prefix="Ticker",
            include_partial_latest=True,
        )

        self.assertGreaterEqual(int(summary.shape[0]), 2)
        historical_row = summary[
            (summary["SnapshotDate"] == "2026-03-28") & (summary["SnapshotTimeBucket"] == "AM_EARLY")
        ].iloc[0]
        current_row = summary[
            (summary["SnapshotDate"] == "2026-03-29") & (summary["SnapshotTimeBucket"] == "PM_LATE")
        ].iloc[0]
        self.assertAlmostEqual(float(historical_row["TickerTargetCloseRetPct"]), ((10.45 / 10.1) - 1.0) * 100.0, places=5)
        self.assertTrue(pd.isna(current_row["TickerTargetCloseRetPct"]))

    def test_summarise_intraday_snapshots_adds_hourly_context_from_5m_history(self) -> None:
        baseline_day = _intraday_rows(
            "2026-03-27",
            prices=[9.9, 10.0],
            volumes=[100, 100],
            times=["09:00:00", "14:30:00"],
        )
        measured_day = _intraday_rows(
            "2026-03-28",
            prices=[10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3],
            volumes=[100] * 14,
            times=[
                "09:00:00",
                "09:05:00",
                "09:10:00",
                "09:15:00",
                "09:20:00",
                "09:25:00",
                "09:30:00",
                "09:35:00",
                "09:40:00",
                "09:45:00",
                "09:50:00",
                "09:55:00",
                "10:00:00",
                "14:30:00",
            ],
        )
        latest_partial = _intraday_rows(
            "2026-03-29",
            prices=[10.5, 10.6, 10.7],
            volumes=[100, 100, 100],
            times=["09:00:00", "13:30:00", "13:50:00"],
        )

        summary = summarise_intraday_snapshots(
            pd.concat([baseline_day, measured_day, latest_partial], ignore_index=True),
            prefix="Ticker",
            include_partial_latest=True,
        )

        row = summary[
            (summary["SnapshotDate"] == "2026-03-28")
            & (summary["SnapshotTs"] == pd.Timestamp("2026-03-28T10:00:00+07:00"))
        ].iloc[0]

        self.assertAlmostEqual(float(row["TickerLast60mRetPct"]), 12.0)
        self.assertAlmostEqual(float(row["TickerRange30mPct"]), 8.0)
        self.assertAlmostEqual(float(row["TickerRange60mPct"]), 14.0)
        self.assertAlmostEqual(float(row["TickerPosIn30mRange"]), 0.875)
        self.assertAlmostEqual(float(row["TickerPosIn60mRange"]), 13.0 / 14.0)

    def test_select_current_intraday_rest_of_session_forecasts_prefers_lower_close_error(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "ridge",
                    "TargetCloseRetPct": 1.0,
                    "PredTargetCloseRetPct": 0.2,
                    "TargetHighRetPct": 2.0,
                    "PredTargetHighRetPct": 1.5,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.6,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.1,
                },
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "random_forest",
                    "TargetCloseRetPct": 1.0,
                    "PredTargetCloseRetPct": 0.9,
                    "TargetHighRetPct": 2.0,
                    "PredTargetHighRetPct": 1.9,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.9,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.8,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    "SnapshotDate": "2026-03-30",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Ticker": "AAA",
                    "Base": 10.0,
                    "Model": "ridge",
                    "PredTargetLowRetPct": -2.0,
                    "PredTargetCloseRetPct": 1.0,
                    "PredTargetHighRetPct": 3.0,
                },
                {
                    "SnapshotDate": "2026-03-30",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Ticker": "AAA",
                    "Base": 10.0,
                    "Model": "random_forest",
                    "PredTargetLowRetPct": -1.5,
                    "PredTargetCloseRetPct": 0.8,
                    "PredTargetHighRetPct": 2.5,
                },
            ]
        )

        report = select_current_intraday_rest_of_session_forecasts(history, current)

        self.assertEqual(report["Ticker"].tolist(), ["AAA"])
        row = report.iloc[0]
        self.assertEqual(str(row["Model"]), "random_forest")
        self.assertAlmostEqual(float(row["Mid"]), 10.08)

    def test_summarise_intraday_rest_of_session_metrics_penalises_upside_miss(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "ridge",
                    "TargetCloseRetPct": 0.5,
                    "PredTargetCloseRetPct": 0.6,
                    "TargetHighRetPct": 4.0,
                    "PredTargetHighRetPct": 1.0,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -1.0,
                    "ActualRangePct": 5.0,
                    "PredRangePct": 2.0,
                },
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "hist_gbm",
                    "TargetCloseRetPct": 0.5,
                    "PredTargetCloseRetPct": 0.8,
                    "TargetHighRetPct": 4.0,
                    "PredTargetHighRetPct": 3.8,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.9,
                    "ActualRangePct": 5.0,
                    "PredRangePct": 4.7,
                },
            ]
        )

        summary = summarise_intraday_rest_of_session_metrics(history)

        self.assertEqual(str(summary.iloc[0]["Model"]), "hist_gbm")
        self.assertGreater(float(summary.loc[summary["Model"] == "ridge", "UpsideMissMAEPct"].iloc[0]), 2.5)

    def test_select_current_intraday_rest_of_session_forecasts_is_bucket_aware(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "AM_LATE",
                    "Model": "ridge",
                    "TargetCloseRetPct": 1.0,
                    "PredTargetCloseRetPct": 0.1,
                    "TargetHighRetPct": 2.0,
                    "PredTargetHighRetPct": 1.8,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.9,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.7,
                },
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "ridge",
                    "TargetCloseRetPct": 1.0,
                    "PredTargetCloseRetPct": 0.2,
                    "TargetHighRetPct": 2.0,
                    "PredTargetHighRetPct": 1.6,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.7,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.3,
                },
                {
                    "Ticker": "AAA",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Model": "random_forest",
                    "TargetCloseRetPct": 1.0,
                    "PredTargetCloseRetPct": 0.95,
                    "TargetHighRetPct": 2.0,
                    "PredTargetHighRetPct": 1.9,
                    "TargetLowRetPct": -1.0,
                    "PredTargetLowRetPct": -0.95,
                    "ActualRangePct": 3.0,
                    "PredRangePct": 2.85,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    "SnapshotDate": "2026-03-30",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Ticker": "AAA",
                    "Base": 10.0,
                    "Model": "ridge",
                    "PredTargetLowRetPct": -2.0,
                    "PredTargetCloseRetPct": 1.0,
                    "PredTargetHighRetPct": 3.0,
                },
                {
                    "SnapshotDate": "2026-03-30",
                    "SnapshotTimeBucket": "PM_LATE",
                    "Ticker": "AAA",
                    "Base": 10.0,
                    "Model": "random_forest",
                    "PredTargetLowRetPct": -1.5,
                    "PredTargetCloseRetPct": 0.8,
                    "PredTargetHighRetPct": 2.5,
                },
            ]
        )

        report = select_current_intraday_rest_of_session_forecasts(history, current)

        self.assertEqual(report["Ticker"].tolist(), ["AAA"])
        self.assertEqual(str(report.iloc[0]["Model"]), "random_forest")

    def test_prepare_current_snapshot_row_relabels_bucket_to_engine_context(self) -> None:
        current_row = pd.DataFrame(
            [
                {
                    "SnapshotTs": pd.Timestamp("2026-03-30T11:25:00+07:00"),
                    "SnapshotDate": "2026-03-30",
                    "SnapshotTimeBucket": "AM_LATE",
                    "BucketCode": 2.0,
                    "TickerBucketCode": 2.0,
                    "TickerMinutesFromOpen": 145.0,
                    "TickerMinutesToClose": 95.0,
                    "TickerSessionProgressPct": (145.0 / 240.0) * 100.0,
                    "IndexBucketCode": 2.0,
                    "IndexMinutesFromOpen": 145.0,
                    "IndexMinutesToClose": 95.0,
                    "IndexSessionProgressPct": (145.0 / 240.0) * 100.0,
                }
            ]
        )

        prepared = _prepare_current_snapshot_row(
            current_row,
            engine_run_at=pd.Timestamp("2026-03-30T11:45:00+07:00"),
            current_bucket="LUNCH_BREAK",
        )

        row = prepared.iloc[0]
        self.assertEqual(str(row["SnapshotTimeBucket"]), "LUNCH_BREAK")
        self.assertAlmostEqual(float(row["BucketCode"]), 3.0)
        self.assertAlmostEqual(float(row["TickerBucketCode"]), 3.0)
        self.assertAlmostEqual(float(row["TickerMinutesFromOpen"]), 150.0)
        self.assertAlmostEqual(float(row["TickerMinutesToClose"]), 90.0)


if __name__ == "__main__":
    unittest.main()
