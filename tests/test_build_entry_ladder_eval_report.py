from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.build_entry_ladder_eval_report import (
    FILL_FEATURE_COLUMNS,
    REQUIRED_OUTPUT_COLUMNS,
    _best_timing_metrics_for_entry,
    _build_candidate_price_map,
    _entry_fill_multiplier,
    _predict_fill_score,
    _snap_buy_price,
    run_report,
    select_best_fill_models,
)
from scripts.analysis.evaluate_ohlc_models import FEATURE_COLUMNS


class BuildEntryLadderEvalReportTest(unittest.TestCase):
    def _write_daily_history(self, path: Path, *, start: str, periods: int, base: float) -> None:
        dates = pd.date_range(start=start, periods=periods, freq="B")
        rows = []
        for idx, date in enumerate(dates):
            close = base + (0.15 * idx)
            rows.append(
                {
                    "t": int(date.timestamp()),
                    "open": round(close - 0.2, 4),
                    "high": round(close + 0.4, 4),
                    "low": round(close - 0.5, 4),
                    "close": round(close, 4),
                    "volume": 1_000_000 + (idx * 1000),
                    "date_vn": date.strftime("%Y-%m-%d"),
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_snap_buy_price_floors_to_tick_and_clamps_to_band(self) -> None:
        self.assertEqual(_snap_buy_price(25.37, 0.05, 24.8, 28.5), 25.35)
        self.assertEqual(_snap_buy_price(24.71, 0.05, 24.8, 28.5), 24.8)

    def test_best_timing_metrics_improves_when_entry_is_deeper(self) -> None:
        timing = pd.DataFrame(
            [
                {
                    "ForecastWindow": "T+3",
                    "PredPeakPrice": 103.0,
                    "PredDrawdownPrice": 97.0,
                    "PredClosePrice": 101.0,
                    "PredPeakDay": 2.0,
                },
                {
                    "ForecastWindow": "T+5",
                    "PredPeakPrice": 104.0,
                    "PredDrawdownPrice": 96.0,
                    "PredClosePrice": 102.0,
                    "PredPeakDay": 3.0,
                },
            ]
        )

        shallow = _best_timing_metrics_for_entry(timing, 100.0)
        deep = _best_timing_metrics_for_entry(timing, 97.0)

        self.assertGreater(float(deep["BestTimingNetEdgePct"]), float(shallow["BestTimingNetEdgePct"]))
        self.assertGreater(
            float(deep["BestTimingCapitalEfficiencyPctPerDay"]),
            float(shallow["BestTimingCapitalEfficiencyPctPerDay"]),
        )

    def test_build_candidate_price_map_dedupes_and_orders_prices(self) -> None:
        universe_row = pd.Series(
            {
                "TickSize": 0.05,
                "Floor": 24.8,
                "Ceil": 28.5,
                "FloorValid": True,
                "CeilValid": True,
                "Last": 26.65,
                "ATR14": 0.75,
                "ValidBid1": 26.6,
                "GridBelow_T1": 26.6,
                "GridBelow_T2": 26.55,
                "GridBelow_T3": 26.5,
            }
        )

        entries = _build_candidate_price_map(
            universe_row,
            forecast_low_t1=26.37,
            range_low_t5=26.34,
            range_low_t10=26.18,
            cycle_drawdown_price=25.54,
        )

        prices = [price for price, _ in entries]
        self.assertEqual(prices, sorted(prices, reverse=True))
        self.assertEqual(len(prices), len(set(prices)))
        self.assertIn(26.6, prices)
        self.assertIn(25.5, prices)

    def test_select_best_fill_models_makes_deeper_entry_harder_to_fill(self) -> None:
        dates = pd.date_range("2025-01-01", periods=140, freq="B")
        rows = []
        for index, date in enumerate(dates):
            depth_threshold = 1.0 + (0.25 if index % 5 == 0 else 0.0)
            for depth_atr in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75):
                row = {column: 0.0 for column in FILL_FEATURE_COLUMNS}
                row.update(
                    {
                        "Date": date,
                        "Ticker": "AAA",
                        "Horizon": 5,
                        "BaseClose": 100.0,
                        "ATR14": 2.0,
                        "TickerATR14Pct": 2.0,
                        "EntryDepthAtr": depth_atr,
                        "EntryVsBasePct": -depth_atr * 2.0,
                        "TargetTouch": float(depth_atr <= depth_threshold),
                    }
                )
                rows.append(row)
        sample = pd.DataFrame(rows)

        models, metrics_df, selected_df = select_best_fill_models(sample, min_train_dates=60, holdout_dates=20)

        self.assertIn(("AAA", 5), models)
        self.assertFalse(metrics_df.empty)
        self.assertIn(str(selected_df.iloc[0]["Model"]), {"logistic_balanced", "hist_gbm"})

        live_feature_row = pd.Series({column: 0.0 for column in FEATURE_COLUMNS} | {"TickerATR14Pct": 2.0})
        shallow = _predict_fill_score(
            models,
            ticker="AAA",
            horizon=5,
            live_feature_row=live_feature_row,
            entry_price=99.0,
            base_price=100.0,
            atr_abs=2.0,
            heuristic_reference=99.0,
            heuristic_scale=0.5,
        )
        deep = _predict_fill_score(
            models,
            ticker="AAA",
            horizon=5,
            live_feature_row=live_feature_row,
            entry_price=96.5,
            base_price=100.0,
            atr_abs=2.0,
            heuristic_reference=96.5,
            heuristic_scale=0.5,
        )
        self.assertGreater(shallow, deep)

    def test_entry_fill_multiplier_penalises_low_fill_without_artificial_floor(self) -> None:
        self.assertAlmostEqual(_entry_fill_multiplier(80.0), 0.8)
        self.assertAlmostEqual(_entry_fill_multiplier(25.0), 0.25)
        self.assertEqual(_entry_fill_multiplier(float("nan")), 0.0)

        shallow_score = 7.0 * _entry_fill_multiplier(80.0)
        deep_score = 14.0 * _entry_fill_multiplier(25.0)
        self.assertGreater(shallow_score, deep_score)

    def test_run_report_writes_required_columns_and_ranks_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history_dir = tmp_path / "data"
            universe_csv = tmp_path / "universe.csv"
            ohlc_csv = tmp_path / "ml_ohlc_next_session.csv"
            range_full_csv = tmp_path / "ml_range_predictions_full_2y.csv"
            range_recent_csv = tmp_path / "ml_range_predictions_recent_focus.csv"
            cycle_csv = tmp_path / "ml_cycle_forecast_best_horizon.csv"
            single_name_csv = tmp_path / "ml_single_name_timing.csv"
            output_dir = tmp_path / "out"
            history_dir.mkdir(parents=True, exist_ok=True)

            self._write_daily_history(history_dir / "AAA_daily.csv", start="2025-01-01", periods=90, base=100.0)
            self._write_daily_history(history_dir / "VNINDEX_daily.csv", start="2025-01-01", periods=90, base=1200.0)

            universe_csv.write_text(
                "Ticker,EngineRunAt,Last,TickSize,LotSize,Floor,Ceil,FloorValid,CeilValid,ValidBid1,GridBelow_T1,GridBelow_T2,GridBelow_T3,ATR14,ATR14Pct\n"
                "AAA,2026-04-06T08:00:00+07:00,100.0,0.05,100,93.0,107.0,True,True,99.5,99.0,98.5,98.0,2.0,2.0\n",
                encoding="utf-8",
            )
            ohlc_csv.write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Model,EvalRows,OpenMAEPct,HighMAEPct,LowMAEPct,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore,ForecastOpen,ForecastHigh,ForecastLow,ForecastClose,ForecastCloseRetPct,ForecastRangePct,ForecastCandleBias\n"
                "2026-04-05,AAA,1,T+1,100.0,2026-04-06,ridge,30,1,1,1,1,1,60,1.0,100.2,101.8,98.8,100.8,0.8,3.0,BULLISH\n",
                encoding="utf-8",
            )
            range_full_csv.write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,Low,Mid,High,PredLowRetPct,PredMidRetPct,PredHighRetPct,RecentFocusWeight,Full2YWeight,CloseMAEPct,RangeMAEPct,CloseDirHitPct\n"
                "2026-04-05,AAA,5,T+5,100.0,98.5,101.0,103.0,-1.5,1.0,3.0,0.5,0.5,1.0,1.0,55.0\n"
                "2026-04-05,AAA,10,T+10,100.0,97.5,102.0,105.0,-2.5,2.0,5.0,0.5,0.5,1.0,1.0,55.0\n",
                encoding="utf-8",
            )
            range_recent_csv.write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,Low,Mid,High,PredLowRetPct,PredMidRetPct,PredHighRetPct,RecentFocusWeight,Full2YWeight,CloseMAEPct,RangeMAEPct,CloseDirHitPct\n"
                "2026-04-05,AAA,5,T+5,100.0,98.0,101.5,103.5,-2.0,1.5,3.5,0.6,0.4,1.0,1.0,55.0\n"
                "2026-04-05,AAA,10,T+10,100.0,96.5,102.5,105.5,-3.5,2.5,5.5,0.6,0.4,1.0,1.0,55.0\n",
                encoding="utf-8",
            )
            cycle_csv.write_text(
                "SnapshotDate,Ticker,HorizonMonths,HorizonDays,ForecastWindow,BaseClose,Variant,Model,EvalRows,PeakRetMAEPct,PeakDayMAE,DrawdownMAEPct,SelectionScore,PredPeakRetPct,PredPeakDays,PredPeakPrice,PredDrawdownPct,PredDrawdownPrice\n"
                "2026-04-05,AAA,1,21,1M,100.0,full_2y,ridge,30,1.0,2.0,1.0,3.0,4.5,8.0,104.5,-3.0,97.0\n",
                encoding="utf-8",
            )
            single_name_csv.write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Variant,Model,EvalRows,PeakRetMAEPct,PeakDayMAE,DrawdownMAEPct,CloseMAEPct,TradeScoreMAEPct,TradeScoreHitPct,SelectionScore,PredPeakRetPct,PredPeakDay,PredDrawdownPct,PredCloseRetPct,PredPeakPrice,PredDrawdownPrice,PredClosePrice,PredRewardRisk,PredTradeScore,PredNetEdgePct,PredCapitalEfficiencyPctPerDay\n"
                "2026-04-05,AAA,3,T+3,100.0,2026-04-08,full_2y,ridge,30,1,1,1,1,1,60,2.0,3.0,2.0,-2.0,0.8,103.0,98.0,100.8,1.5,1.0,0.94,0.47\n"
                "2026-04-05,AAA,5,T+5,100.0,2026-04-10,full_2y,ridge,30,1,1,1,1,1,60,2.0,5.0,3.0,-3.0,1.8,105.0,97.0,101.8,1.66,2.0,1.94,0.65\n",
                encoding="utf-8",
            )

            result = run_report(
                history_dir=history_dir,
                universe_csv=universe_csv,
                ohlc_csv=ohlc_csv,
                range_full_csv=range_full_csv,
                range_recent_csv=range_recent_csv,
                cycle_csv=cycle_csv,
                single_name_csv=single_name_csv,
                output_dir=output_dir,
            )

            self.assertEqual(int(result["ticker_count"]), 1)
            report = pd.read_csv(output_dir / "ml_entry_ladder_eval.csv")
            for column in REQUIRED_OUTPUT_COLUMNS:
                self.assertIn(column, report.columns)
            self.assertGreaterEqual(int(report.shape[0]), 4)
            best = report.sort_values("EntryScoreRank").iloc[0]
            worst = report.sort_values("EntryScoreRank").iloc[-1]
            self.assertGreater(float(best["EntryScore"]), float(worst["EntryScore"]))


if __name__ == "__main__":
    unittest.main()
