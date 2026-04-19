import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TestCodexTotalCapitalParsing(unittest.TestCase):
    def test_plain_integer_is_kvnd(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        self.assertEqual(parse_budget_text_to_total_capital_kvnd("5000000"), 5_000_000)

    def test_ti_variants(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        self.assertEqual(parse_budget_text_to_total_capital_kvnd("5 tỉ"), 5_000_000)
        self.assertEqual(parse_budget_text_to_total_capital_kvnd("5ty"), 5_000_000)
        self.assertEqual(parse_budget_text_to_total_capital_kvnd("1.5 tỉ"), 1_500_000)

    def test_trieu_variants(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        self.assertEqual(parse_budget_text_to_total_capital_kvnd("750 triệu"), 750_000)
        self.assertEqual(parse_budget_text_to_total_capital_kvnd("750trieu"), 750_000)
        self.assertEqual(parse_budget_text_to_total_capital_kvnd("750m"), 750_000)

    def test_nghin_variants(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        self.assertEqual(parse_budget_text_to_total_capital_kvnd("120 nghìn"), 120)
        self.assertEqual(parse_budget_text_to_total_capital_kvnd("120k"), 120)

    def test_vnd_requires_thousand_multiple(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        self.assertEqual(parse_budget_text_to_total_capital_kvnd("5000000000 vnd"), 5_000_000)
        with self.assertRaises(SystemExit):
            parse_budget_text_to_total_capital_kvnd("1234 vnd")

    def test_invalid_fails_fast(self):
        from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

        with self.assertRaises(SystemExit):
            parse_budget_text_to_total_capital_kvnd("")
        with self.assertRaises(SystemExit):
            parse_budget_text_to_total_capital_kvnd("abc")

    def test_resolve_tcbs_account_slug_prefers_explicit_value(self):
        from scripts.codex.exec_resume import resolve_tcbs_account_slug

        with mock.patch.dict("os.environ", {"TCBS_USERNAME": "0366673634"}, clear=False):
            self.assertEqual(resolve_tcbs_account_slug(" TCBS Main 01 "), "TCBS-Main-01")

    def test_resolve_tcbs_account_slug_fails_when_missing(self):
        from scripts.codex.exec_resume import resolve_tcbs_account_slug

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                resolve_tcbs_account_slug("")

    def test_clean_and_require_universe_preserves_required_inputs(self):
        from scripts.codex.exec_resume import clean_and_require_universe

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "bundle_manifest.json").write_text("{\"SchemaVersion\":1}\n", encoding="utf-8")
            (workdir / "universe.csv").write_text("Ticker\nAAA\n", encoding="utf-8")
            (workdir / "market_summary.json").write_text("{}\n", encoding="utf-8")
            (workdir / "sector_summary.csv").write_text("Sector\nTech\n", encoding="utf-8")
            (workdir / "ml_range_predictions_full_2y.csv").write_text(
                "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,5,T+5,3.0\n",
                encoding="utf-8",
            )
            (workdir / "ml_range_predictions_recent_focus.csv").write_text(
                "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,5,T+5,2.0\n",
                encoding="utf-8",
            )
            (workdir / "ml_cycle_forecast_ticker_matrix.csv").write_text(
                "Ticker,PredPeakRetPct_1M,PredPeakDays_1M,PredDrawdownPct_1M\nAAA,5.0,10,-3.0\n",
                encoding="utf-8",
            )
            (workdir / "ml_cycle_forecast_best_horizon.csv").write_text(
                "Ticker,HorizonMonths,Variant,Model,PredPeakRetPct,PredPeakDays,PredDrawdownPct\nAAA,1,full_2y,ridge,5.0,10,-3.0\n",
                encoding="utf-8",
            )
            (workdir / "ticker_playbook_best_configs.csv").write_text(
                "Ticker,StrategyFamily,StrategyLabel,RobustScore,LatestSignal\nAAA,trend_reacceleration,rule,12.0,False\n",
                encoding="utf-8",
            )
            (workdir / "ml_ohlc_next_session.csv").write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Model,EvalRows,OpenMAEPct,HighMAEPct,LowMAEPct,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore,ForecastOpen,ForecastHigh,ForecastLow,ForecastClose,ForecastCloseRetPct,ForecastRangePct,ForecastCandleBias\n"
                "2026-03-30,AAA,1,T+1,10.0,2026-03-31,ridge,25,1.0,1.2,1.1,0.9,1.5,60.0,1.05,10.1,10.4,9.8,10.3,3.0,6.0,BULLISH\n",
                encoding="utf-8",
            )
            (workdir / "human_notes.md").write_text("# notes\n", encoding="utf-8")
            (workdir / "strategy_buckets.csv").write_text(
                "Ticker,StrategyBucket,AllowNewBuy,AllowAvgDown,TargetState\n"
                "AAA,invest_normal,1,1,hold_or_add\n",
                encoding="utf-8",
            )
            (workdir / "ml_intraday_rest_of_session.csv").write_text(
                "SnapshotDate,SnapshotTimeBucket,Ticker,Base,Low,Mid,High,PredLowRetPct,PredMidRetPct,PredHighRetPct,Model,EvalRows,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore\n"
                "2026-03-30,LUNCH_BREAK,AAA,10.0,9.8,10.2,10.4,-2.0,2.0,4.0,ridge,20,0.8,1.1,65.0,0.6\n",
                encoding="utf-8",
            )
            (workdir / "ml_single_name_timing.csv").write_text(
                "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Variant,Model,EvalRows,PeakRetMAEPct,PeakDayMAE,DrawdownMAEPct,CloseMAEPct,TradeScoreMAEPct,TradeScoreHitPct,SelectionScore,PredPeakRetPct,PredPeakDay,PredDrawdownPct,PredCloseRetPct,PredPeakPrice,PredDrawdownPrice,PredClosePrice,PredRewardRisk,PredTradeScore,PredNetEdgePct,PredCapitalEfficiencyPctPerDay\n"
                "2026-03-30,AAA,5,T+5,10.0,2026-04-04,recent_focus,hist_gbm,32,1.0,1.5,1.1,0.9,1.0,62.0,2.2,4.5,3.0,-2.0,1.5,10.45,9.8,10.15,2.25,2.5,2.44,0.81\n",
                encoding="utf-8",
            )
            (workdir / "ml_entry_ladder_eval.csv").write_text(
                "SnapshotDate,Ticker,PriceRank,EntryScoreRank,LimitPrice,EntryAnchor,Base,TickSize,LotSize,EntryVsLastPct,ForecastLowT1,RangeLowBlendT5,RangeLowBlendT10,CycleDrawdownPrice,EntryVsForecastLowT1Pct,EntryVsRangeLowT5Pct,EntryVsRangeLowT10Pct,EntryVsCycleDrawdownPct,NetRetToNextClosePct,NetRetToNextHighPct,BestTimingWindow,BestTimingNetEdgePct,BestTimingCloseRetPct,BestTimingRewardRisk,CycleNetEdgePct,CycleRewardRisk,FillScoreT1,FillScoreT5,FillScoreT10,FillScoreCycle,FillScoreComposite,EntryScore\n"
                "2026-03-30,AAA,1,1,9.8,grid_below_t1,10.0,0.05,100,-2.0,9.9,9.8,9.7,9.6,-1.01,0.0,1.03,2.08,3.0,4.5,T+5,5.2,2.1,1.8,4.1,1.7,63.0,55.0,48.0,40.0,51.5,3.4\n",
                encoding="utf-8",
            )
            (workdir / "orders.csv").write_text("Ticker,Side,Quantity,LimitPrice\n", encoding="utf-8")
            (workdir / "DONE.md").write_text("done\n", encoding="utf-8")

            clean_and_require_universe(workdir)

            self.assertTrue((workdir / "bundle_manifest.json").exists())
            self.assertTrue((workdir / "universe.csv").exists())
            self.assertTrue((workdir / "market_summary.json").exists())
            self.assertTrue((workdir / "sector_summary.csv").exists())
            self.assertTrue((workdir / "ml_range_predictions_full_2y.csv").exists())
            self.assertTrue((workdir / "ml_range_predictions_recent_focus.csv").exists())
            self.assertTrue((workdir / "ml_cycle_forecast_ticker_matrix.csv").exists())
            self.assertTrue((workdir / "ml_cycle_forecast_best_horizon.csv").exists())
            self.assertTrue((workdir / "ticker_playbook_best_configs.csv").exists())
            self.assertTrue((workdir / "ml_ohlc_next_session.csv").exists())
            self.assertTrue((workdir / "human_notes.md").exists())
            self.assertTrue((workdir / "strategy_buckets.csv").exists())
            self.assertTrue((workdir / "ml_intraday_rest_of_session.csv").exists())
            self.assertTrue((workdir / "ml_single_name_timing.csv").exists())
            self.assertTrue((workdir / "ml_entry_ladder_eval.csv").exists())
            self.assertFalse((workdir / "orders.csv").exists())
            self.assertFalse((workdir / "DONE.md").exists())

    def test_clean_and_require_universe_fails_when_summary_inputs_missing(self):
        from scripts.codex.exec_resume import clean_and_require_universe

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "universe.csv").write_text("Ticker\nAAA\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                clean_and_require_universe(workdir)


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, text):
        self.writes.append(text)

    def close(self):
        self.closed = True


class _FakeStream:
    def __init__(self, proc, lines):
        self._proc = proc
        self._lines = list(lines)

    def readline(self):
        while True:
            if self._lines:
                return self._lines.pop(0)
            if self._proc.done:
                return ""
            time.sleep(0.01)


class _FakePopen:
    def __init__(self, stdout_lines=None, stderr_lines=None, returncode=0):
        self.stdin = _FakeStdin()
        self.done = False
        self.returncode = returncode
        self.terminate_called = False
        self.kill_called = False
        self.stdout = _FakeStream(self, stdout_lines or [])
        self.stderr = _FakeStream(self, stderr_lines or [])

    def poll(self):
        return self.returncode if self.done else None

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("fake wait timed out")
            time.sleep(0.01)
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.done = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.done = True
        self.returncode = -9

class TestCodexWatchdog(unittest.TestCase):
    def test_archive_codex_run_copies_orders_done_and_log(self):
        from scripts.codex.exec_resume import archive_codex_run

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / ".git").mkdir()
            archive_root = repo_root / "archives"
            output_csv = repo_root / "codex_universe" / "orders.csv"
            output_csv.parent.mkdir()
            output_csv.write_text("Ticker,Side,Quantity,LimitPrice\nAAA,BUY,100,10.0\n", encoding="utf-8")
            done_file = repo_root / "codex_universe" / "DONE.md"
            done_file.write_text("STATUS: COMPLETE\n", encoding="utf-8")
            log_file = repo_root / "out" / "codex" / "codex_session_1.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("session-log\n", encoding="utf-8")
            wrapper_log = repo_root / "wrapper.log"

            with wrapper_log.open("a", encoding="utf-8") as log_fp:
                archive_dir = archive_codex_run(
                    repo_root=repo_root,
                    archive_root_rel="archives/codex_runs",
                    output_csv=output_csv,
                    completion_file=done_file,
                    log_file=log_file,
                    thread_id="thread-123",
                    tcbs_account="acct-1",
                    log_fp=log_fp,
                )

            self.assertTrue(archive_dir.exists())
            self.assertEqual(
                (archive_dir / "orders.csv").read_text(encoding="utf-8"),
                output_csv.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (archive_dir / "codex_session_1.log").read_text(encoding="utf-8"),
                "session-log\n",
            )
            self.assertEqual(
                (archive_dir / "DONE.md").read_text(encoding="utf-8"),
                "STATUS: COMPLETE\n",
            )
            metadata = (archive_dir / "metadata.json").read_text(encoding="utf-8")
            self.assertIn('"thread_id": "thread-123"', metadata)
            self.assertIn('"done_file": "DONE.md"', metadata)
            self.assertIn('"tcbs_account": "acct-1"', metadata)

    def test_snapshot_codex_workspace_copies_full_workdir(self):
        from scripts.codex.exec_resume import snapshot_codex_workspace

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / ".git").mkdir()
            workdir = repo_root / "codex_universe"
            (workdir / "research" / "tickers" / "NVL").mkdir(parents=True)
            (workdir / "bundle_manifest.json").write_text('{"SchemaVersion":1}\n', encoding="utf-8")
            (workdir / "orders.csv").write_text(
                "Ticker,Side,Quantity,LimitPrice\nNVL,BUY,100,17.60\n",
                encoding="utf-8",
            )
            (workdir / "research" / "tickers" / "NVL" / "state.json").write_text(
                '{"Ticker":"NVL"}\n',
                encoding="utf-8",
            )
            wrapper_log = repo_root / "wrapper.log"

            with wrapper_log.open("a", encoding="utf-8") as log_fp:
                snapshot_dir = snapshot_codex_workspace(
                    repo_root=repo_root,
                    snapshot_root_rel="codex_universe_history",
                    workdir=workdir,
                    tcbs_account="acct-1",
                    stamp="20260415_120520",
                    log_fp=log_fp,
                )

            self.assertEqual(
                snapshot_dir,
                (repo_root / "codex_universe_history" / "acct-1" / "2026" / "20260415_120520").resolve(),
            )
            self.assertEqual(
                (snapshot_dir / "orders.csv").read_text(encoding="utf-8"),
                (workdir / "orders.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (snapshot_dir / "research" / "tickers" / "NVL" / "state.json").read_text(encoding="utf-8"),
                '{"Ticker":"NVL"}\n',
            )
            metadata = (snapshot_dir / "snapshot_metadata.json").read_text(encoding="utf-8")
            self.assertIn('"tcbs_account": "acct-1"', metadata)
            self.assertIn('"source_workdir"', metadata)

    def test_sync_archive_to_order_history_repo_copies_timestamped_orders_and_pushes(self):
        from scripts.codex.exec_resume import sync_archive_to_order_history_repo

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "broker-repo"
            repo_root.mkdir()
            archive_dir = repo_root / "archives" / "codex_runs" / "20260414_153000"
            archive_dir.mkdir(parents=True)
            (archive_dir / "orders.csv").write_text(
                "Ticker,Side,Quantity,LimitPrice\nNVL,BUY,100,17.10\n",
                encoding="utf-8",
            )
            (archive_dir / "DONE.md").write_text("done\n", encoding="utf-8")
            (archive_dir / "codex_session_1.log").write_text("log\n", encoding="utf-8")
            (archive_dir / "metadata.json").write_text(
                '{"thread_id":"thread-123","orders_csv":"orders.csv","done_file":"DONE.md","codex_log":"codex_session_1.log"}\n',
                encoding="utf-8",
            )
            clone_dir = tmp_path / "orders-history"
            (clone_dir / ".git").mkdir(parents=True)
            wrapper_log = tmp_path / "wrapper.log"
            calls = []

            def fake_run(cmd, cwd=None, capture_output=True, text=True):
                calls.append((cmd, cwd))
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

            with wrapper_log.open("a", encoding="utf-8") as log_fp, mock.patch(
                "scripts.codex.exec_resume.subprocess.run", side_effect=fake_run
            ):
                dest_dir = sync_archive_to_order_history_repo(
                    repo_root=repo_root,
                    archive_dir=archive_dir,
                    remote_repo="minhhai2209/tcbs-orders-history",
                    clone_dir_raw=str(clone_dir),
                    tcbs_account="acct-1",
                    gh_bin="gh",
                    log_fp=log_fp,
                )

            self.assertEqual(dest_dir, (clone_dir / "acct-1" / "2026" / "20260414_153000").resolve())
            self.assertTrue((dest_dir / "orders_20260414_153000.csv").exists())
            self.assertTrue((dest_dir / "DONE_20260414_153000.md").exists())
            metadata = (dest_dir / "metadata.json").read_text(encoding="utf-8")
            self.assertIn('"tcbs_account": "acct-1"', metadata)
            self.assertIn('"orders_csv": "orders_20260414_153000.csv"', metadata)
            clone_dir_resolved = str(clone_dir.resolve())
            self.assertTrue(any(cmd[:4] == ["git", "-C", clone_dir_resolved, "pull"] for cmd, _ in calls))
            self.assertTrue(any(cmd[:4] == ["git", "-C", clone_dir_resolved, "push"] for cmd, _ in calls))

    def test_run_codex_idle_timeout_kills_stalled_process_and_preserves_thread_id(self):
        from scripts.codex.exec_resume import run_codex

        fake_proc = _FakePopen(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"turn.started"}\n',
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "codex.log"
            with log_path.open("a", encoding="utf-8") as log_fp, mock.patch(
                "scripts.codex.exec_resume.subprocess.Popen", return_value=fake_proc
            ):
                rc, thread_id, last_agent_text, timed_out = run_codex(
                    codex_bin="codex",
                    args=["exec"],
                    prompt="hello\n",
                    cwd=Path(tmp),
                    log_fp=log_fp,
                    capture_thread_id=True,
                    idle_timeout_seconds=0.05,
                )

        self.assertTrue(timed_out)
        self.assertEqual(thread_id, "thread-123")
        self.assertIsNone(last_agent_text)
        self.assertEqual(rc, -15)
        self.assertTrue(fake_proc.terminate_called)
        self.assertFalse(fake_proc.kill_called)
        self.assertEqual(fake_proc.stdin.writes, ["hello\n"])
        self.assertTrue(fake_proc.stdin.closed)

    def test_write_resume_helper_script_writes_immediately_usable_script(self):
        from scripts.codex.exec_resume import write_resume_helper_script

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_dir = tmp_path / "out" / "codex"
            log_dir.mkdir(parents=True)
            log_path = tmp_path / "wrapper.log"
            with log_path.open("a", encoding="utf-8") as log_fp:
                script_path = write_resume_helper_script(
                    log_dir=log_dir,
                    codex_bin="codex",
                    workdir=tmp_path / "codex_universe",
                    model="gpt-5.4",
                    reasoning="high",
                    thread_id="thread-xyz",
                    log_fp=log_fp,
                )

            self.assertEqual(script_path, log_dir / "resume_last_codex.sh")
            self.assertIsNotNone(script_path)
            self.assertTrue(script_path.exists())
            body = script_path.read_text(encoding="utf-8")
            self.assertIn('"codex" \\', body)
            self.assertIn('resume thread-xyz', body)
            self.assertIn('--cd "', body)
            self.assertTrue(script_path.stat().st_mode & 0o111)

    def test_run_codex_ignores_done_file_until_process_exits(self):
        from scripts.codex.exec_resume import run_codex

        fake_proc = _FakePopen(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "DONE.md").write_text("done\n", encoding="utf-8")
            log_path = tmp_path / "codex.log"
            finisher = threading.Thread(
                target=lambda: (time.sleep(0.05), setattr(fake_proc, "done", True)),
                daemon=True,
            )
            finisher.start()
            with log_path.open("a", encoding="utf-8") as log_fp, mock.patch(
                "scripts.codex.exec_resume.subprocess.Popen", return_value=fake_proc
            ):
                rc, thread_id, last_agent_text, timed_out = run_codex(
                    codex_bin="codex",
                    args=["exec"],
                    prompt="hello\n",
                    cwd=tmp_path,
                    log_fp=log_fp,
                    capture_thread_id=True,
                    idle_timeout_seconds=0.05,
                )
            finisher.join(timeout=1)

        self.assertFalse(timed_out)
        self.assertEqual(thread_id, "thread-123")
        self.assertEqual(last_agent_text, "done")
        self.assertEqual(rc, 0)
        self.assertFalse(fake_proc.terminate_called)

    def test_run_codex_terminates_after_completion_grace_when_files_exist(self):
        from scripts.codex.exec_resume import run_codex

        fake_proc = _FakePopen(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            done_file = tmp_path / "DONE.md"
            output_csv = tmp_path / "orders.csv"
            done_file.write_text("done\n", encoding="utf-8")
            output_csv.write_text("Ticker,Side,Quantity,LimitPrice\n", encoding="utf-8")
            log_path = tmp_path / "codex.log"
            with log_path.open("a", encoding="utf-8") as log_fp, mock.patch(
                "scripts.codex.exec_resume.subprocess.Popen", return_value=fake_proc
            ):
                rc, thread_id, last_agent_text, timed_out = run_codex(
                    codex_bin="codex",
                    args=["exec"],
                    prompt="hello\n",
                    cwd=tmp_path,
                    log_fp=log_fp,
                    capture_thread_id=True,
                    idle_timeout_seconds=0,
                    completion_file=done_file,
                    output_csv=output_csv,
                    completion_grace_seconds=0.05,
                )

        self.assertFalse(timed_out)
        self.assertEqual(thread_id, "thread-123")
        self.assertEqual(last_agent_text, "done")
        self.assertEqual(rc, -15)
        self.assertTrue(fake_proc.terminate_called)

    def test_main_checks_done_between_resume_rounds(self):
        from scripts.codex import exec_resume

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "codex_universe"
            workdir.mkdir()
            prompt_path = tmp_path / "PROMPT.txt"
            prompt_path.write_text("prompt\n", encoding="utf-8")
            (workdir / "bundle_manifest.json").write_text("{\"SchemaVersion\":1}\n", encoding="utf-8")

            for name, content in {
                "universe.csv": "Ticker\nAAA\n",
                "market_summary.json": "{}\n",
                "sector_summary.csv": "Sector\nTech\n",
                "ml_range_predictions_full_2y.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_range_predictions_recent_focus.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_cycle_forecast_ticker_matrix.csv": "Ticker,PredPeakRetPct_1M,PredPeakDays_1M,PredDrawdownPct_1M\nAAA,1.0,5,-1.0\n",
                "ml_cycle_forecast_best_horizon.csv": "Ticker,HorizonMonths,Variant,Model,PredPeakRetPct,PredPeakDays,PredDrawdownPct\nAAA,1,full_2y,ridge,1.0,5,-1.0\n",
                "ticker_playbook_best_configs.csv": "Ticker,StrategyFamily,StrategyLabel,RobustScore,LatestSignal\nAAA,trend_reacceleration,rule,10.0,False\n",
                "ml_ohlc_next_session.csv": (
                    "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Model,EvalRows,"
                    "OpenMAEPct,HighMAEPct,LowMAEPct,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore,"
                    "ForecastOpen,ForecastHigh,ForecastLow,ForecastClose,ForecastCloseRetPct,ForecastRangePct,ForecastCandleBias\n"
                    "2026-03-30,AAA,1,T+1,10.0,2026-03-31,ridge,25,1.0,1.0,1.0,1.0,1.0,60.0,1.0,10.1,10.3,9.9,10.2,2.0,4.0,BULLISH\n"
                ),
            }.items():
                (workdir / name).write_text(content, encoding="utf-8")

            calls = []
            target_csv = (workdir / "orders.csv").resolve()
            done_file = (workdir / "DONE.md").resolve()

            def fake_run_codex(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    target_csv.write_text("Ticker,Side,Quantity,LimitPrice\n", encoding="utf-8")
                    return 0, "thread-123", None, False
                if len(calls) == 3:
                    done_file.write_text("done\n", encoding="utf-8")
                    return 0, "thread-123", None, False
                return 0, None, None, False

            args = SimpleNamespace(
                codex_bin="codex",
                workdir=str(workdir),
                prompt_file=str(prompt_path),
                budget_text="5 tỉ",
                total_capital_kvnd="",
                output_csv="orders.csv",
                done_file="DONE.md",
                continue_message="Tiếp tục.",
                grace_after_csv=0,
                log_dir=str(tmp_path / "out" / "codex"),
                model="gpt-5.4",
                reasoning="high",
                archive_root="archives/codex_runs",
                archive_enabled=1,
                archive_git_commit=1,
                tracked_snapshot_enabled=1,
                tracked_snapshot_root="codex_universe_history",
                order_history_enabled=0,
                order_history_repo="",
                order_history_clone_dir="../tcbs-orders-history",
                order_history_gh_bin="gh",
                tcbs_account="acct-1",
                idle_timeout_seconds=1,
                max_wall_seconds=10,
                max_continues=0,
            )

            with mock.patch("scripts.codex.exec_resume.parse_args", return_value=args), mock.patch(
                "scripts.codex.exec_resume.detect_json_flag", return_value="--json"
            ), mock.patch(
                "scripts.codex.exec_resume.run_codex", side_effect=fake_run_codex
            ), mock.patch(
                "scripts.codex.exec_resume.write_resume_helper_script", return_value=tmp_path / "resume.sh"
            ), mock.patch(
                "scripts.codex.exec_resume.find_git_root", return_value=tmp_path
            ), mock.patch(
                "scripts.codex.exec_resume.archive_codex_run", return_value=tmp_path / "archives" / "run1"
            ) as archive_mock, mock.patch(
                "scripts.codex.exec_resume.snapshot_codex_workspace"
            ) as snapshot_mock, mock.patch(
                "scripts.codex.exec_resume.git_commit_archive"
            ):
                exec_resume.main()

        self.assertEqual(len(calls), 3)
        self.assertNotIn("stop_file", calls[0])
        self.assertNotIn("grace_after_stop_file", calls[0])
        archive_mock.assert_called_once()
        snapshot_mock.assert_called_once()

    def test_main_syncs_order_history_repo_when_enabled(self):
        from scripts.codex import exec_resume

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "codex_universe"
            workdir.mkdir()
            prompt_path = tmp_path / "PROMPT.txt"
            prompt_path.write_text("prompt\n", encoding="utf-8")
            (workdir / "bundle_manifest.json").write_text("{\"SchemaVersion\":1}\n", encoding="utf-8")

            for name, content in {
                "universe.csv": "Ticker\nAAA\n",
                "market_summary.json": "{}\n",
                "sector_summary.csv": "Sector\nTech\n",
                "ml_range_predictions_full_2y.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_range_predictions_recent_focus.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_cycle_forecast_ticker_matrix.csv": "Ticker,PredPeakRetPct_1M,PredPeakDays_1M,PredDrawdownPct_1M\nAAA,1.0,5,-1.0\n",
                "ml_cycle_forecast_best_horizon.csv": "Ticker,HorizonMonths,Variant,Model,PredPeakRetPct,PredPeakDays,PredDrawdownPct\nAAA,1,full_2y,ridge,1.0,5,-1.0\n",
                "ticker_playbook_best_configs.csv": "Ticker,StrategyFamily,StrategyLabel,RobustScore,LatestSignal\nAAA,trend_reacceleration,rule,10.0,False\n",
                "ml_ohlc_next_session.csv": (
                    "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Model,EvalRows,"
                    "OpenMAEPct,HighMAEPct,LowMAEPct,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore,"
                    "ForecastOpen,ForecastHigh,ForecastLow,ForecastClose,ForecastCloseRetPct,ForecastRangePct,ForecastCandleBias\n"
                    "2026-03-30,AAA,1,T+1,10.0,2026-03-31,ridge,25,1.0,1.0,1.0,1.0,1.0,60.0,1.0,10.1,10.3,9.9,10.2,2.0,4.0,BULLISH\n"
                ),
            }.items():
                (workdir / name).write_text(content, encoding="utf-8")

            target_csv = (workdir / "orders.csv").resolve()
            done_file = (workdir / "DONE.md").resolve()

            def fake_run_codex(**kwargs):
                target_csv.write_text("Ticker,Side,Quantity,LimitPrice\n", encoding="utf-8")
                done_file.write_text("done\n", encoding="utf-8")
                return 0, "thread-123", None, False

            args = SimpleNamespace(
                codex_bin="codex",
                workdir=str(workdir),
                prompt_file=str(prompt_path),
                budget_text="5 tỉ",
                total_capital_kvnd="",
                output_csv="orders.csv",
                done_file="DONE.md",
                continue_message="Tiếp tục.",
                grace_after_csv=0,
                log_dir=str(tmp_path / "out" / "codex"),
                model="gpt-5.4",
                reasoning="high",
                archive_root="archives/codex_runs",
                archive_enabled=1,
                archive_git_commit=0,
                tracked_snapshot_enabled=1,
                tracked_snapshot_root="codex_universe_history",
                order_history_enabled=1,
                order_history_repo="minhhai2209/tcbs-orders-history",
                order_history_clone_dir="../tcbs-orders-history",
                order_history_gh_bin="gh",
                tcbs_account="acct-1",
                idle_timeout_seconds=1,
                max_wall_seconds=10,
                max_continues=0,
            )

            with mock.patch("scripts.codex.exec_resume.parse_args", return_value=args), mock.patch(
                "scripts.codex.exec_resume.detect_json_flag", return_value="--json"
            ), mock.patch(
                "scripts.codex.exec_resume.run_codex", side_effect=fake_run_codex
            ), mock.patch(
                "scripts.codex.exec_resume.write_resume_helper_script", return_value=tmp_path / "resume.sh"
            ), mock.patch(
                "scripts.codex.exec_resume.find_git_root", return_value=tmp_path
            ), mock.patch(
                "scripts.codex.exec_resume.archive_codex_run", return_value=tmp_path / "archives" / "run1"
            ), mock.patch(
                "scripts.codex.exec_resume.snapshot_codex_workspace"
            ) as snapshot_mock, mock.patch(
                "scripts.codex.exec_resume.sync_archive_to_order_history_repo"
            ) as sync_mock:
                exec_resume.main()

        snapshot_mock.assert_called_once()
        sync_mock.assert_called_once()

    def test_main_fails_if_done_file_exists_without_output_csv(self):
        from scripts.codex import exec_resume

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workdir = tmp_path / "codex_universe"
            workdir.mkdir()
            prompt_path = tmp_path / "PROMPT.txt"
            prompt_path.write_text("prompt\n", encoding="utf-8")
            (workdir / "bundle_manifest.json").write_text("{\"SchemaVersion\":1}\n", encoding="utf-8")

            for name, content in {
                "universe.csv": "Ticker\nAAA\n",
                "market_summary.json": "{}\n",
                "sector_summary.csv": "Sector\nTech\n",
                "ml_range_predictions_full_2y.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_range_predictions_recent_focus.csv": "Ticker,Horizon,ForecastWindow,PredMidRetPct\nAAA,1,T+1,1.0\n",
                "ml_cycle_forecast_ticker_matrix.csv": "Ticker,PredPeakRetPct_1M,PredPeakDays_1M,PredDrawdownPct_1M\nAAA,1.0,5,-1.0\n",
                "ml_cycle_forecast_best_horizon.csv": "Ticker,HorizonMonths,Variant,Model,PredPeakRetPct,PredPeakDays,PredDrawdownPct\nAAA,1,full_2y,ridge,1.0,5,-1.0\n",
                "ticker_playbook_best_configs.csv": "Ticker,StrategyFamily,StrategyLabel,RobustScore,LatestSignal\nAAA,trend_reacceleration,rule,10.0,False\n",
                "ml_ohlc_next_session.csv": (
                    "SnapshotDate,Ticker,Horizon,ForecastWindow,Base,ForecastDate,Model,EvalRows,"
                    "OpenMAEPct,HighMAEPct,LowMAEPct,CloseMAEPct,RangeMAEPct,CloseDirHitPct,SelectionScore,"
                    "ForecastOpen,ForecastHigh,ForecastLow,ForecastClose,ForecastCloseRetPct,ForecastRangePct,ForecastCandleBias\n"
                    "2026-03-30,AAA,1,T+1,10.0,2026-03-31,ridge,25,1.0,1.0,1.0,1.0,1.0,60.0,1.0,10.1,10.3,9.9,10.2,2.0,4.0,BULLISH\n"
                ),
            }.items():
                (workdir / name).write_text(content, encoding="utf-8")

            done_file = (workdir / "DONE.md").resolve()

            def fake_run_codex(**kwargs):
                done_file.write_text("done\n", encoding="utf-8")
                return 0, "thread-123", None, False

            args = SimpleNamespace(
                codex_bin="codex",
                workdir=str(workdir),
                prompt_file=str(prompt_path),
                budget_text="5 tỉ",
                total_capital_kvnd="",
                output_csv="orders.csv",
                done_file="DONE.md",
                continue_message="Tiếp tục.",
                grace_after_csv=0,
                log_dir=str(tmp_path / "out" / "codex"),
                model="gpt-5.4",
                reasoning="high",
                archive_root="archives/codex_runs",
                archive_enabled=1,
                archive_git_commit=1,
                tracked_snapshot_enabled=1,
                tracked_snapshot_root="codex_universe_history",
                order_history_enabled=0,
                order_history_repo="",
                order_history_clone_dir="../tcbs-orders-history",
                order_history_gh_bin="gh",
                tcbs_account="acct-1",
                idle_timeout_seconds=1,
                max_wall_seconds=10,
                max_continues=0,
            )

            with mock.patch("scripts.codex.exec_resume.parse_args", return_value=args), mock.patch(
                "scripts.codex.exec_resume.detect_json_flag", return_value="--json"
            ), mock.patch(
                "scripts.codex.exec_resume.run_codex", side_effect=fake_run_codex
            ), mock.patch(
                "scripts.codex.exec_resume.write_resume_helper_script", return_value=tmp_path / "resume.sh"
            ):
                with self.assertRaises(SystemExit) as ctx:
                    exec_resume.main()

        self.assertIn("completion file 'DONE.md' exists but output CSV 'orders.csv' is missing", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
