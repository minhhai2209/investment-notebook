#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone


def log(line: str, fp):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    msg = f"[{ts}] {line}"
    print(msg)
    fp.write(msg + "\n")
    fp.flush()


def write_resume_helper_script(
    *,
    log_dir: pathlib.Path,
    codex_bin: str,
    workdir: pathlib.Path,
    model: str,
    reasoning: str,
    thread_id: str,
    log_fp,
) -> pathlib.Path | None:
    resume_script = log_dir / "resume_last_codex.sh"
    try:
        resume_script.write_text(
            "#!/usr/bin/env bash\n"
            f"\"{codex_bin}\" \\\n"
            "  --sandbox danger-full-access \\\n"
            f"  --cd \"{workdir}\" \\\n"
            f"  --model \"{model}\" \\\n"
            f"  --config 'model_reasoning_effort=\"{reasoning}\"' \\\n"
            f"  resume {thread_id}\n",
            encoding="utf-8",
        )
        resume_script.chmod(0o755)
        log(f"resume helper script written: {resume_script}", log_fp)
        return resume_script
    except Exception as exc:  # pragma: no cover - best effort
        log(f"WARNING: failed to write resume script {resume_script}: {exc}", log_fp)
        return None


def find_git_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".git").exists():
            return candidate
    raise SystemExit(f"[codex] ERROR: could not find git root from '{start}'")


def archive_codex_run(
    *,
    repo_root: pathlib.Path,
    archive_root_rel: str,
    output_csv: pathlib.Path,
    completion_file: pathlib.Path | None,
    log_file: pathlib.Path,
    thread_id: str | None,
    tcbs_account: str | None,
    log_fp,
) -> pathlib.Path:
    archive_root = (repo_root / archive_root_rel).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / stamp
    suffix = 2
    while archive_dir.exists():
        archive_dir = archive_root / f"{stamp}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    archived_csv = archive_dir / output_csv.name
    archived_log = archive_dir / log_file.name
    shutil.copy2(output_csv, archived_csv)
    shutil.copy2(log_file, archived_log)
    archived_done = None
    if completion_file is not None and completion_file.exists():
        archived_done = archive_dir / completion_file.name
        shutil.copy2(completion_file, archived_done)

    metadata = {
        "archived_at_local": datetime.now().isoformat(timespec="seconds"),
        "orders_csv": archived_csv.name,
        "codex_log": archived_log.name,
        "thread_id": thread_id,
    }
    if tcbs_account:
        metadata["tcbs_account"] = tcbs_account
    if archived_done is not None:
        metadata["done_file"] = archived_done.name
    (archive_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"archived codex artifacts -> {archive_dir}", log_fp)
    return archive_dir


def snapshot_codex_workspace(
    *,
    repo_root: pathlib.Path,
    snapshot_root_rel: str,
    workdir: pathlib.Path,
    tcbs_account: str,
    stamp: str | None,
    log_fp,
) -> pathlib.Path:
    snapshot_root = (repo_root / snapshot_root_rel).resolve()
    safe_account = sanitize_path_segment(tcbs_account)
    if not safe_account:
        raise SystemExit("[codex] ERROR: TCBS account slug is required for tracked codex_universe snapshots")

    stamp_value = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    year = stamp_value[:4] if re.match(r"^\d{8}_\d{6}", stamp_value) else datetime.now().strftime("%Y")
    snapshot_dir = snapshot_root / safe_account / year / stamp_value
    suffix = 2
    while snapshot_dir.exists():
        snapshot_dir = snapshot_root / safe_account / year / f"{stamp_value}_{suffix}"
        suffix += 1

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workdir, snapshot_dir)

    snapshot_metadata = {
        "copied_at_local": datetime.now().isoformat(timespec="seconds"),
        "tcbs_account": safe_account,
        "source_workdir": str(workdir),
        "snapshot_root": str(snapshot_root),
    }
    (snapshot_dir / "snapshot_metadata.json").write_text(
        json.dumps(snapshot_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"tracked codex_universe snapshot -> {snapshot_dir}", log_fp)
    return snapshot_dir


def git_commit_archive(
    *,
    repo_root: pathlib.Path,
    archive_dir: pathlib.Path,
    log_fp,
) -> None:
    rel = archive_dir.relative_to(repo_root)
    add = subprocess.run(
        ["git", "-C", str(repo_root), "add", "--", str(rel)],
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        raise SystemExit(
            f"[codex] ERROR: git add failed for archive '{rel}': {(add.stderr or add.stdout).strip()}"
        )

    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short", "--", str(rel)],
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise SystemExit(
            f"[codex] ERROR: git status failed for archive '{rel}': {(status.stderr or status.stdout).strip()}"
        )
    if not status.stdout.strip():
        log(f"archive already clean in git index: {rel}", log_fp)
        return

    message = f"Archive Codex run {archive_dir.name}"
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "commit", "-m", message, "--", str(rel)],
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise SystemExit(
            f"[codex] ERROR: git commit failed for archive '{rel}': {(commit.stderr or commit.stdout).strip()}"
        )
    summary = (commit.stdout or commit.stderr or "").strip()
    if summary:
        log(f"git commit ok for archive {rel}\n{summary}", log_fp)
    else:
        log(f"git commit ok for archive {rel}", log_fp)


def sanitize_path_segment(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    slug = slug.strip(".-_")
    return slug


def resolve_tcbs_account_slug(explicit_value: str | None = None) -> str:
    raw = (
        (explicit_value or "").strip()
        or os.environ.get("TCBS_ACCOUNT_SLUG", "").strip()
        or os.environ.get("TCBS_USERNAME", "").strip()
    )
    slug = sanitize_path_segment(raw)
    if not slug:
        raise SystemExit(
            "[codex] ERROR: TCBS account slug is required for order-history sync. "
            "Set TCBS_ACCOUNT_SLUG or TCBS_USERNAME."
        )
    return slug


def resolve_repo_relative_path(repo_root: pathlib.Path, raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def run_checked_subprocess(
    cmd: list[str],
    *,
    cwd: pathlib.Path | None = None,
    desc: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc
    detail = (proc.stderr or proc.stdout or "").strip()
    if not detail:
        detail = f"exit code {proc.returncode}"
    raise SystemExit(f"[codex] ERROR: {desc} failed: {detail}")


def sync_archive_to_order_history_repo(
    *,
    repo_root: pathlib.Path,
    archive_dir: pathlib.Path,
    remote_repo: str,
    clone_dir_raw: str,
    tcbs_account: str,
    gh_bin: str,
    log_fp,
) -> pathlib.Path:
    remote_repo = remote_repo.strip()
    if not remote_repo:
        raise SystemExit(
            "[codex] ERROR: CODEX_ORDER_HISTORY_REPO is required when order-history sync is enabled"
        )

    clone_dir = resolve_repo_relative_path(repo_root, clone_dir_raw)
    clone_parent = clone_dir.parent
    clone_parent.mkdir(parents=True, exist_ok=True)

    if clone_dir.exists():
        if not (clone_dir / ".git").exists():
            raise SystemExit(
                f"[codex] ERROR: order-history clone dir exists but is not a git repo: {clone_dir}"
            )
        run_checked_subprocess(
            ["git", "-C", str(clone_dir), "pull", "--ff-only"],
            desc=f"git pull order-history repo at {clone_dir}",
        )
        log(f"updated order-history repo: {clone_dir}", log_fp)
    else:
        run_checked_subprocess(
            [gh_bin, "repo", "clone", remote_repo, str(clone_dir)],
            cwd=clone_parent,
            desc=f"{gh_bin} repo clone {remote_repo}",
        )
        log(f"cloned order-history repo -> {clone_dir}", log_fp)

    stamp = archive_dir.name
    year = stamp[:4] if re.match(r"^\d{8}_\d{6}", stamp) else datetime.now().strftime("%Y")
    dest_dir = clone_dir / tcbs_account / year / stamp
    if dest_dir.exists():
        raise SystemExit(
            f"[codex] ERROR: order-history destination already exists, refusing to overwrite: {dest_dir}"
        )
    dest_dir.mkdir(parents=True, exist_ok=False)

    local_metadata_path = archive_dir / "metadata.json"
    local_metadata: dict[str, object] = {}
    if local_metadata_path.exists():
        local_metadata = json.loads(local_metadata_path.read_text(encoding="utf-8"))

    copied_orders_name = None
    copied_done_name = None
    copied_log_name = None
    for src in sorted(archive_dir.iterdir()):
        if not src.is_file() or src.name == "metadata.json":
            continue
        if src.name == "orders.csv":
            dest_name = f"orders_{stamp}.csv"
            copied_orders_name = dest_name
        elif src.name == "DONE.md":
            dest_name = f"DONE_{stamp}.md"
            copied_done_name = dest_name
        else:
            dest_name = src.name
            if copied_log_name is None and src.suffix == ".log":
                copied_log_name = dest_name
        shutil.copy2(src, dest_dir / dest_name)

    if copied_log_name is None and isinstance(local_metadata.get("codex_log"), str):
        copied_log_name = str(local_metadata["codex_log"])

    synced_metadata = {
        **local_metadata,
        "tcbs_account": tcbs_account,
        "remote_repo": remote_repo,
        "external_synced_at_local": datetime.now().isoformat(timespec="seconds"),
        "source_archive_dir": str(archive_dir),
    }
    if copied_orders_name:
        synced_metadata["orders_csv"] = copied_orders_name
    if copied_done_name:
        synced_metadata["done_file"] = copied_done_name
    if copied_log_name:
        synced_metadata["codex_log"] = copied_log_name
    (dest_dir / "metadata.json").write_text(
        json.dumps(synced_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rel = dest_dir.relative_to(clone_dir)
    run_checked_subprocess(
        ["git", "-C", str(clone_dir), "add", "--", str(rel)],
        desc=f"git add order-history payload {rel}",
    )
    status = run_checked_subprocess(
        ["git", "-C", str(clone_dir), "status", "--short", "--", str(rel)],
        desc=f"git status order-history payload {rel}",
    )
    if not status.stdout.strip():
        log(f"order-history payload already clean: {rel}", log_fp)
        return dest_dir

    message = f"Archive orders {tcbs_account} {stamp}"
    commit = run_checked_subprocess(
        ["git", "-C", str(clone_dir), "commit", "-m", message, "--", str(rel)],
        desc=f"git commit order-history payload {rel}",
    )
    summary = (commit.stdout or commit.stderr or "").strip()
    if summary:
        log(f"git commit ok for order-history {rel}\n{summary}", log_fp)
    else:
        log(f"git commit ok for order-history {rel}", log_fp)

    push = run_checked_subprocess(
        ["git", "-C", str(clone_dir), "push", "origin", "HEAD"],
        desc=f"git push order-history payload {rel}",
    )
    push_summary = (push.stdout or push.stderr or "").strip()
    if push_summary:
        log(f"git push ok for order-history {rel}\n{push_summary}", log_fp)
    else:
        log(f"git push ok for order-history {rel}", log_fp)
    return dest_dir


def clean_and_require_universe(workdir: pathlib.Path) -> None:
    if not workdir.is_dir():
        raise SystemExit(f"[codex] ERROR: workdir '{workdir}' not found")
    required_inputs = {
        "bundle_manifest.json",
        "universe.csv",
        "market_summary.json",
        "sector_summary.csv",
        "ml_range_predictions_full_2y.csv",
        "ml_range_predictions_recent_focus.csv",
        "ml_cycle_forecast_ticker_matrix.csv",
        "ml_cycle_forecast_best_horizon.csv",
        "ticker_playbook_best_configs.csv",
        "ml_ohlc_next_session.csv",
    }
    optional_keep = {
        "human_notes.md",
        "ml_intraday_rest_of_session.csv",
        "ml_single_name_timing.csv",
        "ml_entry_ladder_eval.csv",
        "strategy_buckets.csv",
    }
    files = list(workdir.iterdir())
    found_inputs: set[str] = set()
    for p in files:
        if p.is_file():
            if p.name in required_inputs:
                found_inputs.add(p.name)
            elif p.name in optional_keep:
                continue
            else:
                p.unlink()
        elif p.is_dir():
            # best-effort: leave dirs untouched to avoid unexpected deletions
            continue
    missing_inputs = sorted(required_inputs - found_inputs)
    if missing_inputs:
        raise SystemExit(
            f"[codex] ERROR: expected input files in {workdir}: {', '.join(missing_inputs)}"
        )


def detect_json_flag(codex_bin: str) -> str | None:
    try:
        out = subprocess.run([codex_bin, "exec", "--help"], capture_output=True, text=True).stdout
    except Exception:
        return "--json"  # best effort
    if "--json" in out:
        return "--json"
    if "--experimental-json" in out:
        return "--experimental-json"
    return None


def parse_budget_text_to_total_capital_kvnd(budget_text: str) -> int:
    """
    Parse a human-friendly budget string into an integer in kVND.

    Supported examples (case/spacing-insensitive):
      - "5 tỉ", "5 ty", "5 billion"       -> 5_000_000 kVND
      - "750 triệu", "750 trieu", "750m"  -> 750_000 kVND
      - "120 nghìn", "120 ngan", "120k"   -> 120 kVND
      - "5000000"                         -> 5_000_000 kVND (assumed already kVND)

    Fail-fast on ambiguity or unsupported formats.
    """
    raw = (budget_text or "").strip()
    if not raw:
        raise SystemExit("[codex] ERROR: empty CODEX_BUDGET_TEXT; cannot derive total_capital_kVND.txt")

    s = raw.lower().strip()

    # Fast-path: plain integer => assume already kVND.
    if s.replace("_", "").isdigit():
        v = int(s.replace("_", ""))
        if v <= 0:
            raise SystemExit("[codex] ERROR: budget must be > 0 (kVND)")
        return v

    # Accept both "<number> <unit>" and compact "<number><unit>" forms.
    parts = s.split()
    if len(parts) >= 2:
        num_raw = parts[0]
        unit_raw = "".join(parts[1:])
    else:
        compact = s.replace(" ", "")
        m = re.match(r"^([0-9][0-9_.,]*)([^\d].*)$", compact)
        if not m:
            raise SystemExit(
                f"[codex] ERROR: cannot parse CODEX_BUDGET_TEXT='{raw}'. "
                "Provide CODEX_TOTAL_CAPITAL_KVND as an integer (kVND), or use a supported unit like 'tỉ'/'triệu'."
            )
        num_raw, unit_raw = m.group(1), m.group(2)

    num_str = num_raw.replace("_", "")
    if "," in num_str and "." not in num_str:
        # Vietnamese-style decimal comma
        num_str = num_str.replace(",", ".")
    else:
        # Treat commas as thousands separators
        num_str = num_str.replace(",", "")

    unit = unit_raw  # tolerate "t i" (rare) by joining
    try:
        num = Decimal(num_str)
    except InvalidOperation:
        raise SystemExit(
            f"[codex] ERROR: cannot parse numeric value from CODEX_BUDGET_TEXT='{raw}'. "
            "Provide CODEX_TOTAL_CAPITAL_KVND as an integer (kVND)."
        )

    if num <= 0:
        raise SystemExit("[codex] ERROR: budget must be > 0")

    # Map unit -> multiplier to kVND.
    # kVND means thousand VND.
    unit = unit.replace("đ", "vnd")
    unit = unicodedata.normalize("NFKD", unit)
    unit = "".join(ch for ch in unit if not unicodedata.combining(ch))
    unit = unit.replace(".", "").replace(" ", "")
    if unit.endswith("vnd") and unit not in {"vnd", "kvnd"}:
        unit = unit[: -len("vnd")]

    if unit in {"ti", "ty", "b", "bn", "billion", "billions"}:
        mult = Decimal("1000000")  # 1 tỉ VND = 1,000,000 kVND
    elif unit in {"trieu", "m", "million", "millions"}:
        mult = Decimal("1000")  # 1 triệu VND = 1,000 kVND
    elif unit in {"nghin", "ngan", "k", "thousand", "thousands"}:
        mult = Decimal("1")  # 1 nghìn VND = 1 kVND
    elif unit in {"kvnd"}:
        mult = Decimal("1")
    elif unit in {"vnd"}:
        # Only accept whole-thousand VND amounts (otherwise would require fractions of kVND).
        kvnd = (num / Decimal("1000"))
        if kvnd != kvnd.to_integral_value():
            raise SystemExit(
                f"[codex] ERROR: CODEX_BUDGET_TEXT='{raw}' is in VND but not divisible by 1000; "
                "provide CODEX_TOTAL_CAPITAL_KVND instead."
            )
        return int(kvnd)
    else:
        raise SystemExit(
            f"[codex] ERROR: unsupported unit in CODEX_BUDGET_TEXT='{raw}'. "
            "Supported: tỉ/ty/billion, triệu/million/m, nghìn/k, kVND, or provide CODEX_TOTAL_CAPITAL_KVND."
        )

    kvnd = num * mult
    if kvnd != kvnd.to_integral_value():
        raise SystemExit(
            f"[codex] ERROR: CODEX_BUDGET_TEXT='{raw}' results in a non-integer kVND amount; "
            "provide CODEX_TOTAL_CAPITAL_KVND."
        )
    return int(kvnd)


def run_codex(
    *,
    codex_bin: str,
    args: list[str],
    prompt: str,
    cwd: pathlib.Path,
    log_fp,
    capture_thread_id: bool = False,
    idle_timeout_seconds: float = 0,
    completion_file: pathlib.Path | None = None,
    output_csv: pathlib.Path | None = None,
    completion_grace_seconds: float = 0,
):
    def log_text_block(tag: str, text: str) -> None:
        text = text.rstrip("\n")
        if not text:
            return
        log(f"{tag}\n{text}", log_fp)

    def handle_stdout_line(line: str) -> tuple[str | None, str | None]:
        current_thread_id: str | None = None
        current_agent_text: str | None = None
        try:
            obj = json.loads(line)
            if capture_thread_id and obj.get("type") == "thread.started" and obj.get("thread_id"):
                current_thread_id = obj["thread_id"]
            typ = obj.get("type")
            if typ:
                log(f"codex.event type={typ}", log_fp)
                if typ.startswith("thread.") or typ.startswith("turn."):
                    return current_thread_id, current_agent_text
                if typ == "item.completed" or typ == "item.started":
                    item = obj.get("item")
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        text = item.get("text")
                        # Only log conversational text to avoid dumping large command blobs
                        if text and item_type in {"agent_message", "reasoning"}:
                            tag = f"codex.reply item={item.get('id', '?')} type={item_type}"
                            log_text_block(tag, text)
                            if item_type == "agent_message":
                                current_agent_text = text
                elif typ == "error":
                    msg = obj.get("message")
                    if msg:
                        log_text_block("codex.error", msg)
        except Exception:
            # Non-JSON lines (unlikely when --json is used)
            log(f"codex.stdout {line.rstrip()}", log_fp)
        return current_thread_id, current_agent_text

    def pump_stream(name: str, stream, out_queue: queue.Queue[tuple[str, str | None]]) -> None:
        try:
            while True:
                line = stream.readline()
                if not line:
                    out_queue.put((name, None))
                    return
                out_queue.put((name, line))
        except Exception as exc:  # pragma: no cover - defensive logging path
            out_queue.put((name, f"__STREAM_ERROR__:{exc}"))
            out_queue.put((name, None))

    proc = subprocess.Popen(
        [codex_bin, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    # Write prompt then close stdin
    proc.stdin.write(prompt)
    proc.stdin.close()

    thread_id: str | None = None
    last_agent_text: str | None = None
    timed_out = False
    forced_stop = False
    event_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_thread = threading.Thread(
        target=pump_stream, args=("stdout", proc.stdout, event_queue), daemon=True
    )
    stderr_thread = threading.Thread(
        target=pump_stream, args=("stderr", proc.stderr, event_queue), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    open_streams = {"stdout", "stderr"}
    last_activity = time.monotonic()
    completion_detected_at: float | None = None

    while open_streams:
        if (
            completion_file is not None
            and output_csv is not None
            and completion_file.exists()
            and output_csv.exists()
        ):
            if completion_detected_at is None:
                completion_detected_at = time.monotonic()
                log(
                    f"detected output+completion during active codex run: {output_csv.name}, {completion_file.name}",
                    log_fp,
                )
            elif (
                completion_grace_seconds >= 0
                and proc.poll() is None
                and (time.monotonic() - completion_detected_at) > completion_grace_seconds
            ):
                forced_stop = True
                log(
                    f"completion grace {completion_grace_seconds}s elapsed after {completion_file.name}; terminating active codex process",
                    log_fp,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log("codex process ignored SIGTERM after completion; sending SIGKILL", log_fp)
                    proc.kill()
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.stderr.close()
                except Exception:
                    pass
                continue
        try:
            stream_name, payload = event_queue.get(timeout=0.25)
        except queue.Empty:
            if (
                idle_timeout_seconds > 0
                and proc.poll() is None
                and (time.monotonic() - last_activity) > idle_timeout_seconds
            ):
                timed_out = True
                forced_stop = True
                log(
                    f"codex idle timeout after {idle_timeout_seconds}s without stdout/stderr; terminating process",
                    log_fp,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log("codex process ignored SIGTERM; sending SIGKILL", log_fp)
                    proc.kill()
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.stderr.close()
                except Exception:
                    pass
                continue
            if forced_stop and proc.poll() is not None:
                break
            if proc.poll() is not None and stdout_thread.is_alive() is False and stderr_thread.is_alive() is False:
                break
            continue

        if payload is None:
            open_streams.discard(stream_name)
            continue

        last_activity = time.monotonic()
        if stream_name == "stdout":
            current_thread_id, current_agent_text = handle_stdout_line(payload)
            if current_thread_id:
                thread_id = current_thread_id
            if current_agent_text:
                last_agent_text = current_agent_text
        else:
            if payload.startswith("__STREAM_ERROR__:"):
                log(f"codex.stderr stream error {payload[len('__STREAM_ERROR__:'):]}", log_fp)
            else:
                log(f"codex.stderr {payload.rstrip()}", log_fp)

    stdout_thread.join(timeout=0.5)
    stderr_thread.join(timeout=0.5)
    return proc.wait(), thread_id, last_agent_text, timed_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Codex exec/resume until completion file appears")
    p.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    p.add_argument("--workdir", required=True)
    p.add_argument("--prompt-file", default=os.environ.get("CODEX_PROMPT_FILE", "prompts/PROMPT.txt"))
    p.add_argument("--budget-text", default=os.environ.get("CODEX_BUDGET_TEXT", "5 tỉ"))
    p.add_argument(
        "--total-capital-kvnd",
        default=os.environ.get("CODEX_TOTAL_CAPITAL_KVND", ""),
        help="If set, write total_capital_kVND.txt as this integer (kVND). Overrides --budget-text parsing.",
    )
    p.add_argument("--output-csv", default=os.environ.get("CODEX_OUTPUT_CSV", "orders.csv"))
    p.add_argument("--done-file", default=os.environ.get("CODEX_DONE_FILE", "DONE.md"))
    p.add_argument("--continue-message", default=os.environ.get("CODEX_CONTINUE_MESSAGE", "Tiếp tục."))
    p.add_argument("--grace-after-csv", type=float, default=float(os.environ.get("CODEX_GRACE_AFTER_CSV", 8)))
    p.add_argument("--log-dir", default=os.environ.get("CODEX_LOG_DIR", "out/codex"))
    p.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.4"))
    p.add_argument("--reasoning", default=os.environ.get("CODEX_REASONING", "high"))
    p.add_argument(
        "--archive-root",
        default=os.environ.get("CODEX_ARCHIVE_ROOT", "archives/codex_runs"),
        help="Repo-relative directory where completed orders/log bundles are archived.",
    )
    p.add_argument(
        "--archive-enabled",
        type=int,
        default=int(os.environ.get("CODEX_ARCHIVE_ENABLED", "1")),
        help="Archive orders.csv + matching codex log after a successful run (1/0).",
    )
    p.add_argument(
        "--archive-git-commit",
        type=int,
        default=int(os.environ.get("CODEX_ARCHIVE_GIT_COMMIT", "0")),
        help="Commit the archive folder to git after creating it (1/0).",
    )
    p.add_argument(
        "--tracked-snapshot-enabled",
        type=int,
        default=int(os.environ.get("CODEX_TRACKED_SNAPSHOT_ENABLED", "1")),
        help="Copy the full codex_universe into a repo-tracked snapshot folder after a successful run (1/0).",
    )
    p.add_argument(
        "--tracked-snapshot-root",
        default=os.environ.get("CODEX_TRACKED_SNAPSHOT_ROOT", "codex_universe_history"),
        help="Repo-relative directory where full codex_universe snapshots are copied.",
    )
    p.add_argument(
        "--order-history-enabled",
        type=int,
        default=int(os.environ.get("CODEX_ORDER_HISTORY_ENABLED", "0")),
        help="Clone/pull a dedicated order-history repo, copy archived artifacts there, then commit+push (1/0).",
    )
    p.add_argument(
        "--order-history-repo",
        default=os.environ.get("CODEX_ORDER_HISTORY_REPO", ""),
        help="GitHub repo in owner/name form used to store archived orders.",
    )
    p.add_argument(
        "--order-history-clone-dir",
        default=os.environ.get("CODEX_ORDER_HISTORY_CLONE_DIR", "../tcbs-orders-history"),
        help="Clone dir for the dedicated order-history repo. Relative paths resolve from the main repo root.",
    )
    p.add_argument(
        "--order-history-gh-bin",
        default=os.environ.get("CODEX_ORDER_HISTORY_GH_BIN", "gh"),
        help="GitHub CLI binary used for first-time clone.",
    )
    p.add_argument(
        "--tcbs-account",
        default=os.environ.get("TCBS_ACCOUNT_SLUG") or os.environ.get("TCBS_USERNAME", ""),
        help="Account slug used under the order-history repo. Defaults to TCBS_ACCOUNT_SLUG or TCBS_USERNAME.",
    )
    p.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=float(os.environ.get("CODEX_IDLE_TIMEOUT_SECONDS", "900")),
        help="Kill a silent codex exec/resume after this many seconds without stdout/stderr activity (<=0 disables).",
    )
    # Ưu tiên CODEX_MAX_WALL_SECONDS; fallback sang CODEX_TIMEOUT_SECONDS cho tương thích ngược.
    max_wall_default = os.environ.get("CODEX_MAX_WALL_SECONDS") or os.environ.get("CODEX_TIMEOUT_SECONDS") or "5400"
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=float(max_wall_default),
        help="Kill wrapper after this many seconds (<=0 disables).",
    )
    p.add_argument(
        "--max-continues",
        type=int,
        default=int(os.environ.get("CODEX_MAX_CONTINUES", "0")),
        help="Stop after this many continue messages (<=0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = pathlib.Path(args.workdir).resolve()
    if not workdir.is_dir():
        raise SystemExit(f"[codex] ERROR: workdir '{workdir}' not found")

    tracked_snapshot_enabled = bool(args.tracked_snapshot_enabled)
    order_history_enabled = bool(args.order_history_enabled)
    tcbs_account = resolve_tcbs_account_slug(args.tcbs_account) if (order_history_enabled or tracked_snapshot_enabled) else None
    if order_history_enabled and not str(args.order_history_repo).strip():
        raise SystemExit(
            "[codex] ERROR: CODEX_ORDER_HISTORY_REPO is required when order-history sync is enabled"
        )

    # Fail fast on budget parsing before mutating the workdir.
    if args.total_capital_kvnd:
        try:
            total_capital_kvnd = int(str(args.total_capital_kvnd).strip())
        except Exception:
            raise SystemExit("[codex] ERROR: CODEX_TOTAL_CAPITAL_KVND must be an integer (kVND)")
        if total_capital_kvnd <= 0:
            raise SystemExit("[codex] ERROR: CODEX_TOTAL_CAPITAL_KVND must be > 0 (kVND)")
    else:
        total_capital_kvnd = parse_budget_text_to_total_capital_kvnd(args.budget_text)

    prompt_path = pathlib.Path(args.prompt_file)
    if not prompt_path.is_file():
        raise SystemExit(f"[codex] ERROR: prompt file '{prompt_path}' not found")
    base_prompt = prompt_path.read_text(encoding="utf-8").rstrip("\n") + "\n"

    # Prepare logs
    log_dir = pathlib.Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"codex_session_{stamp}.log"
    with log_file.open("a", encoding="utf-8") as lf:
        log(f"workdir={workdir}", lf)
        log(f"output_csv={args.output_csv}", lf)
        log(f"done_file={args.done_file}", lf)

        clean_and_require_universe(workdir)

        # Ensure Codex workdir has required inputs for the prompt.
        # The prompt reads `./total_capital_kVND.txt` (single number, kVND).
        capital_path = workdir / "total_capital_kVND.txt"
        capital_path.write_text(f"{total_capital_kvnd}\n", encoding="utf-8")
        log(f"prepared input file: {capital_path.name}", lf)

        # Detect JSON flag
        json_flag = detect_json_flag(args.codex_bin)
        if not json_flag:
            log("WARNING: Codex CLI does not advertise a JSON flag; continuing without", lf)

        # 1) Initial exec
        exec_args = [
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            "danger-full-access",
            "--cd",
            str(workdir),
            "--model",
            args.model,
            "--config",
            f"model_reasoning_effort=\"{args.reasoning}\"",
            "--config",
            "mcp_servers={}",
        ]
        if json_flag:
            exec_args.insert(1, json_flag)

        target_csv = (workdir / args.output_csv).resolve()
        target_done = (workdir / args.done_file).resolve()
        log("starting codex exec (prompt)", lf)
        rc, thread_id, last_agent_text, timed_out = run_codex(
            codex_bin=args.codex_bin,
            args=exec_args,
            prompt=base_prompt,
            cwd=workdir,
            log_fp=lf,
            capture_thread_id=True,
            idle_timeout_seconds=args.idle_timeout_seconds,
            completion_file=target_done,
            output_csv=target_csv,
            completion_grace_seconds=args.grace_after_csv,
        )
        if rc != 0 and not thread_id:
            raise SystemExit(f"[codex] ERROR: codex exec failed with {rc}")
        if not thread_id:
            raise SystemExit("[codex] ERROR: failed to extract thread_id from codex JSON events")
        if timed_out:
            log("initial codex exec hit idle timeout after thread start; switching to resume", lf)
        elif rc != 0:
            log(f"initial codex exec exited {rc} after thread start; switching to resume", lf)
        log(f"thread_id={thread_id}", lf)
        write_resume_helper_script(
            log_dir=log_dir,
            codex_bin=args.codex_bin,
            workdir=workdir,
            model=args.model,
            reasoning=args.reasoning,
            thread_id=thread_id,
            log_fp=lf,
        )
        # 2) Kickoff message: tổng vốn đã được cung cấp qua file total_capital_kVND.txt,
        # nên chỉ cần gửi "Tiếp tục" để chuyển sang bước kế tiếp.
        kickoff = "Tiếp tục\n"
        resume_args = exec_args + ["resume", thread_id, "-"]
        log("sending kickoff", lf)
        rc, _, last_agent_text, timed_out = run_codex(
            codex_bin=args.codex_bin,
            args=resume_args,
            prompt=kickoff,
            cwd=workdir,
            log_fp=lf,
            idle_timeout_seconds=args.idle_timeout_seconds,
            completion_file=target_done,
            output_csv=target_csv,
            completion_grace_seconds=args.grace_after_csv,
        )
        if rc != 0:
            if timed_out:
                log("kickoff resume hit idle timeout; retrying with continue loop", lf)
            else:
                log(f"kickoff resume exited {rc} (continuing)", lf)
        # 3) Loop "Tiếp tục." cho đến khi xuất hiện file completion.
        # Completion is checked only between resume rounds; an in-flight
        # `codex resume` is allowed to finish naturally unless it hits
        # the idle-timeout guard.
        wall_start = time.monotonic()
        continues_sent = 0
        while True:
            if target_done.exists():
                log(f"detected completion file: {target_done.name}", lf)
                time.sleep(args.grace_after_csv)
                break
            cont = args.continue_message.rstrip("\n") + "\n"
            log("sending continue", lf)
            rc, _, last_agent_text, timed_out = run_codex(
                codex_bin=args.codex_bin,
                args=resume_args,
                prompt=cont,
                cwd=workdir,
                log_fp=lf,
                idle_timeout_seconds=args.idle_timeout_seconds,
                completion_file=target_done,
                output_csv=target_csv,
                completion_grace_seconds=args.grace_after_csv,
            )
            if rc != 0:
                if timed_out:
                    log("resume hit idle timeout; killing and retrying same thread", lf)
                else:
                    log(f"resume exited {rc} (continuing)", lf)
            continues_sent += 1
            if args.max_continues > 0 and continues_sent >= args.max_continues:
                raise SystemExit(
                    f"[codex] ERROR: wrapper hit max continues {args.max_continues} waiting for {target_done.name}"
                )
            # Re-check file completion sau mỗi lần gửi continue để tránh race condition
            # khi file được tạo ngay sát ngưỡng timeout.
            if target_done.exists():
                log(f"detected completion file after continue: {target_done.name}", lf)
                time.sleep(args.grace_after_csv)
                break
            if args.max_wall_seconds > 0 and (time.monotonic() - wall_start) > args.max_wall_seconds:
                raise SystemExit(
                    f"[codex] ERROR: wrapper hit max wall time {args.max_wall_seconds}s waiting for {target_done.name}"
                )

        # 4) Print CSV to stdout và ghi assistant-only log
        if not target_csv.exists():
            raise SystemExit(
                f"[codex] ERROR: completion file '{target_done.name}' exists but output CSV '{target_csv.name}' is missing"
            )
        if args.archive_enabled or order_history_enabled or tracked_snapshot_enabled:
            repo_root = find_git_root(workdir)
            lf.flush()
            archive_dir = None
            archive_dir = archive_codex_run(
                repo_root=repo_root,
                archive_root_rel=args.archive_root,
                output_csv=target_csv,
                completion_file=target_done,
                log_file=log_file,
                thread_id=thread_id,
                tcbs_account=tcbs_account,
                log_fp=lf,
            ) if (args.archive_enabled or order_history_enabled) else None
            if archive_dir is not None and args.archive_git_commit:
                git_commit_archive(
                    repo_root=repo_root,
                    archive_dir=archive_dir,
                    log_fp=lf,
                )
            if tracked_snapshot_enabled:
                snapshot_codex_workspace(
                    repo_root=repo_root,
                    snapshot_root_rel=args.tracked_snapshot_root,
                    workdir=workdir,
                    tcbs_account=tcbs_account or "",
                    stamp=archive_dir.name if archive_dir is not None else None,
                    log_fp=lf,
                )
            if order_history_enabled:
                sync_archive_to_order_history_repo(
                    repo_root=repo_root,
                    archive_dir=archive_dir,
                    remote_repo=args.order_history_repo,
                    clone_dir_raw=args.order_history_clone_dir,
                    tcbs_account=tcbs_account,
                    gh_bin=args.order_history_gh_bin,
                    log_fp=lf,
                )
        sys.stdout.write(target_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
