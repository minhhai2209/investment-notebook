#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

load_dotenv_if_present() {
  local env_file="$ROOT_DIR/.env"
  [[ -f "$env_file" ]] || return 0

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line="${raw_line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    [[ -n "$key" ]] || continue
    if [[ -z "${!key+x}" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < "$env_file"
}

load_dotenv_if_present

VENV="venv"
[[ -d .venv ]] && VENV=".venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PY_BIN=""

CODEX_DIR="${CODEX_DIR:-codex_universe}"
CODEX_PROMPT_FILE="${CODEX_PROMPT_FILE:-prompts/PROMPT.txt}"
CODEX_BUDGET_TEXT="${CODEX_BUDGET_TEXT:-5 tỉ}"
CODEX_TOTAL_CAPITAL_KVND="${CODEX_TOTAL_CAPITAL_KVND:-}"
CODEX_OUTPUT_CSV="${CODEX_OUTPUT_CSV:-orders.csv}"
CODEX_DONE_FILE="${CODEX_DONE_FILE:-DONE.md}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-7200}"
CODEX_MAX_WALL_SECONDS="${CODEX_MAX_WALL_SECONDS:-$CODEX_TIMEOUT_SECONDS}"
CODEX_MAX_CONTINUES="${CODEX_MAX_CONTINUES:-200}"
CODEX_GRACE_AFTER_CSV="${CODEX_GRACE_AFTER_CSV:-8}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"
CODEX_REASONING="${CODEX_REASONING:-xhigh}"
CODEX_CONTINUE_MESSAGE="${CODEX_CONTINUE_MESSAGE:-Tiếp tục.}"
CODEX_IDLE_TIMEOUT_SECONDS="${CODEX_IDLE_TIMEOUT_SECONDS:-900}"
CODEX_HEARTBEAT_SECONDS="${CODEX_HEARTBEAT_SECONDS:-5}"
CODEX_LOG_DIR="${CODEX_LOG_DIR:-out/codex}"
CODEX_ARCHIVE_ENABLED="${CODEX_ARCHIVE_ENABLED:-1}"
CODEX_ARCHIVE_ROOT="${CODEX_ARCHIVE_ROOT:-archives/codex_runs}"
CODEX_ARCHIVE_GIT_COMMIT="${CODEX_ARCHIVE_GIT_COMMIT:-0}"
CODEX_TRACKED_SNAPSHOT_ENABLED="${CODEX_TRACKED_SNAPSHOT_ENABLED:-1}"
CODEX_TRACKED_SNAPSHOT_ROOT="${CODEX_TRACKED_SNAPSHOT_ROOT:-codex_universe_history}"
CODEX_ORDER_HISTORY_ENABLED="${CODEX_ORDER_HISTORY_ENABLED:-1}"
CODEX_ORDER_HISTORY_REPO="${CODEX_ORDER_HISTORY_REPO:-minhhai2209/tcbs-orders-history}"
CODEX_ORDER_HISTORY_CLONE_DIR="${CODEX_ORDER_HISTORY_CLONE_DIR:-../tcbs-orders-history}"
CODEX_ORDER_HISTORY_GH_BIN="${CODEX_ORDER_HISTORY_GH_BIN:-gh}"
TCBS_ACCOUNT_SLUG="${TCBS_ACCOUNT_SLUG:-${TCBS_USERNAME:-}}"

ensure_venv() {
  if [[ -x "$VENV/bin/python" ]]; then
    PY_BIN="$VENV/bin/python"
  else
    echo "[setup] Creating virtualenv at ./$VENV"
    python3 -m venv "$VENV"
    PY_BIN="$VENV/bin/python"
  fi

  local stamp="$VENV/.pip-stamp"
  if [[ ! -f "$stamp" || "$REQ_FILE" -nt "$stamp" ]]; then
    echo "[setup] Installing requirements"
    "$PY_BIN" -m pip install -r "$REQ_FILE" >/dev/null
    touch "$stamp"
  fi
}

run_engine() {
  ensure_venv
  echo "[engine] Using: $PY_BIN"
  "$PY_BIN" -m scripts.engine.data_engine --config "${1:-config/data_engine.yaml}"
}

run_ml_cycle_report() {
  ensure_venv
  echo "[ml_cycle] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_cycle_forecast_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis/ml_cycle_forecast
}

run_ticker_playbook_report() {
  ensure_venv
  echo "[playbook] Using: $PY_BIN"
  local tickers
  tickers="$("$PY_BIN" - <<'PY'
import pandas as pd

df = pd.read_csv("out/universe.csv", usecols=["Ticker"])
tickers = []
for raw in df["Ticker"].dropna().tolist():
    ticker = str(raw).strip().upper()
    if not ticker or ticker == "VNINDEX" or ticker in tickers:
        continue
    tickers.append(ticker)
print(",".join(tickers))
PY
)"
  if [[ -z "$tickers" ]]; then
    echo "[playbook] ERROR: out/universe.csv did not provide any ticker for playbook report." >&2
    exit 2
  fi
  "$PY_BIN" -m scripts.analysis.build_ticker_playbook_report \
    --tickers "$tickers" \
      --output-dir out/analysis/ticker_playbooks_live
}

run_ohlc_next_session_report() {
  ensure_venv
  echo "[ml_ohlc_next] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_ohlc_next_session_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis
}

run_intraday_rest_of_session_report() {
  ensure_venv
  echo "[ml_intraday_rest] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_intraday_rest_of_session_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis
}

run_ml_range_report() {
  ensure_venv
  echo "[ml_range] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_range_forecast_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis
}

run_single_name_timing_report() {
  ensure_venv
  echo "[ml_single_name_timing] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_single_name_timing_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis
}

run_entry_ladder_eval_report() {
  ensure_venv
  echo "[ml_entry_ladder_eval] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_entry_ladder_eval_report \
    --history-dir out/data \
    --universe-csv out/universe.csv \
    --ohlc-csv out/analysis/ml_ohlc_next_session.csv \
    --range-full-csv out/analysis/ml_range_predictions_full_2y.csv \
    --range-recent-csv out/analysis/ml_range_predictions_recent_focus.csv \
    --cycle-csv out/analysis/ml_cycle_forecast/cycle_forecast_best_horizon_by_ticker.csv \
    --single-name-csv out/analysis/ml_single_name_timing.csv \
    --output-dir out/analysis
}

run_research_bundle() {
  ensure_venv
  echo "[research] Using: $PY_BIN"
  local total_capital_kvnd
  total_capital_kvnd="$(resolve_total_capital_kvnd)"
  local extra_args=()
  if [[ -f "$ROOT_DIR/human_notes.md" ]]; then
    extra_args+=(--human-notes-path "$ROOT_DIR/human_notes.md")
  fi
  "$PY_BIN" -m scripts.research.build_research_bundle \
    --universe-csv out/universe.csv \
    --market-summary-json out/market_summary.json \
    --sector-summary-csv out/sector_summary.csv \
    --analysis-dir out/analysis \
    --intraday-dir out/data/intraday_5m \
    --research-dir research \
    --total-capital-kvnd "$total_capital_kvnd" \
    "${extra_args[@]}" \
    "$@"
}

resolve_total_capital_kvnd() {
  if [[ -z "$PY_BIN" ]]; then
    ensure_venv
  fi
  "$PY_BIN" - "$CODEX_TOTAL_CAPITAL_KVND" "$CODEX_BUDGET_TEXT" <<'PY'
import sys

from scripts.codex.exec_resume import parse_budget_text_to_total_capital_kvnd

raw_total_capital = str(sys.argv[1]).strip()
budget_text = sys.argv[2]

if raw_total_capital:
    try:
        total_capital_kvnd = int(raw_total_capital)
    except ValueError as exc:
        raise SystemExit("[codex] ERROR: CODEX_TOTAL_CAPITAL_KVND must be an integer (kVND)") from exc
    if total_capital_kvnd <= 0:
        raise SystemExit("[codex] ERROR: CODEX_TOTAL_CAPITAL_KVND must be > 0 (kVND)")
else:
    total_capital_kvnd = parse_budget_text_to_total_capital_kvnd(budget_text)

print(total_capital_kvnd)
PY
}

prepare_codex_total_capital_file() {
  local dest_dir="$ROOT_DIR/$CODEX_DIR"
  local capital_path="$dest_dir/total_capital_kVND.txt"
  local total_capital_kvnd
  total_capital_kvnd="$(resolve_total_capital_kvnd)"
  mkdir -p "$dest_dir"
  printf '%s\n' "$total_capital_kvnd" > "$capital_path"
  echo "[engine] Prepared total capital input -> $capital_path"
}

refresh_dynamic_industry_map() {
  ensure_venv
  echo "[industry_map] Using: $PY_BIN"
  local config_path="$ROOT_DIR/config/data_engine.yaml"
  local configured_tickers=""
  if [[ -f "$config_path" ]]; then
    configured_tickers="$("$PY_BIN" - "$config_path" <<'PY'
import sys
from pathlib import Path
import yaml
import pandas as pd

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if not isinstance(data, dict):
    raise SystemExit(f"{path} must be a YAML mapping")

universe = data.get("universe", {})
if not isinstance(universe, dict):
    raise SystemExit(f"{path} key 'universe' must be a mapping")

portfolio_cfg = data.get("portfolio", {}) or {}
if not isinstance(portfolio_cfg, dict):
    raise SystemExit(f"{path} key 'portfolio' must be a mapping")

config_dir = path.parent.resolve()
repo_root = config_dir.parent

def load_tickers(raw, label: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace(",", " ").split()]
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise SystemExit(f"{path} key '{label}' must be a list or string")
    seen = set()
    ordered = []
    for token in items:
        token = token.upper()
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered

tickers = []
for ticker in load_tickers(universe.get("core_tickers"), "universe.core_tickers"):
    if ticker not in tickers:
        tickers.append(ticker)
if not tickers:
    for ticker in load_tickers(universe.get("tickers"), "universe.tickers"):
        if ticker not in tickers:
            tickers.append(ticker)

portfolio_dir_raw = portfolio_cfg.get("directory", "data/portfolios")
portfolio_dir = Path(portfolio_dir_raw)
if not portfolio_dir.is_absolute():
    portfolio_dir = (config_dir / portfolio_dir).resolve()
portfolio_csv = portfolio_dir / "portfolio.csv"
if portfolio_csv.exists():
    try:
        portfolio_df = pd.read_csv(portfolio_csv, usecols=["Ticker"])
    except ValueError as exc:
        raise SystemExit(f"{portfolio_csv} missing required column 'Ticker': {exc}")
    for raw in portfolio_df["Ticker"].dropna().astype(str).tolist():
        ticker = raw.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
if tickers:
    print(",".join(tickers))
PY
)"
  fi

  if [[ -n "$configured_tickers" ]]; then
    local tmp_csv
    tmp_csv="$(mktemp)"
    {
      echo "Ticker"
      tr ',' '\n' <<< "$configured_tickers"
    } > "$tmp_csv"
    echo "[industry_map] Working universe detected (core + current portfolio); refreshing only: $configured_tickers"
    "$PY_BIN" scripts/tools/refresh_industry_map.py \
      --from-vietstock-profiles-csv "$tmp_csv" \
      --output data/industry_map.csv
    rm -f "$tmp_csv"
    return
  fi

  "$PY_BIN" scripts/tools/refresh_industry_map.py \
    --from-live-vn100-portfolio-profiles \
    --portfolio-csv data/portfolios/portfolio.csv \
    --extra-ticker NVL \
    --output data/industry_map.csv
}

sync_codex_universe() {
  local dest_dir="$ROOT_DIR/$CODEX_DIR"
  local bundle_manifest_dest="$dest_dir/bundle_manifest.json"
  local human_notes_src="$ROOT_DIR/human_notes.md"
  local human_notes_dest="$dest_dir/human_notes.md"
  local research_src="$ROOT_DIR/research"
  local research_dest="$dest_dir/research"
  local strategy_buckets_src="$ROOT_DIR/strategy_buckets.csv"
  local strategy_buckets_dest="$dest_dir/strategy_buckets.csv"
  local files=(
    "universe.csv"
    "market_summary.json"
    "sector_summary.csv"
  )
  local range_full_src="$ROOT_DIR/out/analysis/ml_range_predictions_full_2y.csv"
  local range_recent_src="$ROOT_DIR/out/analysis/ml_range_predictions_recent_focus.csv"
  local range_full_dest="$dest_dir/ml_range_predictions_full_2y.csv"
  local range_recent_dest="$dest_dir/ml_range_predictions_recent_focus.csv"
  local matrix_src="$ROOT_DIR/out/analysis/ml_cycle_forecast/cycle_forecast_ticker_matrix.csv"
  local horizon_src="$ROOT_DIR/out/analysis/ml_cycle_forecast/cycle_forecast_best_horizon_by_ticker.csv"
  local matrix_dest="$dest_dir/ml_cycle_forecast_ticker_matrix.csv"
  local horizon_dest="$dest_dir/ml_cycle_forecast_best_horizon.csv"
  local playbook_src="$ROOT_DIR/out/analysis/ticker_playbooks_live/ticker_playbook_best_configs.csv"
  local playbook_dest="$dest_dir/ticker_playbook_best_configs.csv"
  local ohlc_src="$ROOT_DIR/out/analysis/ml_ohlc_next_session.csv"
  local ohlc_dest="$dest_dir/ml_ohlc_next_session.csv"
  local intraday_src="$ROOT_DIR/out/analysis/ml_intraday_rest_of_session.csv"
  local intraday_dest="$dest_dir/ml_intraday_rest_of_session.csv"
  local single_name_src="$ROOT_DIR/out/analysis/ml_single_name_timing.csv"
  local single_name_dest="$dest_dir/ml_single_name_timing.csv"
  local entry_ladder_src="$ROOT_DIR/out/analysis/ml_entry_ladder_eval.csv"
  local entry_ladder_dest="$dest_dir/ml_entry_ladder_eval.csv"

  if [[ -z "$CODEX_DIR" ]]; then
    echo "[engine] ERROR: CODEX_DIR is empty; cannot sync Codex universe." >&2
    exit 2
  fi
  if [[ ! -d "$dest_dir" ]]; then
    mkdir -p "$dest_dir"
    echo "[engine] Created Codex directory: $dest_dir"
  fi

  local file_name src dest dest_name
  for file_name in "${files[@]}"; do
    src="$ROOT_DIR/out/$file_name"
    dest_name="$(basename "$file_name")"
    dest="$dest_dir/$dest_name"
    if [[ ! -f "$src" ]]; then
      echo "[engine] ERROR: Expected output '$src' not found after engine run." >&2
      exit 2
    fi
    cp -f "$src" "$dest"
  done
  if [[ -z "$PY_BIN" ]]; then
    ensure_venv
  fi
  if [[ ! -f "$range_full_src" || ! -f "$range_recent_src" || ! -f "$matrix_src" || ! -f "$horizon_src" || ! -f "$playbook_src" || ! -f "$ohlc_src" ]]; then
    echo "[engine] ERROR: Expected ML forecast outputs not found after engine run." >&2
    exit 2
  fi
  "$PY_BIN" - \
    "$dest_dir/universe.csv" \
    "$range_full_src" "$range_full_dest" \
    "$range_recent_src" "$range_recent_dest" \
    "$matrix_src" "$matrix_dest" \
    "$horizon_src" "$horizon_dest" \
    "$playbook_src" "$playbook_dest" \
    "$ohlc_src" "$ohlc_dest" <<'PY'
import sys
from pathlib import Path

import pandas as pd

universe_path = Path(sys.argv[1])
path_pairs = [
    (Path(sys.argv[2]), Path(sys.argv[3]), True),
    (Path(sys.argv[4]), Path(sys.argv[5]), True),
    (Path(sys.argv[6]), Path(sys.argv[7]), True),
    (Path(sys.argv[8]), Path(sys.argv[9]), True),
    (Path(sys.argv[10]), Path(sys.argv[11]), False),
    (Path(sys.argv[12]), Path(sys.argv[13]), False),
]

universe_df = pd.read_csv(universe_path, usecols=["Ticker"])
allowed = {
    str(ticker).strip().upper()
    for ticker in universe_df["Ticker"].dropna().tolist()
    if str(ticker).strip()
}
if not allowed:
    raise SystemExit("Synced universe.csv is empty; cannot filter ML cycle forecasts.")

for src, dest, require_index_coverage in path_pairs:
    df = pd.read_csv(src)
    if "Ticker" not in df.columns:
        raise SystemExit(f"{src} missing required column 'Ticker'")
    ticker_series = df["Ticker"].astype(str).str.strip().str.upper()
    filtered = df.loc[ticker_series.isin(allowed)].copy()
    remaining = {
        str(ticker).strip().upper()
        for ticker in filtered["Ticker"].dropna().tolist()
        if str(ticker).strip()
    }
    required = allowed if require_index_coverage else {ticker for ticker in allowed if ticker != "VNINDEX"}
    missing = sorted(required - remaining)
    if missing:
        raise SystemExit(
            f"{src} does not cover synced universe tickers: {', '.join(missing)}"
        )
    filtered.to_csv(dest, index=False)
PY
  if [[ -f "$intraday_src" ]]; then
    "$PY_BIN" - \
      "$dest_dir/universe.csv" \
      "$intraday_src" \
      "$intraday_dest" <<'PY'
import sys
from pathlib import Path

import pandas as pd

universe_path = Path(sys.argv[1])
src = Path(sys.argv[2])
dest = Path(sys.argv[3])

universe_df = pd.read_csv(universe_path, usecols=["Ticker"])
allowed = {
    str(ticker).strip().upper()
    for ticker in universe_df["Ticker"].dropna().tolist()
    if str(ticker).strip() and str(ticker).strip().upper() != "VNINDEX"
}
df = pd.read_csv(src)
if "Ticker" not in df.columns:
    raise SystemExit(f"{src} missing required column 'Ticker'")
filtered = df.loc[df["Ticker"].astype(str).str.strip().str.upper().isin(allowed)].copy()
filtered.to_csv(dest, index=False)
PY
  else
    rm -f "$intraday_dest"
  fi
  if [[ -f "$single_name_src" ]]; then
    "$PY_BIN" - \
      "$dest_dir/universe.csv" \
      "$single_name_src" \
      "$single_name_dest" <<'PY'
import sys
from pathlib import Path

import pandas as pd

universe_path = Path(sys.argv[1])
src = Path(sys.argv[2])
dest = Path(sys.argv[3])

universe_df = pd.read_csv(universe_path, usecols=["Ticker"])
allowed = {
    str(ticker).strip().upper()
    for ticker in universe_df["Ticker"].dropna().tolist()
    if str(ticker).strip() and str(ticker).strip().upper() != "VNINDEX"
}
df = pd.read_csv(src)
if "Ticker" not in df.columns:
    raise SystemExit(f"{src} missing required column 'Ticker'")
filtered = df.loc[df["Ticker"].astype(str).str.strip().str.upper().isin(allowed)].copy()
filtered.to_csv(dest, index=False)
PY
  else
    rm -f "$single_name_dest"
  fi
  if [[ -f "$entry_ladder_src" ]]; then
    "$PY_BIN" - \
      "$dest_dir/universe.csv" \
      "$entry_ladder_src" \
      "$entry_ladder_dest" <<'PY'
import sys
from pathlib import Path

import pandas as pd

universe_path = Path(sys.argv[1])
src = Path(sys.argv[2])
dest = Path(sys.argv[3])

universe_df = pd.read_csv(universe_path, usecols=["Ticker"])
allowed = {
    str(ticker).strip().upper()
    for ticker in universe_df["Ticker"].dropna().tolist()
    if str(ticker).strip() and str(ticker).strip().upper() != "VNINDEX"
}
df = pd.read_csv(src)
if "Ticker" not in df.columns:
    raise SystemExit(f"{src} missing required column 'Ticker'")
filtered = df.loc[df["Ticker"].astype(str).str.strip().str.upper().isin(allowed)].copy()
filtered.to_csv(dest, index=False)
PY
  else
    rm -f "$entry_ladder_dest"
  fi
  if [[ -f "$human_notes_src" ]]; then
    cp -f "$human_notes_src" "$human_notes_dest"
    echo "[engine] Synced human notes -> $human_notes_dest"
  else
    rm -f "$human_notes_dest"
  fi
  rm -rf "$dest_dir/deep_dive_notes"
  if [[ -d "$research_src" ]]; then
    rm -rf "$research_dest"
    cp -R "$research_src" "$research_dest"
    echo "[engine] Synced research artifacts -> $research_dest"
  else
    rm -rf "$research_dest"
  fi
  if [[ -f "$strategy_buckets_src" ]]; then
    "$PY_BIN" scripts/codex/strategy_buckets.py \
      --config "$ROOT_DIR/config/data_engine.yaml" \
      --source "$strategy_buckets_src" \
      --output "$strategy_buckets_dest"
    echo "[engine] Synthesized strategy buckets -> $strategy_buckets_dest"
  else
    rm -f "$strategy_buckets_dest"
  fi
  prepare_codex_total_capital_file
  "$PY_BIN" -m scripts.codex.build_bundle_manifest \
    --bundle-dir "$dest_dir" \
    --output "$bundle_manifest_dest" >/dev/null
  echo "[engine] Built bundle manifest -> $bundle_manifest_dest"
  echo "[engine] Synced Codex inputs -> $dest_dir"
}

run_tests() {
  ensure_venv
  echo "[tests] Using: $PY_BIN"
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
}

clean_codex_universe() {
  if [[ -z "$CODEX_DIR" ]]; then
    echo "[codex] ERROR: CODEX_DIR is empty; cannot clean Codex universe." >&2
    exit 2
  fi
  local dest_dir="$ROOT_DIR/$CODEX_DIR"
  if [[ "$dest_dir" == "/" || "$dest_dir" == "$ROOT_DIR" ]]; then
    echo "[codex] ERROR: Refusing to wipe unsafe path: $dest_dir" >&2
    exit 2
  fi
  if [[ ! -d "$dest_dir" ]]; then
    mkdir -p "$dest_dir"
    echo "[codex] Created Codex directory: $dest_dir"
    return
  fi
  if [[ -z "$(ls -A "$dest_dir" 2>/dev/null)" ]]; then
    echo "[codex] Codex directory already empty: $dest_dir"
    return
  fi
  find "$dest_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  echo "[codex] Cleared Codex directory: $dest_dir"
}

run_codex_chat() {
  ensure_venv
  if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    echo "[codex] ERROR: Codex binary '$CODEX_BIN' not found in PATH" >&2
    exit 2
  fi
  local codex_dir="$ROOT_DIR/$CODEX_DIR"
  local prompt_path="$ROOT_DIR/$CODEX_PROMPT_FILE"
  local log_dir="$ROOT_DIR/$CODEX_LOG_DIR"
  mkdir -p "$log_dir"

  "$PY_BIN" scripts/codex/exec_resume.py \
    --codex-bin "$CODEX_BIN" \
    --workdir "$codex_dir" \
    --prompt-file "$prompt_path" \
    --budget-text "$CODEX_BUDGET_TEXT" \
    --output-csv "$CODEX_OUTPUT_CSV" \
    --done-file "$CODEX_DONE_FILE" \
    --continue-message "$CODEX_CONTINUE_MESSAGE" \
    --idle-timeout-seconds "$CODEX_IDLE_TIMEOUT_SECONDS" \
    --grace-after-csv "$CODEX_GRACE_AFTER_CSV" \
    --max-wall-seconds "$CODEX_MAX_WALL_SECONDS" \
    --max-continues "$CODEX_MAX_CONTINUES" \
    --log-dir "$log_dir" \
    --model "$CODEX_MODEL" \
    --reasoning "$CODEX_REASONING" \
    --archive-enabled "$CODEX_ARCHIVE_ENABLED" \
    --archive-root "$CODEX_ARCHIVE_ROOT" \
    --archive-git-commit "$CODEX_ARCHIVE_GIT_COMMIT" \
    --tracked-snapshot-enabled "$CODEX_TRACKED_SNAPSHOT_ENABLED" \
    --tracked-snapshot-root "$CODEX_TRACKED_SNAPSHOT_ROOT" \
    --order-history-enabled "$CODEX_ORDER_HISTORY_ENABLED" \
    --order-history-repo "$CODEX_ORDER_HISTORY_REPO" \
    --order-history-clone-dir "$CODEX_ORDER_HISTORY_CLONE_DIR" \
    --order-history-gh-bin "$CODEX_ORDER_HISTORY_GH_BIN" \
    --tcbs-account "$TCBS_ACCOUNT_SLUG"
}

main() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [tests|tcbs_login|tcbs|tcbs_orders|portfolio|research|codex]" >&2
    exit 2
  fi
  local task="$1"
  shift || true
  case "$task" in
    tests)
      run_tests "$@"
      ;;
    tcbs_login)
      ensure_venv
      echo "[tcbs_login] Using: $PY_BIN"
      # Ensure Chrome is installed for Playwright (best-effort)
      "$PY_BIN" -m playwright install chrome >/dev/null 2>&1 || true
      "$PY_BIN" -m scripts.scrapers.tcbs --headful --login-only "$@"
      ;;
    tcbs)
      ensure_venv
      clean_codex_universe
      echo "[tcbs] Using: $PY_BIN"
      # Ensure Chrome is installed for Playwright (best-effort)
      "$PY_BIN" -m playwright install chrome >/dev/null 2>&1 || true
      "$PY_BIN" -m scripts.scrapers.tcbs --headful "$@"
      refresh_dynamic_industry_map
      run_engine
      run_ml_range_report
      run_ml_cycle_report
      run_ticker_playbook_report
      run_ohlc_next_session_report
      run_intraday_rest_of_session_report
      run_single_name_timing_report
      run_entry_ladder_eval_report
      run_research_bundle
      sync_codex_universe
      ;;
    tcbs_orders|orders)
      ensure_venv
      echo "[tcbs_orders] Using: $PY_BIN"
      # Ensure Chrome is installed for Playwright (best-effort)
      "$PY_BIN" -m playwright install chrome >/dev/null 2>&1 || true
      "$PY_BIN" -m scripts.scrapers.tcbs_orders --headful "$@"
      ;;
    portfolio)
      ensure_venv
      echo "[portfolio] Using: $PY_BIN"
      "$PY_BIN" -m scripts.engine.import_portfolio_from_downloads "$@"
      ;;
    research)
      run_research_bundle "$@"
      sync_codex_universe
      ;;
    codex)
      sync_codex_universe
      run_codex_chat
      ;;
    *)
      echo "Usage: $0 [tests|tcbs_login|tcbs|tcbs_orders|portfolio|research|codex]" >&2
      exit 2
      ;;
  esac
}

main "$@"
