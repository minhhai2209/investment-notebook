#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG_PATH="config/data_engine.yaml"
VENV_DIR="venv"
[[ -d .venv ]] && VENV_DIR=".venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PY_BIN=""

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

ensure_venv() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PY_BIN="$VENV_DIR/bin/python"
  else
    echo "[setup] Creating virtualenv at ./$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    PY_BIN="$VENV_DIR/bin/python"
  fi

  local stamp="$VENV_DIR/.pip-stamp"
  if [[ ! -f "$stamp" || "$REQ_FILE" -nt "$stamp" ]]; then
    echo "[setup] Installing requirements"
    "$PY_BIN" -m pip install -r "$REQ_FILE" >/dev/null
    touch "$stamp"
  fi
}

run_module() {
  local label="$1"
  shift
  ensure_venv
  echo "[$label] Using: $PY_BIN"
  "$PY_BIN" -m "$@"
}

clean_notebook_artifacts() {
  local analysis_dir="$ROOT_DIR/out/analysis"
  local research_dir="$ROOT_DIR/research"
  rm -rf "$analysis_dir" "$research_dir"
}

run_engine() {
  local config_path="${1:-$DEFAULT_CONFIG_PATH}"
  run_module engine scripts.engine.data_engine --config "$config_path"
}

run_range_report() {
  run_module range scripts.analysis.build_range_forecast_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis \
    "$@"
}

run_cycle_report() {
  run_module cycle scripts.analysis.build_cycle_forecast_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis/ml_cycle_forecast \
    "$@"
}

run_playbook_report() {
  ensure_venv
  local tickers
  tickers="$("$PY_BIN" - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("out/universe.csv")
if not path.exists():
    raise SystemExit("out/universe.csv not found; run ./broker.sh engine first")
df = pd.read_csv(path, usecols=["Ticker"])
seen = []
for raw in df["Ticker"].dropna().tolist():
    ticker = str(raw).strip().upper()
    if not ticker or ticker == "VNINDEX" or ticker in seen:
        continue
    seen.append(ticker)
print(",".join(seen))
PY
)"
  if [[ -z "$tickers" ]]; then
    echo "[playbook] ERROR: out/universe.csv did not contain any reportable tickers." >&2
    exit 2
  fi
  echo "[playbook] Using: $PY_BIN"
  "$PY_BIN" -m scripts.analysis.build_ticker_playbook_report \
    --tickers "$tickers" \
    --output-dir out/analysis/ticker_playbooks_live \
    "$@"
}

run_ohlc_report() {
  run_module ohlc scripts.analysis.build_ohlc_next_session_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis \
    "$@"
}

run_intraday_report() {
  run_module intraday scripts.analysis.build_intraday_rest_of_session_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis \
    "$@"
}

run_timing_report() {
  run_module timing scripts.analysis.build_single_name_timing_report \
    --universe-csv out/universe.csv \
    --output-dir out/analysis \
    "$@"
}

run_entry_ladder_report() {
  run_module entry_ladder scripts.analysis.build_entry_ladder_eval_report \
    --history-dir out/data \
    --universe-csv out/universe.csv \
    --ohlc-csv out/analysis/ml_ohlc_next_session.csv \
    --range-full-csv out/analysis/ml_range_predictions_full_2y.csv \
    --range-recent-csv out/analysis/ml_range_predictions_recent_focus.csv \
    --cycle-csv out/analysis/ml_cycle_forecast/cycle_forecast_best_horizon_by_ticker.csv \
    --single-name-csv out/analysis/ml_single_name_timing.csv \
    --output-dir out/analysis \
    "$@"
}

run_candidate_report() {
  local mode="${1:-auto}"
  shift || true
  run_module candidates scripts.analysis.build_candidate_watchlist_report \
    --mode "$mode" \
    "$@"
}

run_deep_dive_report() {
  local ticker="${1:-}"
  if [[ -z "$ticker" ]]; then
    echo "[deep] ERROR: ticker is required, e.g. ./broker.sh deep VIC" >&2
    exit 2
  fi
  shift || true
  run_module deep scripts.analysis.build_ticker_ml_deep_dive \
    --ticker "$ticker" \
    "$@"
}

run_research_bundle() {
  run_module research scripts.research.build_research_bundle \
    --universe-csv out/universe.csv \
    --market-summary-json out/market_summary.json \
    --sector-summary-csv out/sector_summary.csv \
    --analysis-dir out/analysis \
    --intraday-dir out/data/intraday_5m \
    --research-dir research \
    "$@"
}

run_refresh_vn30_map() {
  run_module refresh_vn30_map scripts.tools.refresh_industry_map \
    --from-vietstock-profiles-vn30 \
    --output data/industry_map.csv \
    "$@"
}

run_refresh_vn30_nvl_map() {
  run_module refresh_vn30_nvl_map scripts.tools.refresh_industry_map \
    --from-vietstock-profiles-vn30 \
    --extra-ticker NVL \
    --output data/industry_map.csv \
    "$@"
}

run_refresh_hose_map() {
  run_module refresh_hose_map scripts.tools.refresh_industry_map \
    --from-vietstock-profiles-hose \
    --output data/industry_map.csv \
    "$@"
}

run_sync_artifacts() {
  local prefix="${1:-core-artifacts-}"
  shift || true
  run_module sync_artifacts scripts.tools.sync_action_artifacts \
    --artifact-prefix "$prefix" \
    "$@"
}

run_eval_deterministic() {
  run_module eval_deterministic scripts.analysis.evaluate_deterministic_strategies "$@"
}

run_eval_ml() {
  run_module eval_ml scripts.analysis.evaluate_ml_models "$@"
}

run_eval_vnindex() {
  run_module eval_vnindex scripts.analysis.evaluate_vnindex_models "$@"
}

run_eval_ohlc() {
  run_module eval_ohlc scripts.analysis.evaluate_ohlc_models "$@"
}

run_eval_macro() {
  run_module eval_macro scripts.analysis.evaluate_macro_factor_sensitivity "$@"
}

run_eval_bctt() {
  run_module eval_bctt scripts.analysis.evaluate_bctt_feature_lift "$@"
}

run_prepare() {
  local config_path="${1:-$DEFAULT_CONFIG_PATH}"
  run_engine "$config_path"
  clean_notebook_artifacts
  run_playbook_report
  run_candidate_report core
  run_ohlc_report
  run_timing_report
  run_cycle_report
  run_range_report
  run_intraday_report
  run_entry_ladder_report
  run_research_bundle
  run_candidate_report full
}

run_prepare_core() {
  local config_path="${1:-$DEFAULT_CONFIG_PATH}"
  run_engine "$config_path"
  clean_notebook_artifacts
  run_playbook_report
  run_candidate_report core
  run_ohlc_report
}

run_prepare_default() {
  local config_path="${1:-$DEFAULT_CONFIG_PATH}"
  run_refresh_vn30_nvl_map
  run_prepare "$config_path"
}

run_tests() {
  ensure_venv
  echo "[tests] Using: $PY_BIN"
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py"
}

usage() {
  cat <<'EOF'
Usage: ./broker.sh <command> [args]

Core commands:
  tests                Run the unit test suite
  engine [config]      Build out/universe.csv, positions.csv, market_summary.json, sector_summary.csv
  prepare_core [config] Run engine + core live reports for interactive screening
  prepare [config]     Run engine + all live report builders + research bundle
  research             Rebuild research/ from the current out/ snapshot

Universe helpers:
  map                  Short alias for refresh_vn30_nvl_map
  refresh_vn30_map     Rebuild data/industry_map.csv from the live VN30 basket via Vietstock profiles
  refresh_vn30_nvl_map Rebuild data/industry_map.csv from the live VN30 basket plus NVL
  refresh_hose_map     Rebuild data/industry_map.csv from the live HOSE basket via Vietstock profiles
  sync_artifacts [prefix] Download/cache the latest GitHub Actions artifact matching a prefix
  prepare_default      Refresh VN30 + NVL scope, then run the full prepare pipeline sequentially

Report builders:
  candidates [mode]    Build ranked candidate watchlist (`auto`, `core`, or `full`)
  deep <ticker>        Build a deep per-ticker ML + research synthesis report
  range                Build range forecast reports
  cycle                Build cycle forecast reports
  playbook             Build per-ticker playbook report
  ohlc                 Build next-session OHLC report
  intraday             Build rest-of-session intraday report when the current timestamp allows it
  timing               Build single-name timing report
  entry_ladder         Build entry ladder evaluation report

Offline evaluation:
  eval_deterministic   Replay deterministic market/ticker strategies
  eval_ml              Run cross-sectional ML baseline
  eval_vnindex         Run VNINDEX ML baseline
  eval_ohlc            Run multi-horizon OHLC ML baseline
  eval_macro           Run macro-factor sensitivity evaluation
  eval_bctt            Run BCTT feature-lift evaluation
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "$cmd" in
    help|-h|--help)
      usage
      ;;
    tests)
      run_tests
      ;;
    engine)
      run_engine "${1:-$DEFAULT_CONFIG_PATH}"
      ;;
    prepare_core)
      run_prepare_core "${1:-$DEFAULT_CONFIG_PATH}"
      ;;
    prepare)
      run_prepare "${1:-$DEFAULT_CONFIG_PATH}"
      ;;
    research)
      run_research_bundle "$@"
      ;;
    refresh_vn30_map)
      run_refresh_vn30_map "$@"
      ;;
    map)
      run_refresh_vn30_nvl_map "$@"
      ;;
    refresh_vn30_nvl_map)
      run_refresh_vn30_nvl_map "$@"
      ;;
    refresh_hose_map)
      run_refresh_hose_map "$@"
      ;;
    sync_artifacts)
      run_sync_artifacts "$@"
      ;;
    prepare_default)
      run_prepare_default "${1:-$DEFAULT_CONFIG_PATH}"
      ;;
    deep)
      run_deep_dive_report "$@"
      ;;
    candidates)
      run_candidate_report "$@"
      ;;
    range)
      run_range_report "$@"
      ;;
    cycle)
      run_cycle_report "$@"
      ;;
    playbook)
      run_playbook_report "$@"
      ;;
    ohlc)
      run_ohlc_report "$@"
      ;;
    intraday)
      run_intraday_report "$@"
      ;;
    timing)
      run_timing_report "$@"
      ;;
    entry_ladder)
      run_entry_ladder_report "$@"
      ;;
    eval_deterministic)
      run_eval_deterministic "$@"
      ;;
    eval_ml)
      run_eval_ml "$@"
      ;;
    eval_vnindex)
      run_eval_vnindex "$@"
      ;;
    eval_ohlc)
      run_eval_ohlc "$@"
      ;;
    eval_macro)
      run_eval_macro "$@"
      ;;
    eval_bctt)
      run_eval_bctt "$@"
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
