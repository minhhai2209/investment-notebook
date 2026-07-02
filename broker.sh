#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python3}"
if [[ -x "venv/bin/python" ]]; then
  PY_BIN="venv/bin/python"
fi

run_module() {
  local label="$1"
  shift
  echo "[$label] Using: $PY_BIN"
  "$PY_BIN" -m "$@"
}

run_hub() {
  run_module hub scripts.data_hub.build_numeric_data_hub "$@"
}

run_collect() {
  run_module collect scripts.data_hub.build_numeric_data_hub --refresh "$@"
}

run_refresh_macro() {
  run_module refresh_macro scripts.data_fetching.macro_factor_cache \
    --config config/macro_factors.yaml \
    --cache-dir out/macro_factors \
    --max-age-hours 24 \
    --timeout 8 \
    --summary-out out/data_hub/macro_factor_cache_summary.csv \
    "$@"
}

run_refresh_bctt() {
  run_module refresh_bctt scripts.data_fetching.refresh_vietstock_bctt_cache "$@"
}

run_refresh_vic_map() {
  run_module refresh_vic_map scripts.tools.refresh_industry_map "$@"
}

run_tests() {
  echo "[tests] Using: $PY_BIN"
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py"
}

usage() {
  cat <<'EOF'
Usage: ./broker.sh <command> [args]

Numeric data hub:
  hub                  Build data-hub/latest from existing caches and enabled config sources
  collect              Refresh enabled numeric sources, then rebuild data-hub/latest

Source helpers:
  refresh_macro        Refresh configured FRED/Stooq numeric macro caches
  refresh_bctt         Refresh Vietstock BCTT numeric financial-statement caches
  refresh_vic_map      Refresh Vietstock industry map

Maintenance:
  tests                Run unit tests

Notes:
  - This repo is numeric-data only.
  - No news collection, no portfolio workflow, no forecast/model report surface.
  - Start ChatGPT browsing from data-hub/latest/manifest.json.
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    hub)
      run_hub "$@"
      ;;
    collect)
      run_collect "$@"
      ;;
    refresh_macro)
      run_refresh_macro "$@"
      ;;
    refresh_bctt)
      run_refresh_bctt "$@"
      ;;
    refresh_vic_map|map)
      run_refresh_vic_map "$@"
      ;;
    tests)
      run_tests
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
