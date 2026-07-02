from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd


DEFAULT_HUB_DIR = Path("data-hub/latest")
DEFAULT_TICKER = "VIC"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _file_check(hub_dir: Path, rel_path: str) -> Dict[str, Any]:
    path = hub_dir / rel_path
    row: Dict[str, Any] = {
        "path": rel_path,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists() and path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
            row["rows"] = int(len(frame))
            row["columns"] = int(len(frame.columns))
        except Exception as exc:
            row["csv_error"] = str(exc)
    return row


def validate_layout(hub_dir: Path = DEFAULT_HUB_DIR, ticker: str = DEFAULT_TICKER) -> Dict[str, Any]:
    hub_dir = Path(hub_dir)
    ticker = ticker.strip().upper()
    failures: List[str] = []
    warnings: List[str] = []

    start_path = hub_dir / "START_HERE.json"
    manifest_path = hub_dir / "manifest.json"
    if not start_path.exists():
        failures.append("Missing START_HERE.json")
        start = {}
    else:
        start = _read_json(start_path)
    if not manifest_path.exists():
        failures.append("Missing manifest.json")
        manifest = {}
    else:
        manifest = _read_json(manifest_path)

    minimal_read_order = list(start.get("minimal_read_order") or [])
    minimal_files = [_file_check(hub_dir, rel_path) for rel_path in minimal_read_order]
    missing_minimal = [item["path"] for item in minimal_files if not item["exists"]]
    if missing_minimal:
        failures.append(f"Missing minimal read-order files: {', '.join(missing_minimal)}")

    source_audit_path = hub_dir / "bundles" / "source_audit.csv"
    if source_audit_path.exists():
        source_audit = pd.read_csv(source_audit_path)
        source_rows = int(len(source_audit))
        source_status_counts = source_audit["Status"].fillna("").astype(str).value_counts().to_dict() if "Status" in source_audit.columns else {}
        if "Status" in source_audit.columns:
            source_errors = source_audit[source_audit["Status"].fillna("").astype(str).eq("error")]
            if not source_errors.empty:
                warnings.append(f"{len(source_errors)} source audit row(s) report error; see bundles/source_audit.csv")
    else:
        source_rows = 0
        source_status_counts = {}
        failures.append("Missing bundles/source_audit.csv")

    ticker_catalog_path = hub_dir / "index" / "ticker_catalog.csv"
    symbol_latest_path = hub_dir / "bundles" / "symbol_latest.csv"
    ticker_drilldown: Dict[str, Any] = {"ticker": ticker}
    if ticker_catalog_path.exists():
        ticker_catalog = pd.read_csv(ticker_catalog_path)
        selected = ticker_catalog[ticker_catalog["Ticker"].astype(str).str.upper().eq(ticker)] if "Ticker" in ticker_catalog.columns else pd.DataFrame()
        if selected.empty:
            failures.append(f"Ticker {ticker} not found in index/ticker_catalog.csv")
        else:
            row = selected.iloc[0].to_dict()
            ticker_drilldown["catalog_row"] = {str(k): _json_safe(v) for k, v in row.items()}
            for key in ("DailyPath", "IntradayPath", "MinuteProfilePath"):
                rel_path = str(row.get(key) or "")
                if rel_path:
                    ticker_drilldown[key] = _file_check(hub_dir, rel_path)
                    if not ticker_drilldown[key]["exists"]:
                        failures.append(f"Ticker {ticker} has missing {key}: {rel_path}")
    else:
        failures.append("Missing index/ticker_catalog.csv")

    if symbol_latest_path.exists():
        symbol_latest = pd.read_csv(symbol_latest_path)
        selected = symbol_latest[symbol_latest["Ticker"].astype(str).str.upper().eq(ticker)] if "Ticker" in symbol_latest.columns else pd.DataFrame()
        if selected.empty:
            failures.append(f"Ticker {ticker} not found in bundles/symbol_latest.csv")
        else:
            compact_columns = [
                column
                for column in ["Ticker", "LatestDate", "LastClose", "Ret1dPct", "Ret20dPct", "DailyVolume", "IntradayLast"]
                if column in selected.columns
            ]
            ticker_drilldown["symbol_latest"] = selected[compact_columns].iloc[0].to_dict()
    else:
        failures.append("Missing bundles/symbol_latest.csv")

    file_catalog_path = hub_dir / "index" / "file_catalog.csv"
    if file_catalog_path.exists():
        file_catalog = pd.read_csv(file_catalog_path)
        file_catalog_rows = int(len(file_catalog))
        expected_paths = set(minimal_read_order + ["manifest.json", "latest_metrics.csv"])
        expected_paths.discard("index/file_catalog.csv")
        present_paths = set(file_catalog["Path"].astype(str)) if "Path" in file_catalog.columns else set()
        missing_from_catalog = sorted(expected_paths - present_paths)
        if missing_from_catalog:
            failures.append(f"index/file_catalog.csv does not list: {', '.join(missing_from_catalog)}")
    else:
        file_catalog_rows = 0
        failures.append("Missing index/file_catalog.csv")

    output: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not failures else "error",
        "hub_dir": str(hub_dir),
        "ticker": ticker,
        "minimal_read_order": minimal_read_order,
        "minimal_file_checks": minimal_files,
        "manifest_generated_at": manifest.get("generated_at"),
        "source_audit": {
            "rows": source_rows,
            "status_counts": source_status_counts,
        },
        "file_catalog_rows": file_catalog_rows,
        "ticker_drilldown": ticker_drilldown,
        "warnings": warnings,
        "failures": failures,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the data hub layout as a repo/Drive connector would read it.")
    parser.add_argument("--hub-dir", type=Path, default=DEFAULT_HUB_DIR)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--output", type=Path, default=None, help="JSON output path. Defaults to <hub-dir>/bundles/retrieval_smoke_test.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_layout(args.hub_dir, args.ticker)
    output = args.output or (args.hub_dir / "bundles" / "retrieval_smoke_test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
