from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_OUTPUT_DIR = Path("reports/active-models/latest")
DEFAULT_TICKER = "VIC"

REPORT_FILES = {
    "universe": "out/universe.csv",
    "ohlc_next_session": "out/analysis/ml_ohlc_next_session.csv",
    "ohlc_multi_session": "out/analysis/ml_ohlc_multi_session.csv",
    "ohlc_model_metrics": "out/analysis/ml_ohlc_model_metrics.csv",
    "intraday_forecast": "out/analysis/ml_intraday_rest_of_session.csv",
    "intraday_metrics": "out/analysis/ml_intraday_rest_of_session_metrics.csv",
    "curated_intraday_current": "out/analysis/curated_intraday_model_locked/curated_intraday_model_current.csv",
    "curated_intraday_metrics": "out/analysis/curated_intraday_model_locked/curated_intraday_model_metrics.csv",
    "curated_intraday_summary": "out/analysis/curated_intraday_model_locked/curated_intraday_model_summary.json",
    "vic_index_expiry_current": "out/analysis/vic_index_expiry_current_forecasts.csv",
    "vic_index_expiry_metrics": "out/analysis/vic_index_expiry_model_metrics.csv",
    "vic_index_expiry_summary": "out/analysis/vic_index_expiry_summary.json",
}


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _filter_csv(source: Path, destination: Path, ticker: str) -> Dict[str, object] | None:
    if not source.exists() or source.suffix.lower() != ".csv":
        return None
    frame = pd.read_csv(source)
    filtered = frame.copy()
    if "Ticker" in filtered.columns:
        filtered = filtered[filtered["Ticker"].astype(str).str.upper().eq(ticker)]
    if "Scope" in filtered.columns:
        scope = filtered["Scope"].astype(str).str.upper()
        filtered = filtered[scope.isin({ticker, "ALL"})]
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(destination, index=False)
    return {
        "source_rows": int(frame.shape[0]),
        "ticker_rows": int(filtered.shape[0]),
        "columns": list(filtered.columns),
    }


def _build_vic_summary(output_dir: Path, ticker: str) -> Dict[str, object]:
    summary: Dict[str, object] = {"ticker": ticker}

    ohlc_path = output_dir / "vic" / "ohlc_next_session.csv"
    if ohlc_path.exists():
        frame = pd.read_csv(ohlc_path)
        if not frame.empty:
            row = frame.iloc[0].to_dict()
            summary["ohlc_t1"] = {
                "snapshot_date": row.get("SnapshotDate"),
                "base": row.get("Base"),
                "forecast_open": row.get("ForecastOpen"),
                "forecast_low": row.get("ForecastLow"),
                "forecast_high": row.get("ForecastHigh"),
                "forecast_close": row.get("ForecastClose"),
                "forecast_close_ret_pct": row.get("ForecastCloseRetPct"),
                "forecast_cum_high_ret_pct": row.get("ForecastCumHighRetPct"),
                "forecast_cum_low_ret_pct": row.get("ForecastCumLowRetPct"),
                "model": row.get("Model"),
                "selection_score": row.get("SelectionScore"),
                "close_mae_pct": row.get("CloseMAEPct"),
                "close_dir_hit_pct": row.get("CloseDirHitPct"),
            }

    expiry_path = output_dir / "vic" / "vic_index_expiry_current.csv"
    if expiry_path.exists():
        frame = pd.read_csv(expiry_path)
        t1 = frame[pd.to_numeric(frame.get("Horizon"), errors="coerce").eq(1)] if "Horizon" in frame else frame
        if not t1.empty:
            row = t1.iloc[0].to_dict()
            summary["index_expiry_t1"] = {
                "snapshot_date": row.get("Date"),
                "base": row.get("BaseClose"),
                "feature_set": row.get("FeatureSet"),
                "model": row.get("Model"),
                "pred_open": row.get("PredOpen"),
                "pred_low": row.get("PredLow"),
                "pred_high": row.get("PredHigh"),
                "pred_close": row.get("PredClose"),
                "pred_close_ret_pct": row.get("PredCloseRetPct"),
                "pred_cum_high_ret_pct": row.get("PredCumHighRetPct"),
                "pred_cum_low_ret_pct": row.get("PredCumLowRetPct"),
            }

    curated_path = output_dir / "vic" / "curated_intraday_current.csv"
    if curated_path.exists():
        frame = pd.read_csv(curated_path)
        if not frame.empty:
            row = frame.iloc[0].to_dict()
            summary["curated_intraday_latest"] = {
                "snapshot_date": row.get("SnapshotDate"),
                "snapshot_bucket": row.get("SnapshotTimeBucket"),
                "base": row.get("Base"),
                "feature_set": row.get("FeatureSet"),
                "model": row.get("Model"),
                "pred_low": row.get("PredLow"),
                "pred_close": row.get("PredClose"),
                "pred_high": row.get("PredHigh"),
            }
    return summary


def publish_reports(*, source_root: Path, output_dir: Path, ticker: str) -> Dict[str, object]:
    ticker = str(ticker).strip().upper()
    _clean_dir(output_dir)
    raw_dir = output_dir / "raw"
    vic_dir = output_dir / "vic"

    copied: List[Dict[str, object]] = []
    filtered: Dict[str, object] = {}
    missing: List[str] = []
    for label, relative_source in REPORT_FILES.items():
        source = source_root / relative_source
        destination = raw_dir / Path(relative_source).name
        if _copy_file(source, destination):
            copied.append({"label": label, "source": relative_source, "path": str(destination)})
            stats = _filter_csv(source, vic_dir / f"{label}.csv", ticker)
            if stats is not None:
                filtered[label] = stats
        else:
            missing.append(relative_source)

    generated_at = datetime.now(VN_TZ).isoformat()
    manifest = {
        "generated_at": generated_at,
        "ticker": ticker,
        "github": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "sha": os.environ.get("GITHUB_SHA"),
            "ref": os.environ.get("GITHUB_REF_NAME"),
            "event": os.environ.get("GITHUB_EVENT_NAME"),
        },
        "copied": copied,
        "filtered": filtered,
        "missing_optional": missing,
    }
    vic_summary = _build_vic_summary(output_dir, ticker)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "vic" / "summary.json").write_text(json.dumps(vic_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Active Model Reports",
                "",
                f"Generated at `{generated_at}`.",
                "",
                "This directory is overwritten by the scheduled GitHub Action.",
                "Detailed raw training histories remain in GitHub Action artifacts; this repo path stores lightweight latest forecasts and metrics.",
                "",
                "- `vic/summary.json`: compact VIC forecast summary.",
                "- `vic/*.csv`: VIC-filtered forecast and metric tables.",
                "- `raw/`: copied latest model artifacts before filtering.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish latest model artifacts into a tracked reports directory.")
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = publish_reports(source_root=args.source_root, output_dir=args.output_dir, ticker=args.ticker)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
