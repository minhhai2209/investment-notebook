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
    "vic_single_model_current": "out/analysis/vic_single_model_current.csv",
    "vic_single_model_candidates": "out/analysis/vic_single_model_candidates.csv",
    "vic_single_model_holdout": "out/analysis/vic_single_model_holdout.csv",
    "vic_single_model_walkback": "out/analysis/vic_single_model_walkback.csv",
    "vic_single_model_feature_engineering": "out/analysis/vic_single_model_feature_engineering.csv",
    "vic_single_model_summary": "out/analysis/vic_single_model_summary.json",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


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

    single_model_path = output_dir / "vic" / "vic_single_model_current.csv"
    if single_model_path.exists():
        frame = pd.read_csv(single_model_path)
        if not frame.empty:
            rows = frame.sort_values("Horizon").to_dict(orient="records") if "Horizon" in frame.columns else frame.to_dict(orient="records")
            first = rows[0]
            summary["vic_single_model"] = {
                "snapshot_date": first.get("Date"),
                "feature_set": first.get("FeatureSet"),
                "model": first.get("Model"),
                "objective": first.get("Objective"),
                "rows": rows,
            }
    selector_summary_path = output_dir / "raw" / "vic_single_model_summary.json"
    if selector_summary_path.exists():
        selector_summary = json.loads(selector_summary_path.read_text(encoding="utf-8"))
        summary["selector_backtest"] = {
            "holdout_base_date": selector_summary.get("HoldoutBaseDate"),
            "holdout_actual_dates": selector_summary.get("HoldoutActualDates"),
            "horizons": selector_summary.get("Horizons"),
            "winner": selector_summary.get("Winner"),
        }
        summary["feature_engineering"] = {
            "feature_sets": selector_summary.get("FeatureSets"),
            "engineered_features": selector_summary.get("EngineeredFeatures"),
            "candidate_rows": selector_summary.get("CandidateRows"),
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
    (output_dir / "vic" / "summary.json").write_text(
        json.dumps(_json_safe(vic_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Active Model Reports",
                "",
                f"Generated at `{generated_at}`.",
                "",
                "This directory is overwritten by the local publish step.",
                "Detailed raw training histories remain in `out/analysis`; this repo path stores lightweight latest forecasts and metrics.",
                "",
                "- `vic/summary.json`: compact VIC forecast summary.",
                "- `vic/vic_single_model_current.csv`: the only active VIC model output.",
                "- `vic/vic_single_model_holdout.csv`: last-5-session holdout backtest.",
                "- `vic/vic_single_model_walkback.csv`: pre-holdout walkback backtest.",
                "- `vic/vic_single_model_candidates.csv`: feature/model candidate audit table.",
                "- `vic/vic_single_model_feature_engineering.csv`: retained feature-set and engineered-feature audit table.",
                "- `raw/`: copied latest selector artifacts before filtering.",
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
