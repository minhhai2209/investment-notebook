from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_ml_models import (
    DEFAULT_CASE_TICKERS,
    _normalise_ticker,
    build_ml_case_studies,
    build_ml_sample,
    build_model_factories,
    numeric_feature_columns,
    summarise_ml_models,
)
from scripts.data_fetching.vietstock_bctt_api import FEATURE_SETS, build_daily_bctt_feature_frame


DEFAULT_BCTT_CACHE_DIR = REPO_ROOT / "out" / "vietstock_bctt"


def walk_forward_predict_variant(
    sample_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    top_k: int,
    min_train_dates: int,
    retrain_every: int,
    variant_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = list(numeric_columns) + ["Sector"]
    labeled = sample_df[sample_df["TargetOutperform10d"].notna()].copy()
    unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
    eval_dates = unique_dates[min_train_dates:]
    latest_date = sample_df["Date"].max()
    current_rows = sample_df[sample_df["Date"] == latest_date].copy()

    factories = build_model_factories(numeric_columns, ["Sector"])
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

    for model_name, factory in factories.items():
        scoped_name = f"{variant_name}::{model_name}"
        block_predictions: List[pd.DataFrame] = []
        for start in range(0, len(eval_dates), retrain_every):
            block_dates = eval_dates[start : start + retrain_every]
            if not block_dates:
                continue
            train_df = labeled[labeled["Date"] < block_dates[0]].copy()
            if train_df["TargetOutperform10d"].nunique(dropna=True) < 2:
                continue
            block_df = labeled[labeled["Date"].isin(block_dates)].copy()
            model = factory()
            model.fit(train_df[feature_columns], train_df["TargetOutperform10d"].astype(int))
            probs = model.predict_proba(block_df[feature_columns])[:, 1]
            out = block_df[
                [
                    "Date",
                    "Ticker",
                    "Sector",
                    "Fwd10Pct",
                    "Excess10Pct",
                    "TargetOutperform10d",
                    "Rel20Pct",
                    "Rel60Pct",
                    "DistSMA20Pct",
                    "SectorBreadth20Pct",
                ]
            ].copy()
            out["Model"] = scoped_name
            out["ProbabilityOutperform10d"] = probs
            block_predictions.append(out)

        if block_predictions:
            history_frames.append(pd.concat(block_predictions, ignore_index=True))

        train_all = labeled[labeled["Date"] < latest_date].copy()
        if train_all["TargetOutperform10d"].nunique(dropna=True) < 2:
            continue
        final_model = factory()
        final_model.fit(train_all[feature_columns], train_all["TargetOutperform10d"].astype(int))
        current_probs = final_model.predict_proba(current_rows[feature_columns])[:, 1]
        current_out = current_rows[
            [
                "Date",
                "Ticker",
                "Sector",
                "Rel20Pct",
                "Rel60Pct",
                "DistSMA20Pct",
                "SectorBreadth20Pct",
            ]
        ].copy()
        current_out["Model"] = scoped_name
        current_out["ProbabilityOutperform10d"] = current_probs
        current_out = current_out.sort_values("ProbabilityOutperform10d", ascending=False).reset_index(drop=True)
        current_out["Rank"] = np.arange(1, len(current_out) + 1)
        current_frames.append(current_out)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    return history_df, current_df


def _with_variant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    parts = out["Model"].astype(str).str.split("::", n=1, expand=True)
    out["FeatureSet"] = parts[0]
    out["BaseModel"] = parts[1]
    return out


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    bctt_cache_dir: Path,
    output_dir: Path,
    top_k: int,
    min_train_dates: int,
    retrain_every: int,
    case_tickers: Sequence[str],
    feature_sets: Sequence[str],
    headless: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_sample = build_ml_sample(history_dir, sector_map_path)
    case_ticker_set = {_normalise_ticker(ticker) for ticker in case_tickers}
    base_sample = base_sample[base_sample["Ticker"].isin(case_ticker_set)].copy()
    base_numeric = numeric_feature_columns()

    variant_samples: Dict[str, pd.DataFrame] = {"baseline": base_sample.copy()}
    for feature_set in feature_sets:
        if feature_set == "baseline":
            continue
        daily_bctt = build_daily_bctt_feature_frame(
            sample_df=base_sample[["Date", "Ticker"]].copy(),
            cache_dir=bctt_cache_dir,
            feature_set=feature_set,
            max_age_hours=0,
            headless=headless,
        )
        merged = base_sample.merge(daily_bctt, on=["Date", "Ticker"], how="left")
        variant_samples[feature_set] = merged

    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []
    for variant_name in feature_sets:
        sample_df = variant_samples[variant_name]
        numeric_cols = list(base_numeric)
        if variant_name != "baseline":
            numeric_cols.extend(FEATURE_SETS[variant_name])
        history_df, current_df = walk_forward_predict_variant(
            sample_df=sample_df,
            numeric_columns=numeric_cols,
            top_k=top_k,
            min_train_dates=min_train_dates,
            retrain_every=retrain_every,
            variant_name=variant_name,
        )
        if not history_df.empty:
            history_frames.append(history_df)
        if not current_df.empty:
            current_frames.append(current_df)

    prediction_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_predictions = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    summary_df = _with_variant_columns(summarise_ml_models(prediction_history, current_predictions, top_k))
    summary_df = summary_df.sort_values(
        ["TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_model = str(summary_df.iloc[0]["Model"]) if not summary_df.empty else ""

    case_studies = _with_variant_columns(
        build_ml_case_studies(
            prediction_history=prediction_history,
            current_predictions=current_predictions,
            case_tickers=case_tickers,
            top_k=top_k,
        )
    )
    current_predictions = _with_variant_columns(current_predictions)
    prediction_history = _with_variant_columns(prediction_history)
    current_top_picks = (
        current_predictions.sort_values(["Model", "ProbabilityOutperform10d"], ascending=[True, False])
        .groupby("Model", as_index=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    best_by_feature = (
        summary_df.sort_values(
            ["FeatureSet", "TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
            ascending=[True, False, False, False],
        )
        .groupby("FeatureSet", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    summary_out = summary_df.copy()
    current_out = current_predictions.copy()
    history_out = prediction_history.copy()
    top_picks_out = current_top_picks.copy()
    case_out = case_studies.copy()
    for frame in (current_out, history_out, top_picks_out):
        if not frame.empty and "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.date.astype(str)

    summary_out.to_csv(output_dir / "ml_bctt_feature_summary.csv", index=False)
    history_out.to_csv(output_dir / "ml_bctt_prediction_history.csv", index=False)
    current_out.to_csv(output_dir / "ml_bctt_current_predictions.csv", index=False)
    top_picks_out.to_csv(output_dir / "ml_bctt_current_top_picks.csv", index=False)
    case_out.to_csv(output_dir / "ml_bctt_case_studies.csv", index=False)
    best_by_feature.to_csv(output_dir / "ml_bctt_best_by_feature_set.csv", index=False)

    payload = {
        "best_model": best_model,
        "feature_sets": list(feature_sets),
        "best_by_feature_set": best_by_feature.to_dict(orient="records"),
        "case_tickers": list(case_tickers),
        "top_k": top_k,
        "min_train_dates": min_train_dates,
        "retrain_every": retrain_every,
    }
    (output_dir / "ml_bctt_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("BCTTFeatureLift")
    if summary_df.empty:
        print("No results.")
    else:
        print(
            summary_df[
                [
                    "FeatureSet",
                    "BaseModel",
                    "TopKAvgFwd10Pct",
                    "TopKAvgExcess10Pct",
                    "TopKHit10Pct",
                    "AUC",
                ]
            ].to_string(index=False)
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ML variants with and without Vietstock BCTT features.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV.")
    parser.add_argument("--bctt-cache-dir", default=str(DEFAULT_BCTT_CACHE_DIR), help="Cache directory for Vietstock BCTT JSON.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write evaluation reports.")
    parser.add_argument("--top-k", default=5, type=int, help="Top N tickers to evaluate per model per day.")
    parser.add_argument("--min-train-dates", default=80, type=int, help="Minimum labeled dates before walk-forward starts.")
    parser.add_argument("--retrain-every", default=5, type=int, help="Retrain cadence in trading days.")
    parser.add_argument(
        "--case-tickers",
        nargs="*",
        default=DEFAULT_CASE_TICKERS,
        help="Tickers to include in the case study report.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=["baseline", "ratios_only", "hybrid_growth"],
        help="Feature-set variants to compare. baseline is always valid.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show Playwright browser while scraping Vietstock BCTT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        history_dir=Path(args.history_dir),
        sector_map_path=Path(args.sector_map),
        bctt_cache_dir=Path(args.bctt_cache_dir),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        min_train_dates=int(args.min_train_dates),
        retrain_every=int(args.retrain_every),
        case_tickers=[_normalise_ticker(ticker) for ticker in args.case_tickers],
        feature_sets=[str(name) for name in args.feature_sets],
        headless=not bool(args.show_browser),
    )


if __name__ == "__main__":
    main()
