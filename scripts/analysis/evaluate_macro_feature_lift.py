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

from scripts.analysis.evaluate_bctt_feature_lift import _with_variant_columns
from scripts.analysis.evaluate_ml_models import (
    DEFAULT_CASE_TICKERS,
    _normalise_ticker,
    build_ml_case_studies,
    build_ml_sample,
    build_model_factories,
    numeric_feature_columns,
    summarise_ml_models,
)
from scripts.data_fetching.macro_factor_cache import DEFAULT_CACHE_DIR, load_macro_factor_matrix


DEFAULT_MACRO_CASE_TICKERS = ["VIC"]
MACRO_RET_WINDOWS = (1, 3, 5, 20)
MACRO_VOL_WINDOWS = (5, 20)


def _clean_factor_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).upper()).strip("_")


def build_macro_feature_frame(
    sample_df: pd.DataFrame,
    cache_dir: Path,
    factor_names: Sequence[str] | None = None,
    *,
    shift_days: int = 1,
) -> tuple[pd.DataFrame, List[str]]:
    factor_values = load_macro_factor_matrix(cache_dir, factor_names=factor_names)
    if factor_values.empty:
        raise RuntimeError(f"No cached macro factor CSVs found in {cache_dir}")

    factor_values = factor_values.sort_index().ffill()
    factor_values.index = pd.to_datetime(factor_values.index)
    lagged_values = factor_values.shift(int(shift_days))
    returns = lagged_values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100.0

    feature_map: Dict[str, pd.Series] = {}
    for factor_name in factor_values.columns:
        clean_name = _clean_factor_name(factor_name)
        for window in MACRO_RET_WINDOWS:
            feature_map[f"Macro_{clean_name}_Ret{window}Pct"] = returns[factor_name].rolling(window).sum()
        for window in MACRO_VOL_WINDOWS:
            feature_map[f"Macro_{clean_name}_Vol{window}Pct"] = returns[factor_name].rolling(window).std()

    macro_features = pd.DataFrame(feature_map).reset_index().rename(columns={"index": "Date"})
    macro_features["Date"] = pd.to_datetime(macro_features["Date"]).dt.normalize()

    keys = sample_df[["Date"]].drop_duplicates().copy()
    keys["Date"] = pd.to_datetime(keys["Date"]).dt.normalize()
    merged = keys.merge(macro_features, on="Date", how="left").sort_values("Date")
    merged = merged.ffill()
    macro_columns = [column for column in merged.columns if column != "Date"]
    return merged, macro_columns


def _best_by_feature_set(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    return (
        summary_df.sort_values(
            ["FeatureSet", "TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
            ascending=[True, False, False, False],
        )
        .groupby("FeatureSet", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def _build_lift_summary(best_by_feature: pd.DataFrame) -> pd.DataFrame:
    if best_by_feature.empty or "baseline" not in set(best_by_feature["FeatureSet"]):
        return pd.DataFrame()
    baseline = best_by_feature[best_by_feature["FeatureSet"] == "baseline"].iloc[0]
    rows: List[Dict[str, object]] = []
    for _, row in best_by_feature.iterrows():
        rows.append(
            {
                "FeatureSet": row["FeatureSet"],
                "BestModel": row["Model"],
                "BaselineBestModel": baseline["Model"],
                "TopKAvgExcess10Pct": row["TopKAvgExcess10Pct"],
                "TopKAvgExcess10PctLift": row["TopKAvgExcess10Pct"] - baseline["TopKAvgExcess10Pct"],
                "TopKHit10Pct": row["TopKHit10Pct"],
                "TopKHit10PctLift": row["TopKHit10Pct"] - baseline["TopKHit10Pct"],
                "AUC": row["AUC"],
                "AUCLift": row["AUC"] - baseline["AUC"],
            }
        )
    return pd.DataFrame(rows)


def walk_forward_predict_variant(
    sample_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    min_train_dates: int,
    retrain_every: int,
    variant_name: str,
    model_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = list(numeric_columns) + ["Sector"]
    labeled = sample_df[sample_df["TargetOutperform10d"].notna()].copy()
    unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
    eval_dates = unique_dates[min_train_dates:]
    latest_date = sample_df["Date"].max()
    current_rows = sample_df[sample_df["Date"] == latest_date].copy()

    available_factories = build_model_factories(numeric_columns, ["Sector"])
    factories = {name: available_factories[name] for name in model_names if name in available_factories}
    if not factories:
        raise ValueError(f"No valid model names selected. Available: {sorted(available_factories)}")

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


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    macro_cache_dir: Path,
    output_dir: Path,
    top_k: int,
    min_train_dates: int,
    retrain_every: int,
    case_tickers: Sequence[str],
    factor_names: Sequence[str] | None,
    shift_days: int,
    model_names: Sequence[str],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_sample = build_ml_sample(history_dir, sector_map_path)
    case_ticker_set = {_normalise_ticker(ticker) for ticker in case_tickers}
    if case_ticker_set:
        base_sample = base_sample[base_sample["Ticker"].isin(case_ticker_set)].copy()
    base_sample["Date"] = pd.to_datetime(base_sample["Date"]).dt.normalize()
    base_numeric = numeric_feature_columns()

    macro_feature_dates, macro_columns = build_macro_feature_frame(
        sample_df=base_sample,
        cache_dir=macro_cache_dir,
        factor_names=factor_names,
        shift_days=shift_days,
    )
    macro_sample = base_sample.merge(macro_feature_dates, on="Date", how="left")

    variants = {
        "baseline": (base_sample, list(base_numeric)),
        "macro_global": (macro_sample, list(base_numeric) + macro_columns),
    }
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []
    for variant_name, (sample_df, numeric_columns) in variants.items():
        history_df, current_df = walk_forward_predict_variant(
            sample_df=sample_df,
            numeric_columns=numeric_columns,
            min_train_dates=min_train_dates,
            retrain_every=retrain_every,
            variant_name=variant_name,
            model_names=model_names,
        )
        if not history_df.empty:
            history_frames.append(history_df)
        if not current_df.empty:
            current_frames.append(current_df)

    prediction_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_predictions = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    summary_df = _with_variant_columns(summarise_ml_models(prediction_history, current_predictions, top_k))
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
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
    best_by_feature = _best_by_feature_set(summary_df)
    lift_summary = _build_lift_summary(best_by_feature)

    summary_df.to_csv(output_dir / "ml_macro_feature_summary.csv", index=False)
    prediction_history.to_csv(output_dir / "ml_macro_prediction_history.csv", index=False)
    current_predictions.to_csv(output_dir / "ml_macro_current_predictions.csv", index=False)
    case_studies.to_csv(output_dir / "ml_macro_case_studies.csv", index=False)
    best_by_feature.to_csv(output_dir / "ml_macro_best_by_feature_set.csv", index=False)
    lift_summary.to_csv(output_dir / "ml_macro_lift_summary.csv", index=False)

    payload = {
        "feature_sets": list(variants.keys()),
        "macro_feature_count": len(macro_columns),
        "macro_features": macro_columns,
        "case_tickers": list(case_tickers),
        "top_k": int(top_k),
        "min_train_dates": int(min_train_dates),
        "retrain_every": int(retrain_every),
        "shift_days": int(shift_days),
        "model_names": list(model_names),
        "best_by_feature_set": best_by_feature.to_dict(orient="records"),
        "lift_summary": lift_summary.to_dict(orient="records"),
    }
    (output_dir / "ml_macro_feature_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MacroFeatureLift")
    if lift_summary.empty:
        print("No lift results.")
    else:
        print(
            lift_summary[
                [
                    "FeatureSet",
                    "BestModel",
                    "TopKAvgExcess10Pct",
                    "TopKAvgExcess10PctLift",
                    "TopKHit10Pct",
                    "TopKHit10PctLift",
                    "AUC",
                    "AUCLift",
                ]
            ].to_string(index=False)
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ML variants with and without cached macro/global equity features.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV.")
    parser.add_argument("--macro-cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory for macro factor CSVs.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write evaluation reports.")
    parser.add_argument("--top-k", default=3, type=int, help="Top N tickers to evaluate per model per day.")
    parser.add_argument("--min-train-dates", default=80, type=int, help="Minimum labeled dates before walk-forward starts.")
    parser.add_argument("--retrain-every", default=5, type=int, help="Retrain cadence in trading days.")
    parser.add_argument("--case-tickers", nargs="*", default=DEFAULT_MACRO_CASE_TICKERS, help="Tickers to include.")
    parser.add_argument("--factor-names", nargs="*", default=None, help="Optional macro factors to include from cache.")
    parser.add_argument("--shift-days", default=1, type=int, help="Lag macro values by this many factor observations to reduce leakage.")
    parser.add_argument(
        "--model-names",
        nargs="*",
        default=["hist_gbm", "logistic_balanced"],
        help="Classifier models to compare. Use random_forest only for slower research runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        history_dir=Path(args.history_dir),
        sector_map_path=Path(args.sector_map),
        macro_cache_dir=Path(args.macro_cache_dir),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        min_train_dates=int(args.min_train_dates),
        retrain_every=int(args.retrain_every),
        case_tickers=[_normalise_ticker(ticker) for ticker in args.case_tickers or DEFAULT_CASE_TICKERS],
        factor_names=[str(name).strip().upper() for name in args.factor_names] if args.factor_names else None,
        shift_days=int(args.shift_days),
        model_names=[str(name).strip() for name in args.model_names],
    )


if __name__ == "__main__":
    main()
