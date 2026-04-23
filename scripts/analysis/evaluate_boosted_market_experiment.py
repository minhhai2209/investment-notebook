from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_deep_market_experiment import (  # noqa: E402
    _prepare_history,
    build_baseline_model_factories,
    build_comparison_summary,
    make_preprocessor,
    summarise_model_group,
    walk_forward_market_predictions,
)
from scripts.analysis.evaluate_vnindex_models import _normalise_symbol, build_vnindex_sample  # noqa: E402


def build_boosted_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "xgb_small": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=4,
                        reg_lambda=1.0,
                        reg_alpha=0.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgb_mid": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=6,
                        reg_lambda=1.5,
                        reg_alpha=0.1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    output_dir: Path,
    min_train_dates: int,
    retrain_every: int,
    market_symbol: str = "VNINDEX",
) -> Dict[str, object]:
    market_symbol = _normalise_symbol(market_symbol) or "VNINDEX"
    _prepare_history(market_symbol, history_dir)
    sample_df = build_vnindex_sample(history_dir, sector_map_path, market_symbol=market_symbol)

    baseline_history, baseline_current = walk_forward_market_predictions(
        sample_df=sample_df,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
        model_factories=build_baseline_model_factories(),
    )
    boosted_history, boosted_current = walk_forward_market_predictions(
        sample_df=sample_df,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
        model_factories=build_boosted_model_factories(),
    )

    baseline_summary = summarise_model_group(baseline_history, group_label="baseline")
    boosted_summary = summarise_model_group(boosted_history, group_label="boosted")
    summary_df = pd.concat([baseline_summary, boosted_summary], ignore_index=True).sort_values(
        ["HorizonDays", "ModelGroup", "AUC", "Brier"],
        ascending=[True, True, False, True],
    )
    comparison_df = build_comparison_summary(
        summary_df.replace({"ModelGroup": {"boosted": "deep"}})
    ).replace({"WinningGroup": {"deep": "boosted"}}).rename(
        columns={
            "DeepBestModel": "BoostedBestModel",
            "DeepBestAUC": "BoostedBestAUC",
            "DeepBestBrier": "BoostedBestBrier",
            "DeepMinusBaselineAUC": "BoostedMinusBaselineAUC",
            "DeepMinusBaselineAccuracy": "BoostedMinusBaselineAccuracy",
            "DeepMinusBaselineBullishAvgFutureRetPct": "BoostedMinusBaselineBullishAvgFutureRetPct",
        }
    )

    baseline_current = baseline_current.copy()
    if not baseline_current.empty:
        baseline_current["ModelGroup"] = "baseline"
    boosted_current = boosted_current.copy()
    if not boosted_current.empty:
        boosted_current["ModelGroup"] = "boosted"
    current_forecast = pd.concat([baseline_current, boosted_current], ignore_index=True)
    prediction_history = pd.concat(
        [
            baseline_history.assign(ModelGroup="baseline") if not baseline_history.empty else baseline_history,
            boosted_history.assign(ModelGroup="boosted") if not boosted_history.empty else boosted_history,
        ],
        ignore_index=True,
    )

    payload = {
        "market_symbol": market_symbol,
        "min_train_dates": int(min_train_dates),
        "retrain_every": int(retrain_every),
        "comparison": comparison_df.to_dict(orient="records"),
        "model_summary": summary_df.to_dict(orient="records"),
        "current_forecast": current_forecast.assign(Date=pd.to_datetime(current_forecast["Date"]).dt.date.astype(str)).to_dict(orient="records")
        if not current_forecast.empty
        else [],
    }
    prefix_output_dir = output_dir
    prefix_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "vnindex" if market_symbol == "VNINDEX" else market_symbol.lower()
    summary_df.to_csv(prefix_output_dir / f"{prefix}_boosted_experiment_model_summary.csv", index=False)
    comparison_df.to_csv(prefix_output_dir / f"{prefix}_boosted_experiment_comparison.csv", index=False)
    history_out = prediction_history.copy()
    if not history_out.empty:
        history_out["Date"] = pd.to_datetime(history_out["Date"]).dt.date.astype(str)
    history_out.to_csv(prefix_output_dir / f"{prefix}_boosted_experiment_prediction_history.csv", index=False)
    current_out = current_forecast.copy()
    if not current_out.empty:
        current_out["Date"] = pd.to_datetime(current_out["Date"]).dt.date.astype(str)
    current_out.to_csv(prefix_output_dir / f"{prefix}_boosted_experiment_current_forecast.csv", index=False)
    (prefix_output_dir / f"{prefix}_boosted_experiment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{market_symbol}BoostedExperiment")
    print(comparison_df.to_string(index=False))
    print()
    print("CurrentForecast")
    print(
        current_forecast[
            [
                "HorizonDays",
                "ModelGroup",
                "Model",
                "ProbabilityUp",
                "CurrentBias",
                "IndexClose",
                "Breadth20Pct",
                "Breadth50Pct",
                "BreadthPositive5Pct",
                "IndexRange60",
            ]
        ].sort_values(["HorizonDays", "ModelGroup", "Model"]).to_string(index=False)
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only boosted experiment against market-direction baselines.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV used for replay.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write experiment reports.")
    parser.add_argument("--min-train-dates", default=80, type=int, help="Minimum labeled dates before walk-forward starts.")
    parser.add_argument("--retrain-every", default=10, type=int, help="Retrain cadence in trading days.")
    parser.add_argument("--market-symbol", default="VNINDEX", help="Market index symbol to model, e.g. VNINDEX or VN30.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        history_dir=Path(args.history_dir),
        sector_map_path=Path(args.sector_map),
        output_dir=Path(args.output_dir),
        min_train_dates=int(args.min_train_dates),
        retrain_every=int(args.retrain_every),
        market_symbol=args.market_symbol,
    )


if __name__ == "__main__":
    main()
