from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_vnindex_models import (  # noqa: E402
    HORIZONS,
    _normalise_symbol,
    build_vnindex_sample,
    numeric_feature_columns,
)
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
)


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def make_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )


def build_baseline_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_balanced": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gbm": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=250,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_deep_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    
    return {
        "mlp_small": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(24, 12),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=350,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "mlp_deep": lambda: Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(48, 24, 12),
                        activation="relu",
                        alpha=3e-3,
                        learning_rate_init=0.0007,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=500,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def walk_forward_market_predictions(
    sample_df: pd.DataFrame,
    min_train_dates: int,
    retrain_every: int,
    model_factories: Dict[str, Callable[[], Pipeline]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = numeric_feature_columns()
    history_rows: List[pd.DataFrame] = []
    current_rows: List[pd.DataFrame] = []
    latest_date = sample_df["Date"].max()

    for horizon in HORIZONS:
        target_col = f"TargetUp{horizon}d"
        ret_col = f"IndexFwd{horizon}Pct"
        labeled = sample_df[sample_df[target_col].notna()].copy()
        unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
        eval_dates = unique_dates[min_train_dates:]

        for model_name, factory in model_factories.items():
            block_predictions: List[pd.DataFrame] = []
            for start in range(0, len(eval_dates), retrain_every):
                block_dates = eval_dates[start : start + retrain_every]
                if not block_dates:
                    continue
                train_df = labeled[labeled["Date"] < block_dates[0]].copy()
                if train_df[target_col].nunique(dropna=True) < 2:
                    continue
                block_df = labeled[labeled["Date"].isin(block_dates)].copy()
                model = factory()
                model.fit(train_df[feature_columns], train_df[target_col].astype(int))
                probs = model.predict_proba(block_df[feature_columns])[:, 1]
                out = block_df[
                    [
                        "Date",
                        "IndexClose",
                        ret_col,
                        target_col,
                        "Breadth20Pct",
                        "Breadth50Pct",
                        "BreadthPositive5Pct",
                        "IndexRange60",
                    ]
                ].copy()
                out = out.rename(columns={ret_col: "FutureRetPct", target_col: "TargetUp"})
                out["HorizonDays"] = horizon
                out["Model"] = model_name
                out["ProbabilityUp"] = probs
                block_predictions.append(out)

            if block_predictions:
                history_rows.append(pd.concat(block_predictions, ignore_index=True))

            train_all = labeled[labeled["Date"] < latest_date].copy()
            if train_all[target_col].nunique(dropna=True) < 2:
                continue
            current_df = sample_df[sample_df["Date"] == latest_date].copy()
            model = factory()
            model.fit(train_all[feature_columns], train_all[target_col].astype(int))
            prob = float(model.predict_proba(current_df[feature_columns])[:, 1][0])
            current_rows.append(
                pd.DataFrame(
                    [
                        {
                            "Date": latest_date,
                            "HorizonDays": horizon,
                            "Model": model_name,
                            "ProbabilityUp": prob,
                            "CurrentBias": "UP" if prob >= 0.55 else "DOWN" if prob <= 0.45 else "NEUTRAL",
                            "IndexClose": float(current_df.iloc[0]["IndexClose"]),
                            "Breadth20Pct": float(current_df.iloc[0]["Breadth20Pct"]),
                            "Breadth50Pct": float(current_df.iloc[0]["Breadth50Pct"]),
                            "BreadthPositive5Pct": float(current_df.iloc[0]["BreadthPositive5Pct"]),
                            "IndexRange60": float(current_df.iloc[0]["IndexRange60"]),
                        }
                    ]
                )
            )

    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    current_df = pd.concat(current_rows, ignore_index=True) if current_rows else pd.DataFrame()
    return history_df, current_df


def summarise_model_group(prediction_history: pd.DataFrame, *, group_label: str) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (horizon, model_name), group in prediction_history.groupby(["HorizonDays", "Model"], sort=False):
        y_true = group["TargetUp"].astype(int)
        y_prob = group["ProbabilityUp"].astype(float)
        y_pred = (y_prob >= 0.5).astype(int)
        bullish = group["ProbabilityUp"] >= 0.55
        bearish = group["ProbabilityUp"] <= 0.45
        rows.append(
            {
                "ModelGroup": group_label,
                "HorizonDays": int(horizon),
                "Model": model_name,
                "EvalDays": int(group.shape[0]),
                "AUC": _safe_auc(y_true, y_prob),
                "Brier": float(brier_score_loss(y_true, y_prob)),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "BullishDays": int(bullish.sum()),
                "BullishAvgFutureRetPct": float(group.loc[bullish, "FutureRetPct"].mean()) if bullish.any() else float("nan"),
                "BullishHitPct": float((group.loc[bullish, "FutureRetPct"] > 0).mean() * 100.0) if bullish.any() else float("nan"),
                "BearishDays": int(bearish.sum()),
                "BearishAvgFutureRetPct": float(group.loc[bearish, "FutureRetPct"].mean()) if bearish.any() else float("nan"),
                "BearishHitPct": float((group.loc[bearish, "FutureRetPct"] > 0).mean() * 100.0) if bearish.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["HorizonDays", "AUC", "BullishAvgFutureRetPct"],
        ascending=[True, False, False],
    )


def build_comparison_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for horizon, horizon_df in summary_df.groupby("HorizonDays", sort=True):
        ordered = horizon_df.sort_values(
            ["AUC", "Brier", "Accuracy", "BullishAvgFutureRetPct"],
            ascending=[False, True, False, False],
        ).reset_index(drop=True)
        top_row = ordered.iloc[0]
        baseline_best = horizon_df[horizon_df["ModelGroup"] == "baseline"].sort_values(
            ["AUC", "Brier", "Accuracy", "BullishAvgFutureRetPct"],
            ascending=[False, True, False, False],
        )
        deep_best = horizon_df[horizon_df["ModelGroup"] == "deep"].sort_values(
            ["AUC", "Brier", "Accuracy", "BullishAvgFutureRetPct"],
            ascending=[False, True, False, False],
        )
        baseline_row = baseline_best.iloc[0] if not baseline_best.empty else None
        deep_row = deep_best.iloc[0] if not deep_best.empty else None
        rows.append(
            {
                "HorizonDays": int(horizon),
                "WinningGroup": str(top_row["ModelGroup"]),
                "WinningModel": str(top_row["Model"]),
                "WinningAUC": float(top_row["AUC"]),
                "WinningBrier": float(top_row["Brier"]),
                "WinningAccuracy": float(top_row["Accuracy"]),
                "WinningBullishAvgFutureRetPct": float(top_row["BullishAvgFutureRetPct"]),
                "BaselineBestModel": str(baseline_row["Model"]) if baseline_row is not None else "",
                "BaselineBestAUC": float(baseline_row["AUC"]) if baseline_row is not None else float("nan"),
                "BaselineBestBrier": float(baseline_row["Brier"]) if baseline_row is not None else float("nan"),
                "DeepBestModel": str(deep_row["Model"]) if deep_row is not None else "",
                "DeepBestAUC": float(deep_row["AUC"]) if deep_row is not None else float("nan"),
                "DeepBestBrier": float(deep_row["Brier"]) if deep_row is not None else float("nan"),
                "DeepMinusBaselineAUC": float(deep_row["AUC"] - baseline_row["AUC"]) if deep_row is not None and baseline_row is not None else float("nan"),
                "DeepMinusBaselineAccuracy": float(deep_row["Accuracy"] - baseline_row["Accuracy"]) if deep_row is not None and baseline_row is not None else float("nan"),
                "DeepMinusBaselineBullishAvgFutureRetPct": float(deep_row["BullishAvgFutureRetPct"] - baseline_row["BullishAvgFutureRetPct"]) if deep_row is not None and baseline_row is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("HorizonDays")


def _prepare_history(symbol: str, history_dir: Path) -> None:
    market_symbol = _normalise_symbol(symbol) or "VNINDEX"
    ensure_ohlc_cache(market_symbol, outdir=str(history_dir), min_days=800, resolution="D")
    ensure_ohlc_cache("VN30" if market_symbol == "VNINDEX" else "VNINDEX", outdir=str(history_dir), min_days=800, resolution="D")


def _write_outputs(
    *,
    output_dir: Path,
    market_symbol: str,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    prediction_history: pd.DataFrame,
    current_forecast: pd.DataFrame,
    payload: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "vnindex" if market_symbol == "VNINDEX" else market_symbol.lower()
    summary_df.to_csv(output_dir / f"{prefix}_deep_experiment_model_summary.csv", index=False)
    comparison_df.to_csv(output_dir / f"{prefix}_deep_experiment_comparison.csv", index=False)
    history_out = prediction_history.copy()
    if not history_out.empty:
        history_out["Date"] = pd.to_datetime(history_out["Date"]).dt.date.astype(str)
    history_out.to_csv(output_dir / f"{prefix}_deep_experiment_prediction_history.csv", index=False)
    current_out = current_forecast.copy()
    if not current_out.empty:
        current_out["Date"] = pd.to_datetime(current_out["Date"]).dt.date.astype(str)
    current_out.to_csv(output_dir / f"{prefix}_deep_experiment_current_forecast.csv", index=False)
    (output_dir / f"{prefix}_deep_experiment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_stdout_summary(comparison_df: pd.DataFrame, current_forecast: pd.DataFrame, market_symbol: str) -> None:
    print(f"{_normalise_symbol(market_symbol)}DeepExperiment")
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
    deep_history, deep_current = walk_forward_market_predictions(
        sample_df=sample_df,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
        model_factories=build_deep_model_factories(),
    )

    baseline_summary = summarise_model_group(baseline_history, group_label="baseline")
    deep_summary = summarise_model_group(deep_history, group_label="deep")
    summary_df = pd.concat([baseline_summary, deep_summary], ignore_index=True).sort_values(
        ["HorizonDays", "ModelGroup", "AUC", "Brier"],
        ascending=[True, True, False, True],
    )
    comparison_df = build_comparison_summary(summary_df)

    baseline_current = baseline_current.copy()
    if not baseline_current.empty:
        baseline_current["ModelGroup"] = "baseline"
    deep_current = deep_current.copy()
    if not deep_current.empty:
        deep_current["ModelGroup"] = "deep"
    current_forecast = pd.concat([baseline_current, deep_current], ignore_index=True)
    prediction_history = pd.concat(
        [
            baseline_history.assign(ModelGroup="baseline") if not baseline_history.empty else baseline_history,
            deep_history.assign(ModelGroup="deep") if not deep_history.empty else deep_history,
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
    _write_outputs(
        output_dir=output_dir,
        market_symbol=market_symbol,
        summary_df=summary_df,
        comparison_df=comparison_df,
        prediction_history=prediction_history,
        current_forecast=current_forecast,
        payload=payload,
    )
    _print_stdout_summary(comparison_df, current_forecast, market_symbol)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only deep experiment against market-direction baselines.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV used for replay.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write experiment reports.")
    parser.add_argument("--min-train-dates", default=80, type=int, help="Minimum labeled dates before walk-forward starts.")
    parser.add_argument("--retrain-every", default=5, type=int, help="Retrain cadence in trading days.")
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
