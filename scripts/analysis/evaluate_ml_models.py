from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_deterministic_strategies import build_feature_pack


warnings.filterwarnings(
    "ignore",
    message="The previous implementation of stack is deprecated",
    category=FutureWarning,
)


DEFAULT_CASE_TICKERS = ["HPG", "FPT", "SSI", "VCB", "NKG"]


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _broadcast_series(series: pd.Series, columns: Sequence[str]) -> pd.DataFrame:
    data = {column: series for column in columns}
    return pd.DataFrame(data, index=series.index)


def _stack_feature(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame.rename_axis(index="Date", columns="Ticker").stack(dropna=False).rename(name)


def build_ml_sample(history_dir: Path, sector_map_path: Path) -> pd.DataFrame:
    features = build_feature_pack(history_dir, sector_map_path)
    tickers = list(features.close.columns)

    index_ret5 = features.index_close.pct_change(5) * 100.0
    index_ret20 = features.index_close.pct_change(20) * 100.0
    index_dist20 = ((features.index_close / features.index_sma20) - 1.0) * 100.0
    index_dist50 = ((features.index_close / features.index_sma50) - 1.0) * 100.0

    feature_frames = {
        "Ret5Pct": features.ret5 * 100.0,
        "Ret20Pct": features.ret20 * 100.0,
        "Ret60Pct": features.ret60 * 100.0,
        "DistSMA20Pct": features.dist20 * 100.0,
        "DistSMA50Pct": features.dist50 * 100.0,
        "VolumeRatio20": features.vol_ratio20,
        "Rel20Pct": features.rel20 * 100.0,
        "Rel60Pct": features.rel60 * 100.0,
        "Rel20Rank": features.rel20_rank,
        "Rel60Rank": features.rel60_rank,
        "Corr20": features.corr20,
        "Beta20": features.beta20,
        "SectorBreadth20Pct": features.sector_support,
        "SectorRel20Pct": features.sector_rel_support * 100.0,
        "SectorRankPct": features.sector_rank_per_ticker,
        "MarketBreadth20Pct": _broadcast_series(features.breadth20, tickers),
        "MarketBreadth50Pct": _broadcast_series(features.breadth50, tickers),
        "MarketPositive5Pct": _broadcast_series(features.breadth_positive5, tickers),
        "IndexRange60": _broadcast_series(features.index_range60, tickers),
        "IndexRet5Pct": _broadcast_series(index_ret5, tickers),
        "IndexRet20Pct": _broadcast_series(index_ret20, tickers),
        "IndexDistSMA20Pct": _broadcast_series(index_dist20, tickers),
        "IndexDistSMA50Pct": _broadcast_series(index_dist50, tickers),
        "Fwd10Pct": features.fwd10 * 100.0,
        "Excess10Pct": features.excess10 * 100.0,
    }

    stacked = [_stack_feature(frame, name) for name, frame in feature_frames.items()]
    sample = pd.concat(stacked, axis=1).reset_index()
    sample = sample.rename(columns={"level_0": "Date", "level_1": "Ticker"})
    sample["Ticker"] = sample["Ticker"].map(_normalise_ticker)
    sample["Sector"] = sample["Ticker"].map(lambda ticker: features.sectors.get(ticker, "Unknown"))
    sample["TargetOutperform10d"] = np.where(
        sample["Excess10Pct"].notna(),
        (sample["Excess10Pct"] > 0).astype(int),
        np.nan,
    )
    return sample.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def numeric_feature_columns() -> List[str]:
    return [
        "Ret5Pct",
        "Ret20Pct",
        "Ret60Pct",
        "DistSMA20Pct",
        "DistSMA50Pct",
        "VolumeRatio20",
        "Rel20Pct",
        "Rel60Pct",
        "Rel20Rank",
        "Rel60Rank",
        "Corr20",
        "Beta20",
        "SectorBreadth20Pct",
        "SectorRel20Pct",
        "SectorRankPct",
        "MarketBreadth20Pct",
        "MarketBreadth50Pct",
        "MarketPositive5Pct",
        "IndexRange60",
        "IndexRet5Pct",
        "IndexRet20Pct",
        "IndexDistSMA20Pct",
        "IndexDistSMA50Pct",
    ]


def build_model_factories(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Dict[str, Callable[[], Pipeline]]:
    def make_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(numeric_columns),
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    list(categorical_columns),
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

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
                        min_samples_leaf=5,
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


def _fit_predict_block(
    model_factory: Callable[[], Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    model = model_factory()
    model.fit(train_df[list(feature_columns)], train_df["TargetOutperform10d"].astype(int))
    return model.predict_proba(test_df[list(feature_columns)])[:, 1]


def walk_forward_predict(
    sample_df: pd.DataFrame,
    top_k: int,
    min_train_dates: int,
    retrain_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = numeric_feature_columns() + ["Sector"]
    labeled = sample_df[sample_df["TargetOutperform10d"].notna()].copy()
    unique_dates = list(pd.Index(sorted(labeled["Date"].unique())))
    eval_dates = unique_dates[min_train_dates:]
    latest_date = sample_df["Date"].max()
    current_rows = sample_df[sample_df["Date"] == latest_date].copy()

    factories = build_model_factories(numeric_feature_columns(), ["Sector"])
    history_frames: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []

    for model_name, factory in factories.items():
        block_predictions: List[pd.DataFrame] = []
        for start in range(0, len(eval_dates), retrain_every):
            block_dates = eval_dates[start : start + retrain_every]
            if not block_dates:
                continue
            train_df = labeled[labeled["Date"] < block_dates[0]].copy()
            if train_df["TargetOutperform10d"].nunique(dropna=True) < 2:
                continue
            block_df = labeled[labeled["Date"].isin(block_dates)].copy()
            probs = _fit_predict_block(factory, train_df, block_df, feature_columns)
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
            out["Model"] = model_name
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
        current_out["Model"] = model_name
        current_out["ProbabilityOutperform10d"] = current_probs
        current_out = current_out.sort_values("ProbabilityOutperform10d", ascending=False).reset_index(drop=True)
        current_out["Rank"] = np.arange(1, len(current_out) + 1)
        current_frames.append(current_out)

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    return history_df, current_df


def summarise_ml_models(
    prediction_history: pd.DataFrame,
    current_predictions: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for model_name, group in prediction_history.groupby("Model", sort=False):
        y_true = group["TargetOutperform10d"].astype(int)
        y_prob = group["ProbabilityOutperform10d"].astype(float)
        y_pred = (y_prob >= 0.5).astype(int)
        top_picks = (
            group.sort_values(["Date", "ProbabilityOutperform10d"], ascending=[True, False])
            .groupby("Date", as_index=False, group_keys=False)
            .head(top_k)
        )
        top_daily = top_picks.groupby("Date", as_index=False)[["Fwd10Pct", "Excess10Pct"]].mean()
        current_group = current_predictions[current_predictions["Model"] == model_name].copy()
        rows.append(
            {
                "Model": model_name,
                "EvalSamples": int(group.shape[0]),
                "EvalDays": int(group["Date"].nunique()),
                "AUC": float(roc_auc_score(y_true, y_prob)),
                "Brier": float(brier_score_loss(y_true, y_prob)),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "TopK": top_k,
                "TopKSignalDays": int(top_daily.shape[0]),
                "TopKAvgFwd10Pct": float(top_daily["Fwd10Pct"].mean()),
                "TopKAvgExcess10Pct": float(top_daily["Excess10Pct"].mean()),
                "TopKHit10Pct": float((top_daily["Fwd10Pct"] > 0).mean() * 100.0),
                "CurrentMeanProbability": float(current_group["ProbabilityOutperform10d"].mean()),
                "CurrentTopKMeanProbability": float(current_group.head(top_k)["ProbabilityOutperform10d"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
        ascending=[False, False, False],
    )


def choose_best_ml_model(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        raise ValueError("ML summary is empty")
    ordered = summary_df.sort_values(
        ["TopKAvgExcess10Pct", "AUC", "TopKHit10Pct"],
        ascending=[False, False, False],
    )
    return str(ordered.iloc[0]["Model"])


def build_ml_case_studies(
    prediction_history: pd.DataFrame,
    current_predictions: pd.DataFrame,
    case_tickers: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    top_history = (
        prediction_history.sort_values(["Model", "Date", "ProbabilityOutperform10d"], ascending=[True, True, False])
        .groupby(["Model", "Date"], as_index=False, group_keys=False)
        .head(top_k)
    )
    for model_name, current_group in current_predictions.groupby("Model", sort=False):
        for ticker in case_tickers:
            current_row = current_group[current_group["Ticker"] == ticker]
            if current_row.empty:
                continue
            current_row = current_row.iloc[0]
            history_rows = top_history[(top_history["Model"] == model_name) & (top_history["Ticker"] == ticker)]
            rows.append(
                {
                    "Date": current_row["Date"].date().isoformat(),
                    "Ticker": ticker,
                    "Model": model_name,
                    "CurrentProbabilityOutperform10d": float(current_row["ProbabilityOutperform10d"]),
                    "CurrentRank": int(current_row["Rank"]),
                    "CurrentTopK": int(current_row["Rank"] <= top_k),
                    "CurrentRel20Pct": float(current_row["Rel20Pct"]),
                    "CurrentRel60Pct": float(current_row["Rel60Pct"]),
                    "CurrentDistSMA20Pct": float(current_row["DistSMA20Pct"]),
                    "CurrentSectorBreadth20Pct": float(current_row["SectorBreadth20Pct"]),
                    "HistoricalTopKCount": int(history_rows.shape[0]),
                    "HistoricalAvgFwd10Pct": float(history_rows["Fwd10Pct"].mean()) if not history_rows.empty else float("nan"),
                    "HistoricalAvgExcess10Pct": float(history_rows["Excess10Pct"].mean()) if not history_rows.empty else float("nan"),
                    "HistoricalHit10Pct": float((history_rows["Fwd10Pct"] > 0).mean() * 100.0)
                    if not history_rows.empty
                    else float("nan"),
                    "LastHistoricalTopKDate": history_rows["Date"].max().date().isoformat()
                    if not history_rows.empty
                    else "",
                }
            )
    return pd.DataFrame(rows)


def _print_stdout_summary(
    summary_df: pd.DataFrame,
    best_model: str,
    current_predictions: pd.DataFrame,
    case_studies: pd.DataFrame,
    top_k: int,
) -> None:
    latest_date = ""
    if not current_predictions.empty:
        latest_date = current_predictions["Date"].iloc[0].date().isoformat()

    print(f"BestMLModel: {best_model}")
    print(f"CurrentDate: {latest_date}")
    print()
    print("MLModels")
    cols = [
        "Model",
        "AUC",
        "Brier",
        "Accuracy",
        "TopKAvgFwd10Pct",
        "TopKAvgExcess10Pct",
        "TopKHit10Pct",
        "CurrentMeanProbability",
        "CurrentTopKMeanProbability",
    ]
    print(summary_df[cols].to_string(index=False))
    print()
    print("CurrentTopPicks")
    top_picks = (
        current_predictions.sort_values(["Model", "ProbabilityOutperform10d"], ascending=[True, False])
        .groupby("Model", as_index=False, group_keys=False)
        .head(top_k)
    )
    if top_picks.empty:
        print("No current ML predictions.")
    else:
        print(
            top_picks[
                [
                    "Model",
                    "Rank",
                    "Ticker",
                    "Sector",
                    "ProbabilityOutperform10d",
                    "Rel20Pct",
                    "Rel60Pct",
                    "DistSMA20Pct",
                ]
            ].to_string(index=False)
        )
    print()
    print("CaseStudies")
    print(
        case_studies[
            [
                "Ticker",
                "Model",
                "CurrentProbabilityOutperform10d",
                "CurrentRank",
                "CurrentTopK",
                "CurrentRel20Pct",
                "CurrentRel60Pct",
                "CurrentDistSMA20Pct",
                "CurrentSectorBreadth20Pct",
                "HistoricalTopKCount",
                "HistoricalAvgExcess10Pct",
            ]
        ].to_string(index=False)
    )


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    output_dir: Path,
    top_k: int,
    min_train_dates: int,
    retrain_every: int,
    case_tickers: Sequence[str],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = build_ml_sample(history_dir, sector_map_path)
    prediction_history, current_predictions = walk_forward_predict(
        sample_df=sample_df,
        top_k=top_k,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
    )
    summary_df = summarise_ml_models(prediction_history, current_predictions, top_k)
    best_model = choose_best_ml_model(summary_df)
    summary_df["SelectedBestModel"] = summary_df["Model"] == best_model

    current_predictions = current_predictions.sort_values(["Model", "ProbabilityOutperform10d"], ascending=[True, False]).reset_index(drop=True)
    current_top_picks = (
        current_predictions.groupby("Model", as_index=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    case_studies = build_ml_case_studies(
        prediction_history=prediction_history,
        current_predictions=current_predictions,
        case_tickers=case_tickers,
        top_k=top_k,
    )

    prediction_history_out = prediction_history.copy()
    prediction_history_out["Date"] = prediction_history_out["Date"].dt.date.astype(str)
    current_predictions_out = current_predictions.copy()
    current_predictions_out["Date"] = current_predictions_out["Date"].dt.date.astype(str)
    current_top_picks_out = current_top_picks.copy()
    current_top_picks_out["Date"] = current_top_picks_out["Date"].dt.date.astype(str)

    summary_df.to_csv(output_dir / "ml_model_summary.csv", index=False)
    prediction_history_out.to_csv(output_dir / "ml_prediction_history.csv", index=False)
    current_predictions_out.to_csv(output_dir / "ml_current_predictions.csv", index=False)
    current_top_picks_out.to_csv(output_dir / "ml_current_top_picks.csv", index=False)
    case_studies.to_csv(output_dir / "ml_case_studies.csv", index=False)

    summary_payload = {
        "best_model": best_model,
        "top_k": top_k,
        "min_train_dates": min_train_dates,
        "retrain_every": retrain_every,
        "models": summary_df.to_dict(orient="records"),
        "case_tickers": list(case_tickers),
    }
    (output_dir / "ml_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_stdout_summary(summary_df, best_model, current_predictions, case_studies, top_k)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay simple machine-learning models on cached daily data.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV used for replay.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        history_dir=Path(args.history_dir),
        sector_map_path=Path(args.sector_map),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        min_train_dates=int(args.min_train_dates),
        retrain_every=int(args.retrain_every),
        case_tickers=[_normalise_ticker(ticker) for ticker in args.case_tickers],
    )


if __name__ == "__main__":
    main()
