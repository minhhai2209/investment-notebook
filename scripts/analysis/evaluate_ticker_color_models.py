from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import FEATURE_COLUMNS, _normalise_ticker, build_ticker_ohlc_sample


DEFAULT_HISTORY_DIR = Path("out/data")
DEFAULT_OUTPUT_DIR = Path("out/analysis")
DEFAULT_TICKERS = ("VIC", "HPG", "NVL", "MBB")
DEFAULT_HORIZONS = (1, 3, 5, 10)
TARGET_SPECS = {
    "GREEN": {
        "label_column": "TargetGreen",
        "positive_threshold": 0.55,
        "negative_threshold": 0.45,
        "realized_column": "FutureCloseRetPct",
        "positive_direction": 1.0,
    },
    "RED": {
        "label_column": "TargetRed",
        "positive_threshold": 0.55,
        "negative_threshold": 0.45,
        "realized_column": "FutureCloseRetPct",
        "positive_direction": -1.0,
    },
    "CEILING": {
        "label_column": "TargetCeilingProxy",
        "positive_threshold": 0.45,
        "negative_threshold": 0.20,
        "realized_column": "FutureHighRetPct",
        "positive_direction": 1.0,
    },
    "FLOOR": {
        "label_column": "TargetFloorProxy",
        "positive_threshold": 0.45,
        "negative_threshold": 0.20,
        "realized_column": "FutureLowRetPct",
        "positive_direction": -1.0,
    },
}


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _parse_comma_list(raw: str | None, fallback: Sequence[str]) -> List[str]:
    if raw is None:
        return list(fallback)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def make_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )


def build_model_factories() -> Dict[str, Callable[[], Pipeline]]:
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


def build_color_sample(ticker: str, history_dir: Path, horizons: Sequence[int]) -> pd.DataFrame:
    sample = build_ticker_ohlc_sample(ticker, history_dir, max_horizon=max(horizons)).copy()
    sample = sample[sample["Horizon"].isin([int(horizon) for horizon in horizons])].copy()
    sample["FutureCloseRetPct"] = sample["TargetCloseRetPct"].astype(float)
    sample["FutureHighRetPct"] = ((sample["ActualHigh"].astype(float) / sample["BaseClose"].astype(float)) - 1.0) * 100.0
    sample["FutureLowRetPct"] = ((sample["ActualLow"].astype(float) / sample["BaseClose"].astype(float)) - 1.0) * 100.0
    sample["TargetGreen"] = np.where(sample["FutureCloseRetPct"].notna(), (sample["FutureCloseRetPct"] > 0.0).astype(int), np.nan)
    sample["TargetRed"] = np.where(sample["FutureCloseRetPct"].notna(), (sample["FutureCloseRetPct"] < 0.0).astype(int), np.nan)
    sample["TargetCeilingProxy"] = np.where(sample["FutureHighRetPct"].notna(), (sample["FutureHighRetPct"] >= 6.75).astype(int), np.nan)
    sample["TargetFloorProxy"] = np.where(sample["FutureLowRetPct"].notna(), (sample["FutureLowRetPct"] <= -6.75).astype(int), np.nan)
    return sample


def walk_forward_ticker_predictions(
    sample_df: pd.DataFrame,
    *,
    target_names: Sequence[str],
    model_names: Sequence[str],
    min_train_dates: int,
    retrain_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_factories = build_model_factories()
    model_factories = {name: available_factories[name] for name in model_names if name in available_factories}
    history_rows: List[pd.DataFrame] = []
    current_rows: List[pd.DataFrame] = []
    for ticker in sorted(sample_df["Ticker"].dropna().unique()):
        ticker_df = sample_df[sample_df["Ticker"] == ticker].copy()
        for horizon in sorted(ticker_df["Horizon"].dropna().unique()):
            horizon_df = ticker_df[ticker_df["Horizon"] == horizon].copy()
            latest_feature_date = horizon_df["Date"].max()

            for target_name in target_names:
                spec = TARGET_SPECS[str(target_name)]
                target_col = str(spec["label_column"])
                labeled = horizon_df[horizon_df[target_col].notna()].copy()
                if labeled.empty:
                    continue
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
                        if block_df.empty:
                            continue
                        model = factory()
                        model.fit(train_df[list(FEATURE_COLUMNS)], train_df[target_col].astype(int))
                        probs = model.predict_proba(block_df[list(FEATURE_COLUMNS)])[:, 1]
                        out = block_df[
                            [
                                "Date",
                                "Ticker",
                                "Horizon",
                                "ForecastWindow",
                                "BaseClose",
                                target_col,
                                "FutureCloseRetPct",
                                "FutureHighRetPct",
                                "FutureLowRetPct",
                            ]
                        ].copy()
                        out = out.rename(columns={target_col: "TargetPositive"})
                        out["TargetName"] = target_name
                        out["Model"] = model_name
                        out["ProbabilityPositive"] = probs
                        block_predictions.append(out)

                    if block_predictions:
                        history_rows.append(pd.concat(block_predictions, ignore_index=True))

                    train_all = labeled[labeled["Date"] < latest_feature_date].copy()
                    if train_all[target_col].nunique(dropna=True) < 2:
                        continue
                    current_df = horizon_df[horizon_df["Date"] == latest_feature_date].copy()
                    if current_df.empty:
                        continue
                    model = factory()
                    model.fit(train_all[list(FEATURE_COLUMNS)], train_all[target_col].astype(int))
                    prob = float(model.predict_proba(current_df[list(FEATURE_COLUMNS)])[:, 1][0])
                    current_rows.append(
                        pd.DataFrame(
                            [
                                {
                                    "Date": latest_feature_date,
                                    "Ticker": str(current_df.iloc[0]["Ticker"]),
                                    "Horizon": int(horizon),
                                    "ForecastWindow": str(current_df.iloc[0]["ForecastWindow"]),
                                    "TargetName": target_name,
                                    "Model": model_name,
                                    "ProbabilityPositive": prob,
                                    "CurrentBias": "YES" if prob >= float(spec["positive_threshold"]) else "NO" if prob <= float(spec["negative_threshold"]) else "NEUTRAL",
                                    "BaseClose": float(current_df.iloc[0]["BaseClose"]),
                                }
                            ]
                        )
                    )

    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    current_df = pd.concat(current_rows, ignore_index=True) if current_rows else pd.DataFrame()
    return history_df, current_df


def summarise_ticker_color_models(prediction_history: pd.DataFrame) -> pd.DataFrame:
    if prediction_history.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (ticker, horizon, target_name, model_name), group in prediction_history.groupby(
        ["Ticker", "Horizon", "TargetName", "Model"],
        sort=False,
    ):
        spec = TARGET_SPECS[str(target_name)]
        y_true = group["TargetPositive"].astype(int)
        y_prob = group["ProbabilityPositive"].astype(float)
        y_pred = (y_prob >= 0.5).astype(int)
        strong_signal = y_prob >= float(spec["positive_threshold"])
        realized_col = str(spec["realized_column"])
        realized_direction = float(spec["positive_direction"])
        signed_realized = group[realized_col].astype(float) * realized_direction
        rows.append(
            {
                "Ticker": str(ticker),
                "Horizon": int(horizon),
                "TargetName": str(target_name),
                "Model": str(model_name),
                "EvalDays": int(group.shape[0]),
                "PositiveBaseRatePct": float(y_true.mean() * 100.0),
                "AUC": _safe_auc(y_true, y_prob),
                "Brier": float(brier_score_loss(y_true, y_prob)),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "StrongSignalDays": int(strong_signal.sum()),
                "StrongSignalHitPct": float(group.loc[strong_signal, "TargetPositive"].mean() * 100.0) if strong_signal.any() else float("nan"),
                "StrongSignalAvgSignedEdgePct": float(signed_realized.loc[strong_signal].mean()) if strong_signal.any() else float("nan"),
                "StrongSignalAvgCloseRetPct": float(group.loc[strong_signal, "FutureCloseRetPct"].mean()) if strong_signal.any() else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["Ticker", "TargetName", "Horizon", "AUC", "Brier", "StrongSignalAvgSignedEdgePct"],
        ascending=[True, True, True, False, True, False],
    ).reset_index(drop=True)


def build_comparison_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (ticker, target_name, horizon), group in summary_df.groupby(["Ticker", "TargetName", "Horizon"], sort=False):
        ordered = group.sort_values(
            ["AUC", "Brier", "Accuracy", "StrongSignalAvgSignedEdgePct"],
            ascending=[False, True, False, False],
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append(
            {
                "Ticker": str(ticker),
                "TargetName": str(target_name),
                "Horizon": int(horizon),
                "BestModel": str(best["Model"]),
                "BestAUC": float(best["AUC"]),
                "BestBrier": float(best["Brier"]),
                "BestAccuracy": float(best["Accuracy"]),
                "BestStrongSignalDays": int(best["StrongSignalDays"]),
                "BestStrongSignalHitPct": float(best["StrongSignalHitPct"]),
                "BestStrongSignalAvgSignedEdgePct": float(best["StrongSignalAvgSignedEdgePct"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Ticker", "TargetName", "Horizon"]).reset_index(drop=True)


def _print_stdout_summary(comparison_df: pd.DataFrame, current_forecast: pd.DataFrame) -> None:
    print("TickerColorModels")
    print(comparison_df.to_string(index=False))
    print()
    print("CurrentForecast")
    print(
        current_forecast[
            [
                "Ticker",
                "Horizon",
                "TargetName",
                "Model",
                "ProbabilityPositive",
                "CurrentBias",
                "BaseClose",
            ]
        ]
        .sort_values(["Ticker", "TargetName", "Horizon", "Model"])
        .to_string(index=False)
    )


def run_analysis(
    *,
    tickers: Sequence[str],
    history_dir: Path,
    output_dir: Path,
    horizons: Sequence[int],
    target_names: Sequence[str],
    model_names: Sequence[str],
    min_train_dates: int,
    retrain_every: int,
) -> Dict[str, object]:
    color_frames = [build_color_sample(ticker, history_dir, horizons) for ticker in tickers]
    sample_df = pd.concat(color_frames, ignore_index=True) if color_frames else pd.DataFrame()
    prediction_history, current_forecast = walk_forward_ticker_predictions(
        sample_df,
        target_names=target_names,
        model_names=model_names,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
    )
    summary_df = summarise_ticker_color_models(prediction_history)
    comparison_df = build_comparison_summary(summary_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_out = prediction_history.copy()
    if not history_out.empty:
        history_out["Date"] = pd.to_datetime(history_out["Date"]).dt.date.astype(str)
    current_out = current_forecast.copy()
    if not current_out.empty:
        current_out["Date"] = pd.to_datetime(current_out["Date"]).dt.date.astype(str)
    summary_df.to_csv(output_dir / "ticker_color_model_summary.csv", index=False)
    comparison_df.to_csv(output_dir / "ticker_color_comparison.csv", index=False)
    history_out.to_csv(output_dir / "ticker_color_prediction_history.csv", index=False)
    current_out.to_csv(output_dir / "ticker_color_current_forecast.csv", index=False)

    payload = {
        "tickers": list(tickers),
        "horizons": [int(horizon) for horizon in horizons],
        "targets": list(target_names),
        "models": list(model_names),
        "min_train_dates": int(min_train_dates),
        "retrain_every": int(retrain_every),
        "model_summary": summary_df.to_dict(orient="records"),
        "comparison": comparison_df.to_dict(orient="records"),
        "current_forecast": current_out.to_dict(orient="records"),
    }
    (output_dir / "ticker_color_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_stdout_summary(comparison_df, current_forecast)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ticker green/red/ceiling/floor classifiers.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers to evaluate.")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR), help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "ticker_color_models"), help="Directory to write reports.")
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS), help="Comma-separated horizons, e.g. 1,3,5,10.")
    parser.add_argument("--targets", default="GREEN,RED", help="Comma-separated targets from GREEN,RED,CEILING,FLOOR.")
    parser.add_argument("--models", default="logistic_balanced,hist_gbm", help="Comma-separated models from logistic_balanced,hist_gbm.")
    parser.add_argument("--min-train-dates", default=120, type=int, help="Minimum labeled dates before walk-forward starts.")
    parser.add_argument("--retrain-every", default=10, type=int, help="Retrain cadence in trading days.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [_normalise_ticker(item) for item in _parse_comma_list(args.tickers, DEFAULT_TICKERS)]
    horizons = [int(item) for item in _parse_comma_list(args.horizons, [str(item) for item in DEFAULT_HORIZONS])]
    target_names = [item for item in _parse_comma_list(args.targets, ["GREEN", "RED"]) if item in TARGET_SPECS]
    model_names = [item for item in _parse_comma_list(args.models, ["LOGISTIC_BALANCED", "HIST_GBM"])]
    run_analysis(
        tickers=tickers,
        history_dir=Path(args.history_dir),
        output_dir=Path(args.output_dir),
        horizons=horizons,
        target_names=target_names,
        model_names=[item.lower() for item in model_names],
        min_train_dates=int(args.min_train_dates),
        retrain_every=int(args.retrain_every),
    )


if __name__ == "__main__":
    main()
