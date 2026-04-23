from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_deterministic_strategies import build_feature_pack
from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache


HORIZONS = (5, 10)


def _range_position(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    return (series - rolling_min) / (rolling_max - rolling_min)


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _normalise_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _load_daily_close(symbol: str, history_dir: Path) -> pd.Series:
    path = history_dir / f"{_normalise_symbol(symbol)}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Empty history file: {path}")
    date_col = "date_vn" if "date_vn" in frame.columns else "t"
    if date_col == "date_vn":
        index = pd.to_datetime(frame[date_col], errors="coerce")
    else:
        index = pd.to_datetime(pd.to_numeric(frame[date_col], errors="coerce"), unit="s", errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(close.to_numpy(), index=index, name=_normalise_symbol(symbol))
    return series[~series.index.isna()].sort_index()


def _direction_streak_state(series: pd.Series, *, min_streak: int = 2) -> pd.Series:
    values = series.astype(float)
    positive = values.gt(0.0)
    negative = values.lt(0.0)
    positive_streak = positive.groupby((~positive).cumsum()).cumsum()
    negative_streak = negative.groupby((~negative).cumsum()).cumsum()
    state = pd.Series(0.0, index=values.index, dtype=float)
    state.loc[positive_streak >= int(min_streak)] = 1.0
    state.loc[negative_streak >= int(min_streak)] = -1.0
    return state


def build_vnindex_sample(history_dir: Path, sector_map_path: Path, market_symbol: str = "VNINDEX") -> pd.DataFrame:
    features = build_feature_pack(history_dir, sector_map_path)
    market_symbol = _normalise_symbol(market_symbol) or "VNINDEX"
    peer_symbol = "VN30" if market_symbol == "VNINDEX" else "VNINDEX"
    market_close = _load_daily_close(market_symbol, history_dir)
    peer_close = _load_daily_close(peer_symbol, history_dir)
    market_close = market_close.reindex(features.index_close.index).ffill()
    peer_close = peer_close.reindex(features.index_close.index).ffill()
    market_ret1 = market_close.pct_change() * 100.0
    peer_ret1 = peer_close.pct_change() * 100.0

    sample = pd.DataFrame(
        {
            "Date": market_close.index,
            "IndexClose": market_close,
            "IndexRet5Pct": market_close.pct_change(5) * 100.0,
            "IndexRet20Pct": market_close.pct_change(20) * 100.0,
            "IndexRet60Pct": market_close.pct_change(60) * 100.0,
            "IndexDistSMA20Pct": ((market_close / market_close.rolling(20).mean()) - 1.0) * 100.0,
            "IndexDistSMA50Pct": ((market_close / market_close.rolling(50).mean()) - 1.0) * 100.0,
            "IndexRange20": _range_position(market_close, 20),
            "IndexRange60": _range_position(market_close, 60),
            "IndexColorStreakState": _direction_streak_state(market_ret1, min_streak=2),
            "Breadth20Pct": features.breadth20,
            "Breadth50Pct": features.breadth50,
            "BreadthPositive5Pct": features.breadth_positive5,
            "MarketDispersion20Pct": features.ret20.std(axis=1) * 100.0,
            "MarketMedianRel20Pct": features.rel20.median(axis=1) * 100.0,
            "MarketMeanCorr20": features.corr20.mean(axis=1),
            "MarketMeanBeta20": features.beta20.mean(axis=1),
            "SectorBreadthLeaderCount": (features.sector_breadth20 >= 50.0).sum(axis=1),
            "SectorBreadthWeakCount": (features.sector_breadth20 <= 30.0).sum(axis=1),
            "PeerRet5Pct": peer_close.pct_change(5) * 100.0,
            "PeerRet20Pct": peer_close.pct_change(20) * 100.0,
            "PeerDistSMA20Pct": ((peer_close / peer_close.rolling(20).mean()) - 1.0) * 100.0,
            "PeerRange20": _range_position(peer_close, 20),
            "PeerColorStreakState": _direction_streak_state(peer_ret1, min_streak=2),
            "IndexVsPeerRet5Pct": (market_close.pct_change(5) - peer_close.pct_change(5)) * 100.0,
        }
    )
    for horizon in HORIZONS:
        sample[f"IndexFwd{horizon}Pct"] = ((market_close.shift(-horizon) / market_close) - 1.0) * 100.0
        sample[f"TargetUp{horizon}d"] = np.where(
            sample[f"IndexFwd{horizon}Pct"].notna(),
            (sample[f"IndexFwd{horizon}Pct"] > 0).astype(int),
            np.nan,
        )
    return sample.sort_values("Date").reset_index(drop=True)


def numeric_feature_columns() -> List[str]:
    return [
        "IndexRet5Pct",
        "IndexRet20Pct",
        "IndexRet60Pct",
        "IndexDistSMA20Pct",
        "IndexDistSMA50Pct",
        "IndexRange20",
        "IndexRange60",
        "IndexColorStreakState",
        "Breadth20Pct",
        "Breadth50Pct",
        "BreadthPositive5Pct",
        "MarketDispersion20Pct",
        "MarketMedianRel20Pct",
        "MarketMeanCorr20",
        "MarketMeanBeta20",
        "SectorBreadthLeaderCount",
        "SectorBreadthWeakCount",
        "PeerRet5Pct",
        "PeerRet20Pct",
        "PeerDistSMA20Pct",
        "PeerRange20",
        "PeerColorStreakState",
        "IndexVsPeerRet5Pct",
    ]


def build_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    def make_preprocessor() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
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


def walk_forward_market_predictions(
    sample_df: pd.DataFrame,
    min_train_dates: int,
    retrain_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = numeric_feature_columns()
    model_factories = build_model_factories()
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


def summarise_vnindex_models(prediction_history: pd.DataFrame) -> pd.DataFrame:
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


def choose_best_vnindex_model(summary_df: pd.DataFrame, horizon_days: int) -> str:
    subset = summary_df[summary_df["HorizonDays"] == horizon_days].copy()
    if subset.empty:
        raise ValueError(f"No VNINDEX model summary for horizon {horizon_days}")
    ordered = subset.sort_values(["AUC", "BullishAvgFutureRetPct", "Accuracy"], ascending=[False, False, False])
    return str(ordered.iloc[0]["Model"])


def _print_stdout_summary(summary_df: pd.DataFrame, current_forecast: pd.DataFrame, market_symbol: str) -> None:
    print(f"{_normalise_symbol(market_symbol)}Models")
    print(
        summary_df[
            [
                "HorizonDays",
                "Model",
                "AUC",
                "Brier",
                "Accuracy",
                "BullishDays",
                "BullishAvgFutureRetPct",
                "BearishDays",
                "BearishAvgFutureRetPct",
            ]
        ].to_string(index=False)
    )
    print()
    print("CurrentForecast")
    print(
        current_forecast[
            [
                "HorizonDays",
                "Model",
                "ProbabilityUp",
                "CurrentBias",
                "IndexClose",
                "Breadth20Pct",
                "Breadth50Pct",
                "BreadthPositive5Pct",
                "IndexRange60",
            ]
        ].to_string(index=False)
    )


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    output_dir: Path,
    min_train_dates: int,
    retrain_every: int,
    market_symbol: str = "VNINDEX",
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    market_symbol = _normalise_symbol(market_symbol) or "VNINDEX"
    ensure_ohlc_cache(market_symbol, outdir=str(history_dir), min_days=800, resolution="D")
    ensure_ohlc_cache("VN30" if market_symbol == "VNINDEX" else "VNINDEX", outdir=str(history_dir), min_days=800, resolution="D")
    sample_df = build_vnindex_sample(history_dir, sector_map_path, market_symbol=market_symbol)
    prediction_history, current_forecast = walk_forward_market_predictions(
        sample_df=sample_df,
        min_train_dates=min_train_dates,
        retrain_every=retrain_every,
    )
    summary_df = summarise_vnindex_models(prediction_history)
    best_models = {str(horizon): choose_best_vnindex_model(summary_df, horizon) for horizon in HORIZONS}
    summary_df["SelectedBestModel"] = summary_df.apply(
        lambda row: best_models.get(str(int(row["HorizonDays"]))) == row["Model"],
        axis=1,
    )

    prediction_history_out = prediction_history.copy()
    prediction_history_out["Date"] = prediction_history_out["Date"].dt.date.astype(str)
    current_forecast_out = current_forecast.copy()
    current_forecast_out["Date"] = current_forecast_out["Date"].dt.date.astype(str)
    file_prefix = "vnindex" if market_symbol == "VNINDEX" else market_symbol.lower()
    summary_df.to_csv(output_dir / f"{file_prefix}_ml_model_summary.csv", index=False)
    prediction_history_out.to_csv(output_dir / f"{file_prefix}_ml_prediction_history.csv", index=False)
    current_forecast_out.to_csv(output_dir / f"{file_prefix}_ml_current_forecast.csv", index=False)

    payload = {
        "market_symbol": market_symbol,
        "best_models": best_models,
        "min_train_dates": min_train_dates,
        "retrain_every": retrain_every,
        "model_summary": summary_df.to_dict(orient="records"),
        "current_forecast": current_forecast_out.to_dict(orient="records"),
    }
    (output_dir / f"{file_prefix}_ml_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_stdout_summary(summary_df, current_forecast, market_symbol)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay machine-learning models on VNINDEX/breadth features.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV used for replay.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write evaluation reports.")
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
