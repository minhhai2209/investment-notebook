from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _normalise_ticker,
    build_ticker_ohlc_sample,
)
from scripts.analysis.evaluate_vic_index_expiry_features import add_index_expiry_features


DEFAULT_TICKER = "VIC"
DEFAULT_HORIZONS = (1, 2, 3, 4, 5)
DEFAULT_HOLDOUT_SESSIONS = 5
DEFAULT_WALKBACK_DATES = 80
DEFAULT_WALKBACK_STEP = 5
OUTPUT_CANDIDATES = "vic_single_model_candidates.csv"
OUTPUT_HOLDOUT = "vic_single_model_holdout.csv"
OUTPUT_WALKBACK = "vic_single_model_walkback.csv"
OUTPUT_CURRENT = "vic_single_model_current.csv"
OUTPUT_SUMMARY = "vic_single_model_summary.json"
OUTPUT_FEATURE_ENGINEERING = "vic_single_model_feature_engineering.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    objective: str
    factory: Callable[[], Pipeline]


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    usable: List[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() and float(values.std(ddof=0) or 0.0) > 1e-12:
            usable.append(column)
    return usable


def _signed_sum(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    out = frame.copy()
    engineered: Dict[str, pd.Series] = {}

    ret_cols = [column for column in ["TickerRet1Pct", "TickerRet5Pct", "TickerRet20Pct"] if column in out.columns]
    if ret_cols:
        engineered["TickerRetStackMeanPct"] = out[ret_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        engineered["TickerRetStackAbsMaxPct"] = out[ret_cols].apply(pd.to_numeric, errors="coerce").abs().max(axis=1)

    if {"TickerRet5Pct", "IndexRet5Pct"}.issubset(out.columns):
        engineered["TickerVsIndexRet5Pct"] = out["TickerRet5Pct"].astype(float) - out["IndexRet5Pct"].astype(float)
    if {"TickerRet20Pct", "IndexRet20Pct"}.issubset(out.columns):
        engineered["TickerVsIndexRet20Pct"] = out["TickerRet20Pct"].astype(float) - out["IndexRet20Pct"].astype(float)
    if {"TickerRet5Pct", "VN30Ret5Pct"}.issubset(out.columns):
        engineered["TickerVsVN30Ret5Pct"] = out["TickerRet5Pct"].astype(float) - out["VN30Ret5Pct"].astype(float)

    breakout_cols = [
        "TickerBreakoutHigh20State",
        "TickerBreakoutHigh60State",
        "TickerBreakoutHigh120State",
        "TickerBreakoutHigh252State",
    ]
    engineered["TickerBreakoutStateCount"] = _signed_sum(out, breakout_cols)

    risk_cols = [
        "TickerLimitProxyState",
        "TickerShockState1D",
        "TickerImpulseState3D",
        "TickerWideRangeState",
        "TickerTrendRegimeState",
        "TickerRelativeRotationState",
        "TickerExhaustionState",
    ]
    engineered["TickerRiskStateComposite"] = _signed_sum(out, risk_cols)

    if {"TickerDistSMA20Pct", "TickerVolatility10"}.issubset(out.columns):
        vol = out["TickerVolatility10"].astype(float).replace(0.0, np.nan)
        engineered["TickerDistSMA20PerVol10"] = out["TickerDistSMA20Pct"].astype(float) / vol
    if {"TickerDistSMA50Pct", "TickerVolatility10"}.issubset(out.columns):
        vol = out["TickerVolatility10"].astype(float).replace(0.0, np.nan)
        engineered["TickerDistSMA50PerVol10"] = out["TickerDistSMA50Pct"].astype(float) / vol
    if {"TickerRet20Pct", "TickerVolRatio20"}.issubset(out.columns):
        engineered["TickerRet20VolumePressure"] = out["TickerRet20Pct"].astype(float) * out["TickerVolRatio20"].astype(float)
    if {"TickerRangePct", "TickerVolRatio20"}.issubset(out.columns):
        engineered["TickerRangeVolumePressure"] = out["TickerRangePct"].astype(float) * out["TickerVolRatio20"].astype(float)
    if {"TickerUpperWickPct", "TickerLowerWickPct"}.issubset(out.columns):
        engineered["TickerWickImbalancePct"] = out["TickerUpperWickPct"].astype(float) - out["TickerLowerWickPct"].astype(float)
    if {"TickerRangePos20", "TickerRangePos60"}.issubset(out.columns):
        engineered["TickerRangePosSpread20v60"] = out["TickerRangePos20"].astype(float) - out["TickerRangePos60"].astype(float)

    for horizon in (20, 60, 120, 252):
        gap = f"TickerGapToPriorHigh{horizon}Pct"
        dist = f"TickerDistPriorHigh{horizon}Pct"
        if gap in out.columns and dist in out.columns:
            engineered[f"TickerPriorHighPressure{horizon}"] = out[dist].astype(float) - out[gap].astype(float)

    added: List[str] = []
    for name, values in engineered.items():
        out[name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        added.append(name)
    return out, added


def build_model_specs() -> List[ModelSpec]:
    def scaled(model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    def tree(model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", model),
            ]
        )

    return [
        ModelSpec("ridge_close_ret", "price", lambda: scaled(Ridge(alpha=4.0))),
        ModelSpec(
            "hist_gbm_close_ret",
            "price",
            lambda: tree(
                HistGradientBoostingRegressor(
                    max_depth=3,
                    learning_rate=0.04,
                    max_iter=180,
                    l2_regularization=0.03,
                    random_state=42,
                )
            ),
        ),
        ModelSpec(
            "rf_compact_close_ret",
            "price",
            lambda: tree(
                RandomForestRegressor(
                    n_estimators=96,
                    max_depth=8,
                    max_features="sqrt",
                    min_samples_leaf=5,
                    n_jobs=4,
                    random_state=42,
                )
            ),
        ),
        ModelSpec(
            "logit_direction",
            "direction",
            lambda: scaled(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ),
        ModelSpec(
            "hist_gbm_direction",
            "direction",
            lambda: tree(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=140, random_state=42)),
        ),
        ModelSpec(
            "rf_compact_direction",
            "direction",
            lambda: tree(
                RandomForestClassifier(
                    n_estimators=128,
                    max_depth=7,
                    max_features="sqrt",
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=4,
                    random_state=42,
                )
            ),
        ),
    ]


def build_feature_sets(sample: pd.DataFrame, added_columns: Sequence[str], engineered_columns: Sequence[str]) -> Dict[str, List[str]]:
    base = [column for column in FEATURE_COLUMNS if column in sample.columns]
    added = [column for column in added_columns if column in sample.columns]
    engineered = [column for column in engineered_columns if column in sample.columns]
    return {
        "baseline_ohlc": base,
        "index_expiry_exvin": base + added,
        "engineered_all": base + added + engineered,
    }


def _predict_candidate(spec: ModelSpec, train_df: pd.DataFrame, test_df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    if spec.objective == "direction":
        target = train_df["TargetCloseRetPct"].astype(float).gt(0.0).astype(int)
        if target.nunique() < 2:
            return np.full(test_df.shape[0], np.nan)
        model = spec.factory()
        model.fit(train_df[list(features)], target)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(test_df[list(features)])[:, 1]
        return model.predict(test_df[list(features)]).astype(float)

    model = spec.factory()
    model.fit(train_df[list(features)], train_df["TargetCloseRetPct"].astype(float))
    return model.predict(test_df[list(features)])


def _prediction_rows(
    sample: pd.DataFrame,
    *,
    feature_set: str,
    feature_columns: Sequence[str],
    spec: ModelSpec,
    eval_dates_by_horizon: Mapping[int, Sequence[pd.Timestamp]],
    current_base_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for horizon, eval_dates in eval_dates_by_horizon.items():
        scoped = sample[sample["Horizon"].astype(int).eq(int(horizon))].copy()
        labeled = scoped[scoped["TargetCloseRetPct"].notna()].copy()
        usable = _numeric_columns(labeled, feature_columns)
        if not usable:
            continue
        for eval_date in eval_dates:
            eval_ts = pd.Timestamp(eval_date)
            train_df = labeled[labeled["Date"] < eval_ts].copy()
            test_df = scoped[scoped["Date"].eq(eval_ts)].copy()
            if train_df.empty or test_df.empty:
                continue
            predicted = _predict_candidate(spec, train_df, test_df, usable)
            if np.isnan(predicted).all():
                continue
            out = test_df[
                [
                    "Date",
                    "Ticker",
                    "Horizon",
                    "ForecastWindow",
                    "BaseClose",
                    "ForecastDate",
                    "ActualClose",
                    "TargetCloseRetPct",
                ]
            ].copy()
            out["EvalKind"] = "current" if eval_ts == current_base_date else "walkback"
            out["FeatureSet"] = feature_set
            out["Model"] = spec.name
            out["Objective"] = spec.objective
            out["UsedFeatureCount"] = len(usable)
            out["TrainRows"] = int(train_df.shape[0])
            out["TrainEndDate"] = str(pd.Timestamp(train_df["Date"].max()).date())
            if spec.objective == "direction":
                out["PredUpProb"] = predicted
                out["PredDirectionUp"] = predicted >= 0.5
                out["PredCloseRetPct"] = np.nan
                out["PredClose"] = np.nan
            else:
                out["PredCloseRetPct"] = predicted
                out["PredClose"] = out["BaseClose"].astype(float) * (1.0 + (out["PredCloseRetPct"].astype(float) / 100.0))
                out["PredUpProb"] = np.nan
                out["PredDirectionUp"] = out["PredCloseRetPct"].astype(float) >= 0.0
            out["ActualDirectionUp"] = out["TargetCloseRetPct"].astype(float) > 0.0
            out["DirectionHit"] = out["PredDirectionUp"].astype(bool) == out["ActualDirectionUp"].astype(bool)
            out["CloseAbsErrPct"] = (
                out["TargetCloseRetPct"].astype(float) - out["PredCloseRetPct"].astype(float)
            ).abs()
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _score_group(group: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "Rows": int(group.shape[0]),
        "DirectionHitPct": float(group["DirectionHit"].astype(bool).mean() * 100.0) if not group.empty else float("nan"),
    }
    price_rows = group[group["PredCloseRetPct"].notna()]
    out["CloseMAEPct"] = (
        float(mean_absolute_error(price_rows["TargetCloseRetPct"].astype(float), price_rows["PredCloseRetPct"].astype(float)))
        if not price_rows.empty
        else float("nan")
    )
    out["PredDownPct"] = float((~group["PredDirectionUp"].astype(bool)).mean() * 100.0) if not group.empty else float("nan")
    return out


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def summarise_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    key_cols = ["FeatureSet", "Model", "Objective"]
    for key, group in predictions.groupby(key_cols, sort=False):
        feature_set, model, objective = key
        walkback = group[group["EvalKind"].eq("walkback")]
        holdout = group[group["EvalKind"].eq("current")]
        walk = _score_group(walkback)
        hold = _score_group(holdout)
        if objective == "price":
            selection_score = (
                float(hold["CloseMAEPct"])
                + (0.35 * float(walk["CloseMAEPct"]))
                - (0.015 * float(hold["DirectionHitPct"]))
                - (0.005 * float(walk["DirectionHitPct"]))
            )
        else:
            selection_score = (
                (100.0 - float(hold["DirectionHitPct"]))
                + (0.35 * (100.0 - float(walk["DirectionHitPct"])))
            )
        rows.append(
            {
                "FeatureSet": feature_set,
                "Model": model,
                "Objective": objective,
                "SelectionScore": float(selection_score),
                "HoldoutRows": hold["Rows"],
                "HoldoutCloseMAEPct": hold["CloseMAEPct"],
                "HoldoutDirectionHitPct": hold["DirectionHitPct"],
                "HoldoutPredDownPct": hold["PredDownPct"],
                "WalkbackRows": walk["Rows"],
                "WalkbackCloseMAEPct": walk["CloseMAEPct"],
                "WalkbackDirectionHitPct": walk["DirectionHitPct"],
                "WalkbackPredDownPct": walk["PredDownPct"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["SelectionScore", "HoldoutDirectionHitPct", "HoldoutCloseMAEPct"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def select_winner(candidates: pd.DataFrame, *, expected_holdout_rows: int) -> Dict[str, object]:
    if candidates.empty:
        raise RuntimeError("No candidate models were evaluated.")
    direction = candidates[candidates["Objective"].eq("direction")].copy()
    price = candidates[candidates["Objective"].eq("price")].copy()

    qualified_direction = direction[
        (direction["HoldoutDirectionHitPct"] >= 60.0)
        & (direction["WalkbackDirectionHitPct"] >= 50.0)
        & (direction["HoldoutRows"] >= int(expected_holdout_rows))
    ].copy()
    if not qualified_direction.empty:
        row = qualified_direction.sort_values(
            ["HoldoutDirectionHitPct", "WalkbackDirectionHitPct", "SelectionScore"],
            ascending=[False, False, True],
        ).iloc[0]
        return row.to_dict()

    qualified_price = price[
        (price["HoldoutCloseMAEPct"] <= 4.50)
        & (price["WalkbackCloseMAEPct"] <= 8.00)
        & (price["HoldoutRows"] >= int(expected_holdout_rows))
    ].copy()
    if not qualified_price.empty:
        row = qualified_price.sort_values(
            ["HoldoutCloseMAEPct", "WalkbackCloseMAEPct", "SelectionScore"],
            ascending=[True, True, True],
        ).iloc[0]
        return row.to_dict()

    return candidates.iloc[0].to_dict()


def build_feature_engineering_audit(
    feature_sets: Mapping[str, Sequence[str]],
    *,
    added_columns: Sequence[str],
    engineered_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    added = set(added_columns)
    engineered = set(engineered_columns)
    for feature_set, columns in feature_sets.items():
        for column in columns:
            if column in engineered:
                source = "engineered"
            elif column in added:
                source = "index_expiry_exvin"
            else:
                source = "baseline_ohlc"
            rows.append(
                {
                    "FeatureSet": feature_set,
                    "Feature": column,
                    "Source": source,
                }
            )
    return pd.DataFrame(rows)


def current_forecast(
    sample: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    winner: Mapping[str, object],
) -> pd.DataFrame:
    feature_set = str(winner["FeatureSet"])
    model_name = str(winner["Model"])
    specs = {spec.name: spec for spec in build_model_specs()}
    spec = specs[model_name]
    rows: List[pd.DataFrame] = []
    latest_date = pd.Timestamp(sample["Date"].max())
    for horizon in sorted(sample["Horizon"].dropna().astype(int).unique()):
        scoped = sample[sample["Horizon"].astype(int).eq(int(horizon))].copy()
        labeled = scoped[scoped["TargetCloseRetPct"].notna()].copy()
        current_row = scoped[scoped["Date"].eq(latest_date)].copy()
        if current_row.empty:
            continue
        train_df = labeled[labeled["Date"] < latest_date].copy()
        usable = _numeric_columns(train_df, feature_sets[feature_set])
        if not usable:
            continue
        predicted = _predict_candidate(spec, train_df, current_row, usable)
        out = current_row[["Date", "Ticker", "Horizon", "ForecastWindow", "BaseClose", "ForecastDate"]].copy()
        out["FeatureSet"] = feature_set
        out["Model"] = model_name
        out["Objective"] = spec.objective
        out["UsedFeatureCount"] = len(usable)
        out["TrainRows"] = int(train_df.shape[0])
        out["TrainEndDate"] = str(pd.Timestamp(train_df["Date"].max()).date())
        if spec.objective == "direction":
            out["PredUpProb"] = predicted
            out["PredDirectionUp"] = predicted >= 0.5
            out["PredCloseRetPct"] = np.nan
            out["PredClose"] = np.nan
        else:
            out["PredCloseRetPct"] = predicted
            out["PredClose"] = out["BaseClose"].astype(float) * (1.0 + (out["PredCloseRetPct"].astype(float) / 100.0))
            out["PredUpProb"] = np.nan
            out["PredDirectionUp"] = out["PredCloseRetPct"].astype(float) >= 0.0
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_selection(
    *,
    ticker: str,
    history_dir: Path,
    output_dir: Path,
    horizons: Sequence[int],
    holdout_sessions: int,
    walkback_dates: int,
    walkback_step: int,
) -> Dict[str, object]:
    ticker = _normalise_ticker(ticker)
    daily = pd.read_csv(history_dir / f"{ticker}_daily.csv")
    date_col = "date_vn" if "date_vn" in daily.columns else "t"
    dates = pd.to_datetime(daily[date_col], unit=None if date_col == "date_vn" else "s", errors="coerce")
    dates = pd.Series(dates.dropna().sort_values().unique())
    if dates.shape[0] <= int(holdout_sessions):
        raise RuntimeError(f"Need more than {holdout_sessions} daily rows for {ticker}.")
    holdout_base_date = pd.Timestamp(dates.iloc[-int(holdout_sessions) - 1])

    sample = build_ticker_ohlc_sample(ticker, history_dir, max_horizon=max(int(h) for h in horizons))
    sample = sample[sample["Horizon"].isin([int(h) for h in horizons])].copy()
    sample, added_columns = add_index_expiry_features(sample, history_dir, ticker=ticker)
    sample, engineered_columns = add_engineered_features(sample)
    feature_sets = build_feature_sets(sample, added_columns, engineered_columns)

    eval_dates_by_horizon: Dict[int, List[pd.Timestamp]] = {}
    for horizon in horizons:
        scoped = sample[sample["Horizon"].astype(int).eq(int(horizon))]
        labeled_dates = pd.Series(pd.to_datetime(scoped.loc[scoped["TargetCloseRetPct"].notna(), "Date"]).sort_values().unique())
        previous_dates = labeled_dates[labeled_dates < holdout_base_date].tail(int(walkback_dates))
        stepped = list(previous_dates.iloc[:: max(1, int(walkback_step))])
        eval_dates_by_horizon[int(horizon)] = [pd.Timestamp(date) for date in stepped] + [holdout_base_date]

    frames: List[pd.DataFrame] = []
    for feature_set, columns in feature_sets.items():
        for spec in build_model_specs():
            frames.append(
                _prediction_rows(
                    sample,
                    feature_set=feature_set,
                    feature_columns=columns,
                    spec=spec,
                    eval_dates_by_horizon=eval_dates_by_horizon,
                    current_base_date=holdout_base_date,
                )
            )
    predictions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    candidates = summarise_candidates(predictions)
    winner = select_winner(candidates, expected_holdout_rows=len(horizons))
    winner_mask = (
        candidates["FeatureSet"].astype(str).eq(str(winner["FeatureSet"]))
        & candidates["Model"].astype(str).eq(str(winner["Model"]))
        & candidates["Objective"].astype(str).eq(str(winner["Objective"]))
    )
    candidates["IsWinner"] = winner_mask
    candidates = candidates.sort_values(
        ["IsWinner", "SelectionScore", "HoldoutDirectionHitPct", "HoldoutCloseMAEPct"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    current = current_forecast(sample, feature_sets, winner)
    feature_audit = build_feature_engineering_audit(
        feature_sets,
        added_columns=added_columns,
        engineered_columns=engineered_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions[predictions["EvalKind"].eq("current")].to_csv(output_dir / OUTPUT_HOLDOUT, index=False)
    predictions[predictions["EvalKind"].eq("walkback")].to_csv(output_dir / OUTPUT_WALKBACK, index=False)
    candidates.to_csv(output_dir / OUTPUT_CANDIDATES, index=False)
    current.to_csv(output_dir / OUTPUT_CURRENT, index=False)
    feature_audit.to_csv(output_dir / OUTPUT_FEATURE_ENGINEERING, index=False)

    payload = {
        "Ticker": ticker,
        "HoldoutBaseDate": str(holdout_base_date.date()),
        "HoldoutActualDates": [str(pd.Timestamp(date).date()) for date in dates.tail(int(holdout_sessions))],
        "Horizons": [int(horizon) for horizon in horizons],
        "FeatureSets": {name: len(columns) for name, columns in feature_sets.items()},
        "EngineeredFeatures": engineered_columns,
        "CandidateRows": int(candidates.shape[0]),
        "Winner": winner,
        "Outputs": {
            "Candidates": str(output_dir / OUTPUT_CANDIDATES),
            "Holdout": str(output_dir / OUTPUT_HOLDOUT),
            "Walkback": str(output_dir / OUTPUT_WALKBACK),
            "Current": str(output_dir / OUTPUT_CURRENT),
            "FeatureEngineering": str(output_dir / OUTPUT_FEATURE_ENGINEERING),
        },
    }
    payload = _json_safe(payload)
    (output_dir / OUTPUT_SUMMARY).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one VIC model for either price or direction using last-5 holdout.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", default=",".join(str(horizon) for horizon in DEFAULT_HORIZONS))
    parser.add_argument("--holdout-sessions", type=int, default=DEFAULT_HOLDOUT_SESSIONS)
    parser.add_argument("--walkback-dates", type=int, default=DEFAULT_WALKBACK_DATES)
    parser.add_argument("--walkback-step", type=int, default=DEFAULT_WALKBACK_STEP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizons = [int(raw.strip()) for raw in str(args.horizons).split(",") if raw.strip()]
    payload = run_selection(
        ticker=str(args.ticker),
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        horizons=horizons,
        holdout_sessions=int(args.holdout_sessions),
        walkback_dates=int(args.walkback_dates),
        walkback_step=int(args.walkback_step),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
