from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.build_intraday_rest_of_session_report import (
    DEFAULT_INTRADAY_HISTORY_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESOLUTION,
    DEPTH_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_multi_ticker_rest_of_session_sample,
)
from scripts.data_fetching.fetch_ticker_data import ensure_intraday_cache
from scripts.analysis.evaluate_intraday_feature_families import (
    add_micro_volume_features,
    build_intraday_feature_families,
)
from scripts.analysis.evaluate_ohlc_models import _normalise_ticker


OUTPUT_HISTORY = "curated_intraday_model_history.csv"
OUTPUT_METRICS = "curated_intraday_model_metrics.csv"
OUTPUT_CURRENT = "curated_intraday_model_current.csv"
OUTPUT_JSON = "curated_intraday_model_summary.json"

DEFAULT_MIN_TRAIN_DATES = 45
DEFAULT_TEST_BLOCK_DATES = 5
DEFAULT_HISTORY_CALENDAR_DAYS = 420
DEFAULT_CURATED_TICKERS = (
    "ACB",
    "BID",
    "BSR",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VPL",
    "VRE",
)
LOCKED_BUCKET_CONFIGS = {
    "AM_EARLY": ("price_volume_core", "ridge"),
    "AM_LATE": ("price_volume_core", "hist_gbm"),
    "LUNCH_BREAK": ("price_volume_core", "hist_gbm"),
    "PM_EARLY": ("price_volume_core", "hist_gbm"),
    "PM_LATE": ("range_micro_volume", "ridge"),
}


def _auto_tickers(history_dir: Path, resolution: str) -> List[str]:
    suffix = f"_{str(resolution).strip()}m.csv"
    tickers: List[str] = []
    for path in sorted(history_dir.glob(f"*{suffix}")):
        ticker = _normalise_ticker(path.name[: -len(suffix)])
        if ticker and ticker != "VNINDEX" and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def refresh_intraday_caches(
    tickers: Sequence[str],
    history_dir: Path,
    *,
    resolution: str,
    history_calendar_days: int,
) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ["VNINDEX", *tickers]:
        ensure_intraday_cache(
            _normalise_ticker(ticker),
            outdir=str(history_dir),
            min_days=int(history_calendar_days),
            resolution=str(resolution),
        )


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any() and float(values.std(ddof=0) or 0.0) > 1e-12:
                out.append(column)
    return out


def build_curated_feature_sets(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    families = build_intraday_feature_families(feature_columns)
    depth = set(DEPTH_FEATURE_COLUMNS)

    def fam(*names: str) -> List[str]:
        columns: List[str] = []
        for name in names:
            columns.extend(families.get(name, []))
        seen: List[str] = []
        for column in columns:
            if column not in depth and column not in seen:
                seen.append(column)
        return seen

    baseline = [column for column in FEATURE_COLUMNS if column in feature_columns and column not in depth]
    return {
        "baseline_live_no_depth": baseline,
        "range_micro_volume": fam(
            "time_bucket",
            "ticker_range_position",
            "ticker_volume",
            "ticker_micro_volume",
            "relative_micro_volume",
            "ticker_price_volume_interaction",
        ),
        "price_volume_core": fam(
            "time_bucket",
            "ticker_price_momentum",
            "ticker_range_position",
            "ticker_vwap",
            "ticker_volume",
            "ticker_micro_volume",
            "ticker_price_volume_interaction",
            "relative_price",
            "relative_volume",
            "relative_micro_volume",
        ),
        "all_price_volume_micro": fam("time_bucket", "all_price_volume"),
        "all_available_no_depth": [column for column in feature_columns if column not in depth],
    }


def build_model_factories() -> Dict[str, Callable[[], Pipeline]]:
    def pipe(model: object, *, scale: bool = True) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps=steps)

    return {
        "ridge": lambda: pipe(Ridge(alpha=10.0)),
        "hist_gbm": lambda: pipe(
            HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.045,
                max_leaf_nodes=15,
                l2_regularization=0.02,
                random_state=42,
            ),
            scale=False,
        ),
        "rf_compact": lambda: pipe(
            RandomForestRegressor(
                n_estimators=96,
                max_depth=8,
                max_features="sqrt",
                min_samples_leaf=5,
                n_jobs=4,
                random_state=42,
            ),
            scale=False,
        ),
    }


def _selection_score(group: pd.DataFrame) -> float:
    close_mae = float(mean_absolute_error(group["TargetCloseRetPct"], group["PredTargetCloseRetPct"]))
    range_mae = float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"]))
    upside_miss = np.maximum(
        pd.to_numeric(group["TargetHighRetPct"], errors="coerce")
        - pd.to_numeric(group["PredTargetHighRetPct"], errors="coerce"),
        0.0,
    )
    downside_miss = np.maximum(
        pd.to_numeric(group["PredTargetLowRetPct"], errors="coerce")
        - pd.to_numeric(group["TargetLowRetPct"], errors="coerce"),
        0.0,
    )
    close_dir = float(
        (
            np.sign(pd.to_numeric(group["TargetCloseRetPct"], errors="coerce"))
            == np.sign(pd.to_numeric(group["PredTargetCloseRetPct"], errors="coerce"))
        ).mean()
        * 100.0
    )
    return float(
        close_mae
        + (0.35 * range_mae)
        + (0.90 * float(np.nanmean(upside_miss)))
        + (0.60 * float(np.nanmean(downside_miss)))
        - (0.01 * close_dir)
    )


def _bucket_config(bucket: object) -> tuple[str, str]:
    return LOCKED_BUCKET_CONFIGS.get(str(bucket), LOCKED_BUCKET_CONFIGS["AM_LATE"])


def walk_forward_models(
    sample_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    min_train_dates: int,
    test_block_dates: int,
    selection_mode: str,
) -> pd.DataFrame:
    labeled = sample_df.dropna(subset=TARGET_COLUMNS).copy()
    labeled["TradeDate"] = pd.to_datetime(labeled["TradeDate"]).dt.normalize()
    dates = pd.Series(labeled["TradeDate"].drop_duplicates()).sort_values().reset_index(drop=True)
    factories = build_model_factories()
    frames: List[pd.DataFrame] = []
    if dates.shape[0] <= int(min_train_dates):
        return pd.DataFrame()

    base_columns = [
        "TradeDate",
        "SnapshotDate",
        "SnapshotTs",
        "SnapshotTimeBucket",
        "Ticker",
        "PrevClose",
        "Base",
        "SnapshotRetFromPrevClosePct",
        "SnapshotRedFromPrevClose",
        *TARGET_COLUMNS,
    ]
    for start in range(int(min_train_dates), int(dates.shape[0]), int(test_block_dates)):
        train_dates = set(dates.iloc[:start].tolist())
        test_dates = set(dates.iloc[start : start + int(test_block_dates)].tolist())
        train_df = labeled[labeled["TradeDate"].isin(train_dates)].copy()
        test_df = labeled[labeled["TradeDate"].isin(test_dates)].copy()
        if train_df.empty or test_df.empty:
            continue

        if selection_mode == "full":
            model_items = [
                (feature_set, model_name, test_df, raw_columns)
                for feature_set, raw_columns in feature_sets.items()
                for model_name in factories
            ]
        else:
            model_items = []
            for bucket, bucket_test_df in test_df.groupby("SnapshotTimeBucket", sort=False):
                feature_set, model_name = _bucket_config(bucket)
                model_items.append((feature_set, model_name, bucket_test_df, feature_sets.get(feature_set, [])))

        for feature_set, model_name, predict_df, raw_columns in model_items:
            usable = _numeric_columns(train_df, raw_columns)
            if not usable or model_name not in factories:
                continue
            factory = factories[model_name]
            out = predict_df[base_columns].copy()
            out["FeatureSet"] = feature_set
            out["Model"] = model_name
            out["UsedFeatureCount"] = len(usable)
            out["TrainEndDate"] = dates.iloc[start - 1].strftime("%Y-%m-%d")
            out["TestBlockStartDate"] = dates.iloc[start].strftime("%Y-%m-%d")
            for target_column in TARGET_COLUMNS:
                model = factory()
                model.fit(train_df[usable], train_df[target_column].astype(float))
                out[f"Pred{target_column}"] = model.predict(predict_df[usable])
            frames.append(out)

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if history.empty:
        return history
    history["ActualRangePct"] = history["TargetHighRetPct"].astype(float) - history["TargetLowRetPct"].astype(float)
    history["PredRangePct"] = (
        history["PredTargetHighRetPct"].astype(float) - history["PredTargetLowRetPct"].astype(float)
    )
    history["CloseAbsErrPct"] = (
        history["TargetCloseRetPct"].astype(float) - history["PredTargetCloseRetPct"].astype(float)
    ).abs()
    history["HighAbsErrPct"] = (
        history["TargetHighRetPct"].astype(float) - history["PredTargetHighRetPct"].astype(float)
    ).abs()
    history["LowAbsErrPct"] = (
        history["TargetLowRetPct"].astype(float) - history["PredTargetLowRetPct"].astype(float)
    ).abs()
    history["CloseDirHit"] = (
        np.sign(history["TargetCloseRetPct"].astype(float))
        == np.sign(history["PredTargetCloseRetPct"].astype(float))
    )
    return history


def summarise_metrics(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (bucket, feature_set, model), group in history_df.groupby(
        ["SnapshotTimeBucket", "FeatureSet", "Model"],
        sort=False,
    ):
        rows.append(
            {
                "Scope": "ALL",
                "Ticker": "ALL",
                "SnapshotTimeBucket": bucket,
                "FeatureSet": feature_set,
                "Model": model,
                "Rows": int(group.shape[0]),
                "UsedFeatureCount": float(group["UsedFeatureCount"].mean()),
                "CloseMAEPct": float(group["CloseAbsErrPct"].mean()),
                "HighMAEPct": float(group["HighAbsErrPct"].mean()),
                "LowMAEPct": float(group["LowAbsErrPct"].mean()),
                "RangeMAEPct": float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"])),
                "CloseDirHitPct": float(group["CloseDirHit"].astype(bool).mean() * 100.0),
                "SelectionScore": _selection_score(group),
            }
        )
    for (ticker, bucket, feature_set, model), group in history_df.groupby(
        ["Ticker", "SnapshotTimeBucket", "FeatureSet", "Model"],
        sort=False,
    ):
        rows.append(
            {
                "Scope": ticker,
                "Ticker": ticker,
                "SnapshotTimeBucket": bucket,
                "FeatureSet": feature_set,
                "Model": model,
                "Rows": int(group.shape[0]),
                "UsedFeatureCount": float(group["UsedFeatureCount"].mean()),
                "CloseMAEPct": float(group["CloseAbsErrPct"].mean()),
                "HighMAEPct": float(group["HighAbsErrPct"].mean()),
                "LowMAEPct": float(group["LowAbsErrPct"].mean()),
                "RangeMAEPct": float(mean_absolute_error(group["ActualRangePct"], group["PredRangePct"])),
                "CloseDirHitPct": float(group["CloseDirHit"].astype(bool).mean() * 100.0),
                "SelectionScore": _selection_score(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Scope", "Ticker", "SnapshotTimeBucket", "SelectionScore", "CloseMAEPct"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def fit_current_predictions(
    sample_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    labeled = sample_df.dropna(subset=TARGET_COLUMNS).copy()
    if labeled.empty or sample_df.empty:
        return pd.DataFrame()
    factories = build_model_factories()
    latest_rows = sample_df.sort_values("SnapshotTs").groupby("Ticker", sort=False).tail(1).copy()
    frames: List[pd.DataFrame] = []
    for bucket, bucket_rows in latest_rows.groupby("SnapshotTimeBucket", sort=False):
        feature_set, model_name = _bucket_config(bucket)
        if model_name not in factories or feature_set not in feature_sets:
            continue
        usable = _numeric_columns(labeled, feature_sets[feature_set])
        if not usable:
            continue
        out = bucket_rows[
            [
                "TradeDate",
                "SnapshotDate",
                "SnapshotTs",
                "SnapshotTimeBucket",
                "Ticker",
                "PrevClose",
                "Base",
                "SnapshotRetFromPrevClosePct",
            ]
        ].copy()
        out["FeatureSet"] = feature_set
        out["Model"] = model_name
        out["UsedFeatureCount"] = len(usable)
        for target_column in TARGET_COLUMNS:
            model = factories[model_name]()
            model.fit(labeled[usable], labeled[target_column].astype(float))
            out[f"Pred{target_column}"] = model.predict(bucket_rows[usable])
        frames.append(out)
    current = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if current.empty:
        return current
    current["PredLow"] = current["Base"].astype(float) * (1.0 + current["PredTargetLowRetPct"].astype(float) / 100.0)
    current["PredClose"] = current["Base"].astype(float) * (
        1.0 + current["PredTargetCloseRetPct"].astype(float) / 100.0
    )
    current["PredHigh"] = current["Base"].astype(float) * (1.0 + current["PredTargetHighRetPct"].astype(float) / 100.0)
    return current.sort_values(["Ticker"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward experiment for curated intraday feature models.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_INTRADAY_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "curated_intraday_model")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--tickers",
        default="auto",
        help="Comma-separated tickers, `auto` from intraday cache, or `curated` for the locked pooled universe.",
    )
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--test-block-dates", type=int, default=DEFAULT_TEST_BLOCK_DATES)
    parser.add_argument("--history-calendar-days", type=int, default=DEFAULT_HISTORY_CALENDAR_DAYS)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=("locked", "full"),
        default="locked",
        help="`locked` runs only the chosen bucket configs; `full` reruns every feature-set/model pair.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ticker_arg = str(args.tickers).strip().lower()
    if ticker_arg == "auto":
        tickers = _auto_tickers(args.history_dir, str(args.resolution))
    elif ticker_arg == "curated":
        tickers = list(DEFAULT_CURATED_TICKERS)
    else:
        tickers = [_normalise_ticker(raw) for raw in str(args.tickers).split(",") if raw.strip()]
    if not tickers:
        raise SystemExit(f"No tickers resolved from {args.history_dir}")
    if bool(args.refresh_cache):
        refresh_intraday_caches(
            tickers,
            args.history_dir,
            resolution=str(args.resolution),
            history_calendar_days=int(args.history_calendar_days),
        )

    sample_df = build_multi_ticker_rest_of_session_sample(
        tickers,
        args.history_dir,
        str(args.resolution),
        require_depth=False,
    )
    sample_df, micro_features = add_micro_volume_features(sample_df, args.history_dir, str(args.resolution), tickers)
    feature_columns = list(FEATURE_COLUMNS) + list(micro_features)
    feature_sets = build_curated_feature_sets(feature_columns)
    history = walk_forward_models(
        sample_df,
        feature_sets,
        min_train_dates=int(args.min_train_dates),
        test_block_dates=int(args.test_block_dates),
        selection_mode=str(args.selection_mode),
    )
    metrics = summarise_metrics(history)
    current = fit_current_predictions(sample_df, feature_sets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / OUTPUT_HISTORY
    metrics_path = args.output_dir / OUTPUT_METRICS
    current_path = args.output_dir / OUTPUT_CURRENT
    json_path = args.output_dir / OUTPUT_JSON
    history.to_csv(history_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    current.to_csv(current_path, index=False)

    vic_metrics = metrics[(metrics["Scope"] == "VIC") | ((metrics["Scope"] == "ALL") & (metrics["Ticker"] == "ALL"))]
    best_rows = vic_metrics.sort_values("SelectionScore").head(12)
    payload = {
        "Tickers": tickers,
        "TickerCount": len(tickers),
        "SampleRows": int(sample_df.shape[0]),
        "SampleDates": int(pd.to_datetime(sample_df["TradeDate"]).dt.normalize().nunique()),
        "HistoryRows": int(history.shape[0]),
        "FeatureSets": {name: len(columns) for name, columns in feature_sets.items()},
        "MicroFeatureCount": int(len(micro_features)),
        "SelectionMode": str(args.selection_mode),
        "LockedBucketConfigs": {
            bucket: {"FeatureSet": feature_set, "Model": model}
            for bucket, (feature_set, model) in LOCKED_BUCKET_CONFIGS.items()
        },
        "Outputs": {
            "History": str(history_path),
            "Metrics": str(metrics_path),
            "Current": str(current_path),
        },
        "BestVicAndAllRows": best_rows.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
