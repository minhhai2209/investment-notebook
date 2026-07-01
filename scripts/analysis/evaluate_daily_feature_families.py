from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.analysis.evaluate_intraday_feature_families import (
    _learn_feature_signs,
    _series_corr,
    _top_bottom_rows,
    summarise_feature_stats,
)
from scripts.analysis.evaluate_ohlc_models import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    build_ticker_ohlc_sample,
    _normalise_ticker,
)
from scripts.analysis.evaluate_vic_index_expiry_features import add_index_expiry_features


OUTPUT_SUMMARY = "daily_feature_family_walkback_summary.csv"
OUTPUT_FEATURES = "daily_feature_family_feature_stats.csv"
OUTPUT_SIGNALS = "daily_feature_family_walkback_signals.csv"
OUTPUT_JSON = "daily_feature_family_walkback_summary.json"
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 20)
DEFAULT_TARGET = "TargetCloseRetPct"
DEFAULT_MIN_TRAIN_DATES = 120
DEFAULT_TEST_BLOCK_DATES = 10


def _columns_containing(columns: Iterable[str], tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        if any(token in column for token in tokens):
            out.append(column)
    return out


def build_daily_feature_families(columns: Sequence[str]) -> Dict[str, List[str]]:
    cols = list(columns)
    families = {
        "ticker_candle_shape": _columns_containing(
            cols,
            ["TickerGapPct", "TickerBodyPct", "TickerRangePct", "TickerUpperWickPct", "TickerLowerWickPct"],
        ),
        "ticker_returns_momentum": _columns_containing(cols, ["TickerRet1Pct", "TickerRet5Pct", "TickerRet20Pct"]),
        "ticker_volume_volatility": _columns_containing(
            cols,
            ["TickerVolRatio20", "TickerVolatility10", "TickerWeekToDateVolumePctPrevWeek", "TickerPrevWeekVolRatio4"],
        ),
        "ticker_trend_range": _columns_containing(
            cols,
            ["TickerDistSMA", "TickerRangePos", "TickerWeekToDateRangePct", "TickerDistPrevWeek"],
        ),
        "ticker_breakout": _columns_containing(cols, ["TickerDistPriorHigh", "TickerGapToPriorHigh", "TickerBreakout"]),
        "ticker_state_flags": _columns_containing(
            cols,
            [
                "TickerColorStreakState",
                "TickerLimitProxyState",
                "TickerShockState1D",
                "TickerImpulseState3D",
                "TickerWideRangeState",
                "TickerTrendRegimeState",
                "TickerCompressionState",
                "TickerReclaimState",
                "TickerRelativeRotationState",
                "TickerExhaustionState",
            ],
        ),
        "ticker_weekly_context": _columns_containing(cols, ["TickerWeekToDate", "TickerPrevWeek"]),
        "relative_strength": _columns_containing(cols, ["Rel", "Corr20", "Beta20"]),
        "index_context": _columns_containing(cols, ["Index"]),
        "vn30_context": _columns_containing(cols, ["VN30"]),
        "derivative_expiry": _columns_containing(cols, ["Deriv"]),
        "ex_vin_market": _columns_containing(cols, ["ExVin"]),
        "vin_basket": _columns_containing(cols, ["VinBasket"]),
        "ticker_vs_exvin": _columns_containing(cols, ["TickerRet1MinusExVin", "TickerRet5MinusExVin"]),
        "vnindex_minus_exvin": _columns_containing(cols, ["VNIndexRet"]),
    }
    price_families = [
        "ticker_candle_shape",
        "ticker_returns_momentum",
        "ticker_trend_range",
        "ticker_breakout",
        "ticker_state_flags",
        "relative_strength",
    ]
    market_families = ["index_context", "vn30_context", "derivative_expiry", "ex_vin_market", "vin_basket"]
    families["all_ticker_price"] = sorted(set(sum((families[name] for name in price_families), [])))
    families["all_market_context"] = sorted(set(sum((families[name] for name in market_families), [])))
    families["all_daily_available"] = list(cols)
    return {
        name: [column for column in family_columns if column in cols]
        for name, family_columns in families.items()
        if any(column in cols for column in family_columns)
    }


def _auto_tickers(history_dir: Path) -> List[str]:
    tickers: List[str] = []
    for path in sorted(history_dir.glob("*_daily.csv")):
        ticker = _normalise_ticker(path.name.removesuffix("_daily.csv"))
        if not ticker or ticker in {"VNINDEX", "VN30"} or ticker in tickers:
            continue
        tickers.append(ticker)
    return tickers


def build_daily_family_sample(
    tickers: Sequence[str],
    history_dir: Path,
    horizons: Sequence[int],
    *,
    include_index_expiry_exvin: bool,
) -> tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    max_horizon = max(int(horizon) for horizon in horizons)
    for ticker in tickers:
        sample = build_ticker_ohlc_sample(ticker, history_dir, max_horizon=max_horizon)
        sample = sample[sample["Horizon"].isin([int(horizon) for horizon in horizons])].copy()
        if include_index_expiry_exvin:
            sample, _ = add_index_expiry_features(sample, history_dir, ticker=ticker)
        frames.append(sample)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    feature_columns = [column for column in out.columns if column in set(FEATURE_COLUMNS) or _is_added_family_column(column)]
    return out, feature_columns


def _is_added_family_column(column: str) -> bool:
    return any(
        token in column
        for token in (
            "Deriv",
            "ExVin",
            "VinBasket",
            "TickerRet1MinusExVin",
            "TickerRet5MinusExVin",
            "VNIndexRet",
        )
    )


def _score_family_block(
    test_df: pd.DataFrame,
    *,
    family: str,
    signs: Mapping[str, float],
    medians: Mapping[str, float],
    scales: Mapping[str, float],
    target_column: str,
) -> pd.DataFrame:
    usable = [feature for feature in signs if feature in test_df.columns]
    if not usable:
        return pd.DataFrame()
    z_parts = []
    for feature in usable:
        values = pd.to_numeric(test_df[feature], errors="coerce")
        z = ((values - medians[feature]) / scales[feature]).clip(-5.0, 5.0)
        z_parts.append(z * signs[feature])
    signal = pd.concat(z_parts, axis=1).mean(axis=1)
    out = test_df[["Date", "Ticker", "Horizon", "ForecastWindow", target_column]].copy()
    out = out.rename(columns={target_column: "Target"})
    out["Family"] = family
    out["Signal"] = signal.astype(float)
    out["UsedFeatureCount"] = len(usable)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Target", "Signal"])


def walk_back_daily_feature_families(
    sample_df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    *,
    target_column: str,
    min_train_dates: int,
    test_block_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = sample_df[sample_df[target_column].notna()].copy()
    dates = pd.Series(labeled["Date"].drop_duplicates()).sort_values().reset_index(drop=True)
    signal_frames: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []
    if dates.shape[0] <= int(min_train_dates):
        return pd.DataFrame(), pd.DataFrame()
    for horizon, horizon_df in labeled.groupby("Horizon", sort=True):
        horizon_dates = pd.Series(horizon_df["Date"].drop_duplicates()).sort_values().reset_index(drop=True)
        if horizon_dates.shape[0] <= int(min_train_dates):
            continue
        for start in range(int(min_train_dates), int(horizon_dates.shape[0]), int(test_block_dates)):
            train_dates = set(horizon_dates.iloc[:start].tolist())
            test_dates = set(horizon_dates.iloc[start : start + int(test_block_dates)].tolist())
            train_df = horizon_df[horizon_df["Date"].isin(train_dates)].copy()
            test_df = horizon_df[horizon_df["Date"].isin(test_dates)].copy()
            if train_df.empty or test_df.empty:
                continue
            for family, feature_columns in families.items():
                signs, medians, scales = _learn_feature_signs(train_df, target_column, feature_columns)
                for feature in feature_columns:
                    values = pd.to_numeric(train_df.get(feature), errors="coerce")
                    coverage = float(values.notna().mean() * 100.0) if values is not None else 0.0
                    corr = _series_corr(values, pd.to_numeric(train_df[target_column], errors="coerce"))
                    feature_rows.append(
                        {
                            "BlockStartDate": horizon_dates.iloc[start].strftime("%Y-%m-%d"),
                            "Horizon": int(horizon),
                            "Family": family,
                            "Feature": feature,
                            "TrainCorr": corr,
                            "TrainAbsCorr": abs(corr) if np.isfinite(corr) else np.nan,
                            "TrainSign": signs.get(feature, np.nan),
                            "TrainCoveragePct": coverage,
                            "Selected": feature in signs,
                        }
                    )
                scored = _score_family_block(
                    test_df,
                    family=family,
                    signs=signs,
                    medians=medians,
                    scales=scales,
                    target_column=target_column,
                )
                if not scored.empty:
                    signal_frames.append(scored)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    feature_stats = pd.DataFrame(feature_rows)
    return signals, feature_stats


def _safe_spearman(group: pd.DataFrame) -> float:
    if group.shape[0] < 5 or group["Signal"].nunique(dropna=True) < 2 or group["Target"].nunique(dropna=True) < 2:
        return float("nan")
    return float(group["Signal"].rank().corr(group["Target"].rank()))


def _top_bottom(group: pd.DataFrame, top_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = group.sort_values("Signal", ascending=False)
    count = max(1, int(math.ceil(float(group.shape[0]) * float(top_fraction))))
    return ordered.head(count), ordered.tail(count)


def summarise_daily_signals(signals: pd.DataFrame, *, top_fraction: float) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (family, horizon), group in signals.groupby(["Family", "Horizon"], sort=False):
        grouped = [
            g
            for _, g in group.groupby("Date", sort=True)
            if g.shape[0] >= 5 and g["Signal"].nunique(dropna=True) >= 2 and g["Target"].nunique(dropna=True) >= 2
        ]
        rank_ics = [_safe_spearman(g) for g in grouped]
        rank_ics = [value for value in rank_ics if np.isfinite(value)]
        top_targets: List[float] = []
        bottom_targets: List[float] = []
        all_targets: List[float] = []
        for g in grouped:
            top, bottom = _top_bottom(g, top_fraction)
            top_targets.append(float(top["Target"].mean()))
            bottom_targets.append(float(bottom["Target"].mean()))
            all_targets.append(float(g["Target"].mean()))
        target = pd.to_numeric(group["Target"], errors="coerce")
        signal = pd.to_numeric(group["Signal"], errors="coerce")
        rows.append(
            {
                "Family": family,
                "Horizon": int(horizon),
                "Rows": int(group.shape[0]),
                "Dates": int(group["Date"].nunique()),
                "CrossSectionDates": int(len(grouped)),
                "MeanUsedFeatureCount": float(group["UsedFeatureCount"].mean()),
                "PearsonCorr": _series_corr(signal, target),
                "MeanRankIC": float(np.nanmean(rank_ics)) if rank_ics else np.nan,
                "RankICPositivePct": float((np.array(rank_ics) > 0.0).mean() * 100.0) if rank_ics else np.nan,
                "TopQuintileAvgTargetPct": float(np.nanmean(top_targets)) if top_targets else np.nan,
                "AllAvgTargetPct": float(np.nanmean(all_targets)) if all_targets else np.nan,
                "BottomQuintileAvgTargetPct": float(np.nanmean(bottom_targets)) if bottom_targets else np.nan,
                "TopVsAllPct": (
                    float(np.nanmean(top_targets) - np.nanmean(all_targets)) if top_targets and all_targets else np.nan
                ),
                "TopVsBottomPct": (
                    float(np.nanmean(top_targets) - np.nanmean(bottom_targets))
                    if top_targets and bottom_targets
                    else np.nan
                ),
                "DirectionalHitPct": float((np.sign(signal) == np.sign(target)).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Horizon", "TopVsAllPct", "MeanRankIC"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-back audit of daily OHLC feature families.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tickers", default="auto")
    parser.add_argument("--horizons", default=",".join(str(horizon) for horizon in DEFAULT_HORIZONS))
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--test-block-dates", type=int, default=DEFAULT_TEST_BLOCK_DATES)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--no-index-expiry-exvin", action="store_true")
    parser.add_argument("--write-signals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.tickers).strip().lower() == "auto":
        tickers = _auto_tickers(args.history_dir)
    else:
        tickers = [_normalise_ticker(raw) for raw in str(args.tickers).split(",") if raw.strip()]
    horizons = [int(raw.strip()) for raw in str(args.horizons).split(",") if raw.strip()]
    sample_df, feature_columns = build_daily_family_sample(
        tickers,
        args.history_dir,
        horizons,
        include_index_expiry_exvin=not bool(args.no_index_expiry_exvin),
    )
    if args.target not in sample_df.columns:
        raise SystemExit(f"Unknown target column: {args.target}")
    families = build_daily_feature_families(feature_columns)
    signals, raw_feature_stats = walk_back_daily_feature_families(
        sample_df,
        families,
        target_column=str(args.target),
        min_train_dates=int(args.min_train_dates),
        test_block_dates=int(args.test_block_dates),
    )
    summary = summarise_daily_signals(signals, top_fraction=float(args.top_fraction))
    feature_stats = summarise_feature_stats(raw_feature_stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / OUTPUT_SUMMARY
    feature_path = args.output_dir / OUTPUT_FEATURES
    signal_path = args.output_dir / OUTPUT_SIGNALS
    json_path = args.output_dir / OUTPUT_JSON
    summary.to_csv(summary_path, index=False)
    feature_stats.to_csv(feature_path, index=False)
    if args.write_signals:
        signals.to_csv(signal_path, index=False)
    else:
        signal_path.unlink(missing_ok=True)
    payload = {
        "Target": str(args.target),
        "Horizons": horizons,
        "Tickers": tickers,
        "TickerCount": len(tickers),
        "SampleRows": int(sample_df.shape[0]),
        "SampleDates": int(sample_df["Date"].nunique()),
        "SignalRows": int(signals.shape[0]),
        "FamilyCount": len(families),
        "FeatureCount": int(len(feature_columns)),
        "Outputs": {
            "Summary": str(summary_path),
            "FeatureStats": str(feature_path),
            "Signals": str(signal_path) if args.write_signals else None,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
