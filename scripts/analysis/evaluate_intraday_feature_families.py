from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.analysis.build_intraday_rest_of_session_report import (
    DEFAULT_INTRADAY_HISTORY_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESOLUTION,
    FEATURE_COLUMNS,
    build_multi_ticker_rest_of_session_sample,
    load_intraday_cache_frame,
)
from scripts.analysis.evaluate_ohlc_models import _normalise_ticker


OUTPUT_SUMMARY = "intraday_feature_family_walkback_summary.csv"
OUTPUT_FEATURES = "intraday_feature_family_feature_stats.csv"
OUTPUT_SIGNALS = "intraday_feature_family_walkback_signals.csv"
OUTPUT_JSON = "intraday_feature_family_walkback_summary.json"
DEFAULT_TARGET = "TargetCloseRetPct"
DEFAULT_MIN_TRAIN_DATES = 45
DEFAULT_TEST_BLOCK_DATES = 5
DEFAULT_MIN_GROUP_SIZE = 5
MICRO_VOLUME_WINDOWS = (5, 15, 30, 60)


def _prefixed(columns: Iterable[str], prefixes: Sequence[str]) -> List[str]:
    return [column for column in columns if any(column.startswith(prefix) for prefix in prefixes)]


def build_intraday_feature_families(columns: Sequence[str]) -> Dict[str, List[str]]:
    cols = list(columns)
    families = {
        "time_bucket": [
            "BucketCode",
            "TickerMinutesFromOpen",
            "TickerMinutesToClose",
            "TickerSessionProgressPct",
            "IndexMinutesFromOpen",
            "IndexMinutesToClose",
            "IndexSessionProgressPct",
        ],
        "ticker_price_momentum": [
            "TickerSnapshotRetFromPrevClosePct",
            "TickerGapPct",
            "TickerOpenToSnapshotRetPct",
            "TickerLast5mRetPct",
            "TickerLast15mRetPct",
            "TickerLast30mRetPct",
            "TickerLast60mRetPct",
            "TickerAfternoonOpenToSnapshotRetPct",
        ],
        "ticker_range_position": [
            "TickerRange30mPct",
            "TickerRange60mPct",
            "TickerPosIn30mRange",
            "TickerPosIn60mRange",
            "TickerSessionRangePct",
            "TickerPosInSessionRange",
            "TickerCloseToSessionHighPct",
            "TickerCloseToSessionLowPct",
        ],
        "ticker_vwap": ["TickerVWAPDeviationPct"],
        "ticker_volume": [
            "TickerSessionVolumePctADV20",
            "TickerAfternoonVolumePctADV20",
        ],
        "index_price_momentum": [
            "IndexSnapshotRetFromPrevClosePct",
            "IndexGapPct",
            "IndexOpenToSnapshotRetPct",
            "IndexLast5mRetPct",
            "IndexLast15mRetPct",
            "IndexLast30mRetPct",
            "IndexLast60mRetPct",
            "IndexAfternoonOpenToSnapshotRetPct",
        ],
        "index_range_position": [
            "IndexRange30mPct",
            "IndexRange60mPct",
            "IndexPosIn30mRange",
            "IndexPosIn60mRange",
            "IndexSessionRangePct",
            "IndexPosInSessionRange",
        ],
        "index_vwap": ["IndexVWAPDeviationPct"],
        "index_volume": [
            "IndexSessionVolumePctADV20",
            "IndexAfternoonVolumePctADV20",
        ],
        "relative_price": [
            "RelSnapshotRetFromPrevClosePct",
            "RelOpenToSnapshotPct",
            "RelLast15mPct",
            "RelLast60mPct",
            "RelVWAPDeviationPct",
            "RelAfternoonOpenToSnapshotPct",
        ],
        "relative_volume": ["RelSessionVolumePctADV20"],
        "ticker_micro_volume": _prefixed(
            cols,
            [
                "TickerLast5mVolume",
                "TickerLast15mVolume",
                "TickerLast30mVolume",
                "TickerLast60mVolume",
                "TickerVolumeAcceleration",
                "TickerUpVolumeShare",
                "TickerDownVolumeShare",
            ],
        ),
        "ticker_price_volume_interaction": _prefixed(
            cols,
            [
                "TickerPriceVolumeCorr",
                "TickerSignedVolumePressure",
                "TickerRetTimesVolume",
            ],
        ),
        "index_micro_volume": _prefixed(
            cols,
            [
                "IndexLast5mVolume",
                "IndexLast15mVolume",
                "IndexLast30mVolume",
                "IndexLast60mVolume",
                "IndexVolumeAcceleration",
                "IndexUpVolumeShare",
                "IndexDownVolumeShare",
            ],
        ),
        "index_price_volume_interaction": _prefixed(
            cols,
            [
                "IndexPriceVolumeCorr",
                "IndexSignedVolumePressure",
                "IndexRetTimesVolume",
            ],
        ),
        "relative_micro_volume": _prefixed(cols, ["RelLast", "RelVolumeAcceleration", "RelUpVolumeShare"]),
        "depth_orderbook": _prefixed(cols, ["TickerDepth", "TickerBidAsk", "TickerTopBid", "TickerTopAsk"]),
    }
    families["all_price"] = sorted(
        set(
            families["ticker_price_momentum"]
            + families["ticker_range_position"]
            + families["ticker_vwap"]
            + families["index_price_momentum"]
            + families["index_range_position"]
            + families["index_vwap"]
            + families["relative_price"]
        )
    )
    families["all_volume"] = sorted(
        set(
            families["ticker_volume"]
            + families["index_volume"]
            + families["relative_volume"]
            + families.get("ticker_micro_volume", [])
            + families.get("ticker_price_volume_interaction", [])
            + families.get("index_micro_volume", [])
            + families.get("index_price_volume_interaction", [])
            + families.get("relative_micro_volume", [])
        )
    )
    families["all_price_volume"] = sorted(set(families["all_price"] + families["all_volume"]))
    families["all_intraday_available"] = list(cols)
    return {
        name: [column for column in columns_for_family if column in cols]
        for name, columns_for_family in families.items()
        if any(column in cols for column in columns_for_family)
    }


def _auto_tickers(history_dir: Path, resolution: str) -> List[str]:
    suffix = f"_{str(resolution).strip()}m.csv"
    tickers: List[str] = []
    for path in sorted(history_dir.glob(f"*{suffix}")):
        ticker = _normalise_ticker(path.name[: -len(suffix)])
        if not ticker or ticker == "VNINDEX" or ticker in tickers:
            continue
        tickers.append(ticker)
    return tickers


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _micro_volume_rows(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    adv20_by_date = (
        frame.groupby("TradeDate", sort=True)["Volume"]
        .sum()
        .rolling(20)
        .mean()
        .shift(1)
    )
    frames: List[pd.DataFrame] = []
    for trade_date, session_rows in frame.groupby("TradeDate", sort=True):
        session = session_rows.sort_values("Timestamp").reset_index(drop=True)
        if session.empty:
            continue
        out = pd.DataFrame({"SnapshotTs": session["Timestamp"]})
        close = pd.to_numeric(session["Close"], errors="coerce").astype(float)
        volume = pd.to_numeric(session["Volume"], errors="coerce").fillna(0.0).astype(float)
        returns = close.pct_change(fill_method=None).fillna(0.0)
        signed_volume = np.sign(returns) * volume
        up_volume = volume.where(returns > 0.0, 0.0)
        down_volume = volume.where(returns < 0.0, 0.0)
        day_volume = float(volume.sum())
        adv20 = float(adv20_by_date.get(trade_date, np.nan))

        rolling_volumes: Dict[int, pd.Series] = {}
        for window in MICRO_VOLUME_WINDOWS:
            roll_volume = volume.rolling(window, min_periods=1).sum()
            rolling_volumes[window] = roll_volume
            roll_up = up_volume.rolling(window, min_periods=1).sum()
            roll_down = down_volume.rolling(window, min_periods=1).sum()
            roll_signed = signed_volume.rolling(window, min_periods=1).sum()
            ret_pct = ((close / close.shift(window - 1)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
                corr = returns.rolling(window, min_periods=4).corr(volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            out[f"{prefix}Last{window}mVolumePctSession"] = (roll_volume / day_volume).replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0.0) * 100.0
            out[f"{prefix}Last{window}mVolumePctADV20"] = (roll_volume / adv20).replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0.0) * 100.0
            out[f"{prefix}UpVolumeShare{window}m"] = (roll_up / roll_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[f"{prefix}DownVolumeShare{window}m"] = (roll_down / roll_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[f"{prefix}SignedVolumePressure{window}m"] = (roll_signed / roll_volume).replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0.0)
            out[f"{prefix}PriceVolumeCorr{window}m"] = corr
            out[f"{prefix}RetTimesVolume{window}m"] = ret_pct * out[f"{prefix}Last{window}mVolumePctADV20"]

        out[f"{prefix}VolumeAcceleration15v60"] = ((rolling_volumes[15] / 15.0) / (rolling_volumes[60] / 60.0)).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)
        out[f"{prefix}VolumeAcceleration5v30"] = ((rolling_volumes[5] / 5.0) / (rolling_volumes[30] / 30.0)).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["SnapshotTs"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["SnapshotTs"], keep="last")


def add_micro_volume_features(
    sample_df: pd.DataFrame,
    history_dir: Path,
    resolution: str,
    tickers: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    out = sample_df.copy()
    index_micro = _micro_volume_rows(load_intraday_cache_frame(history_dir, "VNINDEX", resolution), prefix="Index")
    index_columns = [column for column in index_micro.columns if column != "SnapshotTs"]
    out = out.merge(index_micro, on="SnapshotTs", how="left")
    ticker_frames: List[pd.DataFrame] = []
    for ticker in tickers:
        ticker = _normalise_ticker(ticker)
        micro = _micro_volume_rows(load_intraday_cache_frame(history_dir, ticker, resolution), prefix="Ticker")
        if micro.empty:
            continue
        micro["Ticker"] = ticker
        ticker_frames.append(micro)
    ticker_micro = pd.concat(ticker_frames, ignore_index=True) if ticker_frames else pd.DataFrame()
    ticker_columns = [column for column in ticker_micro.columns if column not in {"SnapshotTs", "Ticker"}]
    if not ticker_micro.empty:
        out = out.merge(ticker_micro, on=["Ticker", "SnapshotTs"], how="left")

    relative_columns: List[str] = []
    for window in MICRO_VOLUME_WINDOWS:
        for suffix in ("VolumePctADV20", "VolumePctSession", "SignedVolumePressure", "UpVolumeShare"):
            ticker_col = f"TickerLast{window}m{suffix}" if suffix.startswith("Volume") else f"Ticker{suffix}{window}m"
            index_col = f"IndexLast{window}m{suffix}" if suffix.startswith("Volume") else f"Index{suffix}{window}m"
            if ticker_col in out.columns and index_col in out.columns:
                rel_col = f"RelLast{window}m{suffix}"
                out[rel_col] = pd.to_numeric(out[ticker_col], errors="coerce") - pd.to_numeric(out[index_col], errors="coerce")
                relative_columns.append(rel_col)
    for suffix in ("VolumeAcceleration15v60", "VolumeAcceleration5v30"):
        ticker_col = f"Ticker{suffix}"
        index_col = f"Index{suffix}"
        if ticker_col in out.columns and index_col in out.columns:
            rel_col = f"Rel{suffix}"
            out[rel_col] = pd.to_numeric(out[ticker_col], errors="coerce") - pd.to_numeric(out[index_col], errors="coerce")
            relative_columns.append(rel_col)
    added = index_columns + ticker_columns + relative_columns
    out[added] = out[added].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, added


def _series_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.shape[0] < 8:
        return float("nan")
    if valid["x"].std(ddof=0) <= 1e-12 or valid["y"].std(ddof=0) <= 1e-12:
        return float("nan")
    return float(valid["x"].corr(valid["y"]))


def _safe_spearman(group: pd.DataFrame) -> float:
    if group.shape[0] < DEFAULT_MIN_GROUP_SIZE:
        return float("nan")
    if group["Signal"].nunique(dropna=True) < 2 or group["Target"].nunique(dropna=True) < 2:
        return float("nan")
    return float(group["Signal"].rank().corr(group["Target"].rank()))


def _learn_feature_signs(
    train_df: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    signs: Dict[str, float] = {}
    medians: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    target = pd.to_numeric(train_df[target_column], errors="coerce")
    for feature in feature_columns:
        values = pd.to_numeric(train_df[feature], errors="coerce")
        corr = _series_corr(values, target)
        if not np.isfinite(corr) or abs(corr) <= 1e-12:
            continue
        median = float(values.median())
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        signs[feature] = 1.0 if corr > 0.0 else -1.0
        medians[feature] = median if np.isfinite(median) else 0.0
        scales[feature] = scale
    return signs, medians, scales


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
    out = test_df[
        [
            "TradeDate",
            "SnapshotTs",
            "SnapshotDate",
            "SnapshotTimeBucket",
            "Ticker",
            target_column,
        ]
    ].copy()
    out = out.rename(columns={target_column: "Target"})
    out["Family"] = family
    out["Signal"] = signal.astype(float)
    out["UsedFeatureCount"] = len(usable)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Target", "Signal"])


def walk_back_feature_families(
    sample_df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    *,
    target_column: str,
    min_train_dates: int,
    test_block_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = sample_df[sample_df[target_column].notna()].copy()
    labeled["TradeDate"] = pd.to_datetime(labeled["TradeDate"]).dt.normalize()
    dates = pd.Series(labeled["TradeDate"].drop_duplicates()).sort_values().reset_index(drop=True)
    signal_frames: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []
    if dates.shape[0] <= int(min_train_dates):
        return pd.DataFrame(), pd.DataFrame()

    for start in range(int(min_train_dates), int(dates.shape[0]), int(test_block_dates)):
        train_dates = set(dates.iloc[:start].tolist())
        test_dates = set(dates.iloc[start : start + int(test_block_dates)].tolist())
        train_df = labeled[labeled["TradeDate"].isin(train_dates)].copy()
        test_df = labeled[labeled["TradeDate"].isin(test_dates)].copy()
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
                        "BlockStartDate": dates.iloc[start].strftime("%Y-%m-%d"),
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


def _top_bottom_rows(group: pd.DataFrame, top_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = group.sort_values("Signal", ascending=False)
    count = max(1, int(math.ceil(float(group.shape[0]) * float(top_fraction))))
    return ordered.head(count), ordered.tail(count)


def summarise_family_signals(
    signals: pd.DataFrame,
    *,
    top_fraction: float,
    min_group_size: int,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for family, group in signals.groupby("Family", sort=False):
        grouped = [
            g
            for _, g in group.groupby("SnapshotTs", sort=True)
            if g.shape[0] >= int(min_group_size)
            and g["Signal"].nunique(dropna=True) >= 2
            and g["Target"].nunique(dropna=True) >= 2
        ]
        rank_ics = [_safe_spearman(g) for g in grouped]
        rank_ics = [value for value in rank_ics if np.isfinite(value)]
        top_targets: List[float] = []
        bottom_targets: List[float] = []
        all_targets: List[float] = []
        for g in grouped:
            top, bottom = _top_bottom_rows(g, top_fraction)
            top_targets.append(float(top["Target"].mean()))
            bottom_targets.append(float(bottom["Target"].mean()))
            all_targets.append(float(g["Target"].mean()))
        snapshot_means = (
            group.groupby("SnapshotTs", sort=True)
            .agg(Signal=("Signal", "mean"), Target=("Target", "mean"))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        temporal_corr = (
            _series_corr(snapshot_means["Signal"], snapshot_means["Target"])
            if not snapshot_means.empty
            else np.nan
        )
        if snapshot_means.empty:
            high_signal_snapshot_avg = np.nan
            low_signal_snapshot_avg = np.nan
            high_signal_snapshot_vs_all = np.nan
        else:
            top_snapshots, bottom_snapshots = _top_bottom_rows(snapshot_means.reset_index(), top_fraction)
            high_signal_snapshot_avg = float(top_snapshots["Target"].mean())
            low_signal_snapshot_avg = float(bottom_snapshots["Target"].mean())
            high_signal_snapshot_vs_all = float(top_snapshots["Target"].mean() - snapshot_means["Target"].mean())
        sorted_dates = pd.Series(group["TradeDate"].drop_duplicates()).sort_values().reset_index(drop=True)
        midpoint = sorted_dates.iloc[int(len(sorted_dates) / 2)] if not sorted_dates.empty else None
        first_half = group[group["TradeDate"] <= midpoint] if midpoint is not None else group.iloc[0:0]
        second_half = group[group["TradeDate"] > midpoint] if midpoint is not None else group.iloc[0:0]
        first_ics = [
            _safe_spearman(g)
            for _, g in first_half.groupby("SnapshotTs")
            if g.shape[0] >= int(min_group_size) and g["Signal"].nunique(dropna=True) >= 2
        ]
        second_ics = [
            _safe_spearman(g)
            for _, g in second_half.groupby("SnapshotTs")
            if g.shape[0] >= int(min_group_size) and g["Signal"].nunique(dropna=True) >= 2
        ]
        first_ics = [value for value in first_ics if np.isfinite(value)]
        second_ics = [value for value in second_ics if np.isfinite(value)]
        target = pd.to_numeric(group["Target"], errors="coerce")
        signal = pd.to_numeric(group["Signal"], errors="coerce")
        rows.append(
            {
                "Family": family,
                "Rows": int(group.shape[0]),
                "Dates": int(group["TradeDate"].nunique()),
                "SnapshotGroups": int(group["SnapshotTs"].nunique()),
                "CrossSectionGroups": int(len(grouped)),
                "MeanUsedFeatureCount": float(group["UsedFeatureCount"].mean()),
                "PearsonCorr": _series_corr(signal, target),
                "TemporalSnapshotCorr": temporal_corr,
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
                "HighSignalSnapshotAvgTargetPct": high_signal_snapshot_avg,
                "LowSignalSnapshotAvgTargetPct": low_signal_snapshot_avg,
                "HighSignalSnapshotVsAllPct": high_signal_snapshot_vs_all,
                "DirectionalHitPct": float((np.sign(signal) == np.sign(target)).mean() * 100.0),
                "FirstHalfMeanRankIC": float(np.nanmean(first_ics)) if first_ics else np.nan,
                "SecondHalfMeanRankIC": float(np.nanmean(second_ics)) if second_ics else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["TopVsAllPct", "MeanRankIC", "HighSignalSnapshotVsAllPct", "TemporalSnapshotCorr"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def summarise_feature_stats(feature_stats: pd.DataFrame) -> pd.DataFrame:
    if feature_stats.empty:
        return pd.DataFrame()
    grouped = feature_stats.groupby(["Family", "Feature"], sort=False)
    out = grouped.agg(
        Blocks=("BlockStartDate", "nunique"),
        SelectedPct=("Selected", lambda s: float(s.astype(bool).mean() * 100.0)),
        MeanTrainCorr=("TrainCorr", "mean"),
        MeanTrainAbsCorr=("TrainAbsCorr", "mean"),
        PositiveSignPct=("TrainSign", lambda s: float((pd.to_numeric(s, errors="coerce") > 0.0).mean() * 100.0)),
        MeanCoveragePct=("TrainCoveragePct", "mean"),
    ).reset_index()
    return out.sort_values(["MeanTrainAbsCorr", "SelectedPct"], ascending=[False, False]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-back audit of intraday feature families.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_INTRADAY_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--tickers", default="auto", help="Comma-separated tickers or `auto` from intraday cache.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--test-block-dates", type=int, default=DEFAULT_TEST_BLOCK_DATES)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    parser.add_argument("--no-micro-volume", action="store_true")
    parser.add_argument("--write-signals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.tickers).strip().lower() == "auto":
        tickers = _auto_tickers(args.history_dir, str(args.resolution))
    else:
        tickers = [_normalise_ticker(raw) for raw in str(args.tickers).split(",") if raw.strip()]
    if not tickers:
        raise SystemExit(f"No tickers resolved from {args.history_dir}")

    sample_df = build_multi_ticker_rest_of_session_sample(
        tickers,
        args.history_dir,
        str(args.resolution),
        require_depth=False,
    )
    feature_columns = list(FEATURE_COLUMNS)
    micro_feature_count = 0
    if not bool(args.no_micro_volume):
        sample_df, micro_features = add_micro_volume_features(
            sample_df,
            args.history_dir,
            str(args.resolution),
            tickers,
        )
        feature_columns = feature_columns + micro_features
        micro_feature_count = len(micro_features)
    if args.target not in sample_df.columns:
        raise SystemExit(f"Unknown target column: {args.target}")
    families = build_intraday_feature_families(feature_columns)
    signals, raw_feature_stats = walk_back_feature_families(
        sample_df,
        families,
        target_column=str(args.target),
        min_train_dates=int(args.min_train_dates),
        test_block_dates=int(args.test_block_dates),
    )
    summary = summarise_family_signals(
        signals,
        top_fraction=float(args.top_fraction),
        min_group_size=int(args.min_group_size),
    )
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
        "Tickers": tickers,
        "TickerCount": len(tickers),
        "SampleRows": int(sample_df.shape[0]),
        "SampleDates": int(pd.to_datetime(sample_df["TradeDate"]).dt.normalize().nunique()),
        "SignalRows": int(signals.shape[0]),
        "FamilyCount": len(families),
        "FeatureCount": int(len(feature_columns)),
        "MicroFeatureCount": int(micro_feature_count),
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
