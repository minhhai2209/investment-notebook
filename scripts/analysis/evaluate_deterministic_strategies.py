from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_CASE_TICKERS = ["HPG", "FPT", "SSI", "VCB", "NKG"]
MARKET_STATES = ("DEPLOY", "WAIT", "REDUCE")


@dataclass(frozen=True)
class FeaturePack:
    close: pd.DataFrame
    volume: pd.DataFrame
    index_close: pd.Series
    sectors: Dict[str, str]
    ret1: pd.DataFrame
    ret5: pd.DataFrame
    ret20: pd.DataFrame
    ret60: pd.DataFrame
    dist20: pd.DataFrame
    dist50: pd.DataFrame
    dist100: pd.DataFrame
    vol_ratio20: pd.DataFrame
    rel20: pd.DataFrame
    rel60: pd.DataFrame
    rel20_rank: pd.DataFrame
    rel60_rank: pd.DataFrame
    breadth20: pd.Series
    breadth50: pd.Series
    breadth_positive5: pd.Series
    index_range60: pd.Series
    index_sma20: pd.Series
    index_sma50: pd.Series
    index_sma200: pd.Series
    sector_breadth20: pd.DataFrame
    sector_rel20: pd.DataFrame
    sector_support: pd.DataFrame
    sector_rel_support: pd.DataFrame
    sector_rank_per_ticker: pd.DataFrame
    corr20: pd.DataFrame
    beta20: pd.DataFrame
    fwd5: pd.DataFrame
    fwd10: pd.DataFrame
    excess5: pd.DataFrame
    excess10: pd.DataFrame
    asset_future_dd10: pd.DataFrame
    index_fwd5: pd.Series
    index_fwd10: pd.Series
    index_future_dd10: pd.Series


@dataclass(frozen=True)
class SignalBundle:
    score: pd.DataFrame
    active: pd.DataFrame


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _load_sector_map(path: Path) -> Dict[str, str]:
    sector_df = pd.read_csv(path)
    if "Ticker" not in sector_df.columns or "Sector" not in sector_df.columns:
        raise ValueError(f"Invalid sector map: {path}")
    return {
        _normalise_ticker(ticker): str(sector).strip() or "Unknown"
        for ticker, sector in zip(sector_df["Ticker"], sector_df["Sector"])
    }


def _resolve_ohlcv_column(frame: pd.DataFrame, column: str) -> str:
    for candidate in (column, column.lower(), column.upper(), column.capitalize()):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Missing required column '{column}' in history frame")


def _load_history_matrices(history_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_series: Dict[str, pd.Series] = {}
    volume_series: Dict[str, pd.Series] = {}
    for csv_path in sorted(history_dir.glob("*_daily.csv")):
        ticker = _normalise_ticker(csv_path.stem.replace("_daily", ""))
        raw = pd.read_csv(csv_path)
        if raw.empty:
            continue
        date_col = "date_vn" if "date_vn" in raw.columns else _resolve_ohlcv_column(raw, "t")
        if date_col == "date_vn":
            index = pd.to_datetime(raw[date_col], errors="coerce")
        else:
            index = pd.to_datetime(pd.to_numeric(raw[date_col], errors="coerce"), unit="s", errors="coerce")
        close_col = _resolve_ohlcv_column(raw, "close")
        volume_col = _resolve_ohlcv_column(raw, "volume")
        close_series[ticker] = pd.Series(pd.to_numeric(raw[close_col], errors="coerce").values, index=index)
        volume_series[ticker] = pd.Series(pd.to_numeric(raw[volume_col], errors="coerce").values, index=index)
    if not close_series:
        raise RuntimeError(f"No cached daily data found in {history_dir}")
    close = pd.DataFrame(close_series).sort_index()
    close = close[~close.index.isna()]
    close = close.loc[:, ~close.columns.duplicated()]
    volume = pd.DataFrame(volume_series).sort_index().reindex(close.index)
    if "VNINDEX" not in close.columns:
        raise RuntimeError("Expected VNINDEX_daily.csv in history cache")
    return close, volume


def _asset_columns(columns: Iterable[str]) -> List[str]:
    return [ticker for ticker in columns if ticker not in {"VNINDEX", "VN30", "VN100"}]


def _scale_range(frame: pd.DataFrame | pd.Series, lower: float, upper: float) -> pd.DataFrame | pd.Series:
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    clipped = frame.clip(lower=lower, upper=upper)
    return ((clipped - lower) / (upper - lower)) * 100.0


def _forward_extreme(frame: pd.DataFrame | pd.Series, horizon: int, mode: str) -> pd.DataFrame | pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    shifts = [frame.shift(-step) for step in range(1, horizon + 1)]
    if not shifts:
        return frame.copy()
    result = shifts[0].copy()
    reducer = np.fmin if mode == "min" else np.fmax
    for shifted in shifts[1:]:
        result = result.combine(shifted, reducer)
    return result


def _broadcast_sector_metric(metric_df: pd.DataFrame, sector_lookup: Dict[str, str], tickers: Sequence[str]) -> pd.DataFrame:
    values = {}
    for ticker in tickers:
        sector = sector_lookup.get(ticker, "Unknown")
        if sector in metric_df.columns:
            values[ticker] = metric_df[sector]
        else:
            values[ticker] = pd.Series(np.nan, index=metric_df.index)
    return pd.DataFrame(values, index=metric_df.index)


def build_feature_pack(history_dir: Path, sector_map_path: Path) -> FeaturePack:
    sectors = _load_sector_map(sector_map_path)
    close_all, volume_all = _load_history_matrices(history_dir)
    assets = _asset_columns(close_all.columns)
    close = close_all[assets]
    volume = volume_all[assets]
    index_close = pd.to_numeric(close_all["VNINDEX"], errors="coerce")

    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()
    dist20 = (close / sma20) - 1.0
    dist50 = (close / sma50) - 1.0
    dist100 = (close / sma100) - 1.0
    vol20 = volume.rolling(20).mean()
    vol_ratio20 = volume / vol20

    index_ret20 = index_close.pct_change(20)
    index_ret60 = index_close.pct_change(60)
    rel20 = ret20.sub(index_ret20, axis=0)
    rel60 = ret60.sub(index_ret60, axis=0)
    rel20_rank = rel20.rank(axis=1, method="average", pct=True) * 100.0
    rel60_rank = rel60.rank(axis=1, method="average", pct=True) * 100.0

    breadth20 = (dist20 > 0).mean(axis=1) * 100.0
    breadth50 = (dist50 > 0).mean(axis=1) * 100.0
    breadth_positive5 = (ret5 > 0).mean(axis=1) * 100.0
    index_range60 = (index_close - index_close.rolling(60).min()) / (
        index_close.rolling(60).max() - index_close.rolling(60).min()
    )
    index_sma20 = index_close.rolling(20).mean()
    index_sma50 = index_close.rolling(50).mean()
    index_sma200 = index_close.rolling(200).mean()

    sector_names = sorted({sectors.get(ticker, "Unknown") for ticker in assets})
    sector_members = {sector: [ticker for ticker in assets if sectors.get(ticker, "Unknown") == sector] for sector in sector_names}
    sector_breadth20 = pd.DataFrame(index=close.index)
    sector_rel20 = pd.DataFrame(index=close.index)
    for sector, members in sector_members.items():
        if not members:
            continue
        sector_breadth20[sector] = (dist20[members] > 0).mean(axis=1) * 100.0
        sector_rel20[sector] = rel20[members].median(axis=1)
    sector_support = _broadcast_sector_metric(sector_breadth20, sectors, assets)
    sector_rel_support = _broadcast_sector_metric(sector_rel20, sectors, assets)
    sector_rank_per_ticker = _broadcast_sector_metric(
        sector_rel20.rank(axis=1, method="average", pct=True) * 100.0,
        sectors,
        assets,
    )

    index_returns_1d = index_close.pct_change()
    corr20 = ret1.rolling(20).corr(index_returns_1d)
    index_var20 = index_returns_1d.rolling(20).var()
    beta20 = pd.DataFrame(index=close.index, columns=assets, dtype=float)
    for ticker in assets:
        beta20[ticker] = ret1[ticker].rolling(20).cov(index_returns_1d) / index_var20

    fwd5 = (close.shift(-5) / close) - 1.0
    fwd10 = (close.shift(-10) / close) - 1.0
    excess5 = fwd5.sub((index_close.shift(-5) / index_close) - 1.0, axis=0)
    excess10 = fwd10.sub((index_close.shift(-10) / index_close) - 1.0, axis=0)
    asset_future_dd10 = (_forward_extreme(close, 10, "min") / close) - 1.0
    index_fwd5 = (index_close.shift(-5) / index_close) - 1.0
    index_fwd10 = (index_close.shift(-10) / index_close) - 1.0
    index_future_dd10 = (_forward_extreme(index_close, 10, "min") / index_close) - 1.0

    return FeaturePack(
        close=close,
        volume=volume,
        index_close=index_close,
        sectors=sectors,
        ret1=ret1,
        ret5=ret5,
        ret20=ret20,
        ret60=ret60,
        dist20=dist20,
        dist50=dist50,
        dist100=dist100,
        vol_ratio20=vol_ratio20,
        rel20=rel20,
        rel60=rel60,
        rel20_rank=rel20_rank,
        rel60_rank=rel60_rank,
        breadth20=breadth20,
        breadth50=breadth50,
        breadth_positive5=breadth_positive5,
        index_range60=index_range60,
        index_sma20=index_sma20,
        index_sma50=index_sma50,
        index_sma200=index_sma200,
        sector_breadth20=sector_breadth20,
        sector_rel20=sector_rel20,
        sector_support=sector_support,
        sector_rel_support=sector_rel_support,
        sector_rank_per_ticker=sector_rank_per_ticker,
        corr20=corr20,
        beta20=beta20,
        fwd5=fwd5,
        fwd10=fwd10,
        excess5=excess5,
        excess10=excess10,
        asset_future_dd10=asset_future_dd10,
        index_fwd5=index_fwd5,
        index_fwd10=index_fwd10,
        index_future_dd10=index_future_dd10,
    )


def _market_engine_hybrid(features: FeaturePack) -> pd.Series:
    signals = pd.Series("WAIT", index=features.index_close.index, dtype="object")
    regime = pd.Series("Sideway", index=features.index_close.index, dtype="object")
    bull_mask = (features.index_close > features.index_sma50) & (features.index_sma50 > features.index_sma200)
    bear_mask = (features.index_close < features.index_sma50) & (features.index_sma50 < features.index_sma200)
    regime.loc[bull_mask] = "Bull"
    regime.loc[bear_mask] = "Bear"

    for date in signals.index:
        range_pos = features.index_range60.loc[date]
        breadth20 = features.breadth20.loc[date]
        breadth50 = features.breadth50.loc[date]
        breadth_positive5 = features.breadth_positive5.loc[date]
        if pd.isna(range_pos) or pd.isna(breadth20) or pd.isna(breadth50) or pd.isna(breadth_positive5):
            continue

        deploy_score = 0.0
        reduce_score = 0.0
        wait_score = 0.0
        regime_value = regime.loc[date]
        if regime_value == "Bull":
            deploy_score += 35.0
        elif regime_value == "Bear":
            reduce_score += 35.0
        else:
            wait_score += 25.0

        if range_pos <= 0.25:
            deploy_score += 15.0 if regime_value != "Bear" else 5.0
            wait_score += 5.0
        elif range_pos >= 0.75:
            reduce_score += 15.0
        else:
            wait_score += 10.0

        if breadth20 >= 60.0 and breadth50 >= 50.0:
            deploy_score += 20.0
        elif breadth20 <= 35.0 and breadth50 <= 35.0:
            reduce_score += 20.0
        else:
            wait_score += 10.0

        if breadth_positive5 >= 55.0:
            deploy_score += 10.0
        elif breadth_positive5 <= 40.0:
            reduce_score += 10.0
        else:
            wait_score += 5.0

        state_scores = {"DEPLOY": deploy_score, "REDUCE": reduce_score, "WAIT": wait_score}
        signals.loc[date] = max(state_scores, key=state_scores.get)
    return signals


def _market_trend_breadth(features: FeaturePack) -> pd.Series:
    signals = pd.Series("WAIT", index=features.index_close.index, dtype="object")
    deploy_mask = (
        (features.index_close > features.index_sma20)
        & (features.breadth20 >= 55.0)
        & (features.breadth50 >= 45.0)
    )
    reduce_mask = (
        (features.index_close < features.index_sma20)
        & (features.breadth20 <= 40.0)
        & (features.breadth50 <= 35.0)
    )
    signals.loc[deploy_mask] = "DEPLOY"
    signals.loc[reduce_mask] = "REDUCE"
    return signals


def _market_trend_stack(features: FeaturePack) -> pd.Series:
    signals = pd.Series("WAIT", index=features.index_close.index, dtype="object")
    deploy_mask = (
        (features.index_close > features.index_sma20)
        & (features.index_sma20 > features.index_sma50)
        & (features.breadth20 >= 50.0)
    )
    reduce_mask = (
        (features.index_close < features.index_sma20)
        & (features.index_sma20 < features.index_sma50)
        & (features.breadth20 <= 40.0)
    )
    signals.loc[deploy_mask] = "DEPLOY"
    signals.loc[reduce_mask] = "REDUCE"
    return signals


def _market_exhaustion_overlay(features: FeaturePack) -> pd.Series:
    signals = pd.Series("WAIT", index=features.index_close.index, dtype="object")
    deploy_mask = (
        (features.index_close > features.index_sma20)
        & (features.breadth20 >= 60.0)
        & (features.breadth_positive5 >= 55.0)
    )
    reduce_mask = (features.index_range60 >= 0.85) & (features.breadth_positive5 <= 45.0)
    signals.loc[deploy_mask] = "DEPLOY"
    signals.loc[reduce_mask] = "REDUCE"
    return signals


def build_market_signal_table(features: FeaturePack) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_hybrid": _market_engine_hybrid(features),
            "trend_breadth": _market_trend_breadth(features),
            "trend_stack": _market_trend_stack(features),
            "exhaustion_overlay": _market_exhaustion_overlay(features),
        },
        index=features.index_close.index,
    )


def summarise_market_algorithms(features: FeaturePack, market_signals: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = features.index_fwd10.notna()
    baseline_ret10 = float((features.index_fwd10[baseline_mask] * 100.0).mean())
    baseline_hit10 = float(((features.index_fwd10[baseline_mask] > 0).mean()) * 100.0)
    baseline_dd10 = float((features.index_future_dd10[baseline_mask] * 100.0).mean())

    rows: List[Dict[str, object]] = []
    for algorithm in market_signals.columns:
        signal_series = market_signals[algorithm]
        row: Dict[str, object] = {
            "Algorithm": algorithm,
            "CurrentBias": signal_series.iloc[-1],
            "BaselineAvgRet10Pct": baseline_ret10,
            "BaselineHit10Pct": baseline_hit10,
            "BaselineAvgDrawdown10Pct": baseline_dd10,
        }
        for state in MARKET_STATES:
            state_mask = (signal_series == state) & baseline_mask
            row[f"{state}Days"] = int(state_mask.sum())
            if state_mask.any():
                row[f"{state}AvgRet5Pct"] = float((features.index_fwd5[state_mask] * 100.0).mean())
                row[f"{state}AvgRet10Pct"] = float((features.index_fwd10[state_mask] * 100.0).mean())
                row[f"{state}Hit10Pct"] = float(((features.index_fwd10[state_mask] > 0).mean()) * 100.0)
                row[f"{state}AvgDrawdown10Pct"] = float((features.index_future_dd10[state_mask] * 100.0).mean())
            else:
                row[f"{state}AvgRet5Pct"] = float("nan")
                row[f"{state}AvgRet10Pct"] = float("nan")
                row[f"{state}Hit10Pct"] = float("nan")
                row[f"{state}AvgDrawdown10Pct"] = float("nan")
        row["DeployVsAll10Pct"] = row["DEPLOYAvgRet10Pct"] - baseline_ret10
        row["ReduceVsAll10Pct"] = row["REDUCEAvgRet10Pct"] - baseline_ret10
        row["DeployMinusReduce10Pct"] = row["DEPLOYAvgRet10Pct"] - row["REDUCEAvgRet10Pct"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(["DeployVsAll10Pct", "DEPLOYHit10Pct"], ascending=[False, False]).reset_index(drop=True)


def choose_market_algorithm(summary_df: pd.DataFrame, min_deploy_days: int = 50) -> str:
    if summary_df.empty:
        raise ValueError("market summary is empty")
    candidates = summary_df[summary_df["DEPLOYDays"] >= min_deploy_days]
    if candidates.empty:
        candidates = summary_df
    ordered = candidates.sort_values(
        ["DeployVsAll10Pct", "DEPLOYHit10Pct", "DeployMinusReduce10Pct"],
        ascending=[False, False, False],
    )
    return str(ordered.iloc[0]["Algorithm"])


def build_ticker_signal_bundles(features: FeaturePack, market_gate: pd.Series) -> Dict[str, SignalBundle]:
    gate_buyable = market_gate.isin({"DEPLOY", "WAIT"}).to_numpy()[:, None]

    trend_score = (
        (0.35 * features.rel20_rank)
        + (0.25 * features.rel60_rank)
        + (0.15 * _scale_range(features.dist20, -0.15, 0.15))
        + (0.10 * _scale_range(features.dist50, -0.20, 0.20))
        + (0.15 * _scale_range(features.vol_ratio20, 0.0, 2.0))
    )
    trend_active = (
        (features.dist20 > 0)
        & (features.dist50 > 0)
        & (features.rel20 > 0)
        & (features.rel60 > 0)
        & gate_buyable
    )

    breakout_score = (
        (0.30 * features.rel20_rank)
        + (0.20 * features.rel60_rank)
        + (0.15 * features.sector_support)
        + (0.20 * _scale_range(features.vol_ratio20, 0.0, 3.0))
        + (0.15 * _scale_range(features.dist20, 0.0, 0.15))
    )
    breakout_active = (
        (features.dist20 > 0)
        & (features.dist50 > 0)
        & (features.ret5 > 0)
        & (features.vol_ratio20 > 1.0)
        & (features.sector_support >= 40.0)
        & gate_buyable
    )

    recovery_score = (
        (0.25 * features.rel20_rank)
        + (0.30 * features.rel60_rank)
        + (0.20 * _scale_range((-features.dist20).clip(lower=0.0), 0.0, 0.08))
        + (0.15 * features.sector_support)
        + (0.10 * _scale_range(features.vol_ratio20, 0.0, 2.0))
    )
    recovery_active = (
        (features.rel60 > 0)
        & (features.ret5 < 0)
        & (features.dist20 < 0)
        & (features.dist20 > -0.10)
        & (features.dist50 > -0.03)
        & (features.sector_support >= 20.0)
        & (features.sector_rel_support > 0)
        & gate_buyable
    )

    sector_rotation_score = (
        (0.40 * features.sector_rank_per_ticker)
        + (0.25 * features.sector_support)
        + (0.20 * features.rel20_rank)
        + (0.15 * features.rel60_rank)
    )
    sector_rotation_active = (
        (features.sector_rank_per_ticker >= 70.0)
        & (features.sector_support >= 50.0)
        & (features.rel20_rank >= 60.0)
        & gate_buyable
    )

    residual_momentum_score = (
        (0.40 * features.rel20_rank)
        + (0.25 * features.rel60_rank)
        + (0.20 * _scale_range(features.dist100, -0.20, 0.20))
        + (0.15 * features.sector_support)
    )
    residual_momentum_active = (
        (features.rel20 > 0)
        & (features.rel60 > 0)
        & (features.close > features.close.rolling(50).mean())
        & (features.sector_support >= 35.0)
        & gate_buyable
    )

    return {
        "trend_leader": SignalBundle(score=trend_score, active=trend_active),
        "breakout_confirmation": SignalBundle(score=breakout_score, active=breakout_active),
        "recovery_strength": SignalBundle(score=recovery_score, active=recovery_active),
        "sector_rotation": SignalBundle(score=sector_rotation_score, active=sector_rotation_active),
        "residual_momentum": SignalBundle(score=residual_momentum_score, active=residual_momentum_active),
    }


def _collect_replay_rows(
    features: FeaturePack,
    market_gate_name: str,
    market_gate: pd.Series,
    signal_bundles: Dict[str, SignalBundle],
    top_k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    evaluable_dates = features.fwd10.index[features.fwd10.notna().any(axis=1)]
    for algorithm, bundle in signal_bundles.items():
        scored = bundle.score.where(bundle.active)
        for date in evaluable_dates:
            candidates = scored.loc[date].dropna().sort_values(ascending=False).head(top_k)
            if candidates.empty:
                continue
            for rank, (ticker, score) in enumerate(candidates.items(), start=1):
                rows.append(
                    {
                        "Date": date.date().isoformat(),
                        "MarketGateAlgorithm": market_gate_name,
                        "MarketGateBias": market_gate.loc[date],
                        "Algorithm": algorithm,
                        "Rank": rank,
                        "Ticker": ticker,
                        "Sector": features.sectors.get(ticker, "Unknown"),
                        "Score": float(score),
                        "Fwd5Pct": float(features.fwd5.loc[date, ticker] * 100.0),
                        "Fwd10Pct": float(features.fwd10.loc[date, ticker] * 100.0),
                        "Excess5Pct": float(features.excess5.loc[date, ticker] * 100.0),
                        "Excess10Pct": float(features.excess10.loc[date, ticker] * 100.0),
                        "FutureDrawdown10Pct": float(features.asset_future_dd10.loc[date, ticker] * 100.0),
                    }
                )
    return pd.DataFrame(rows)


def summarise_ticker_algorithms(replay_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if replay_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for algorithm, group in replay_df.groupby("Algorithm", sort=False):
        basket_daily = (
            group.groupby("Date", as_index=False)[["Fwd5Pct", "Fwd10Pct", "Excess5Pct", "Excess10Pct"]]
            .mean()
            .rename(
                columns={
                    "Fwd5Pct": "BasketFwd5Pct",
                    "Fwd10Pct": "BasketFwd10Pct",
                    "Excess5Pct": "BasketExcess5Pct",
                    "Excess10Pct": "BasketExcess10Pct",
                }
            )
        )
        rows.append(
            {
                "Algorithm": algorithm,
                "SignalDays": int(basket_daily.shape[0]),
                "TotalPicks": int(group.shape[0]),
                "AvgPicksPerDay": float(group.shape[0] / basket_daily.shape[0]) if basket_daily.shape[0] else 0.0,
                "ConfiguredTopK": top_k,
                "BasketAvgFwd5Pct": float(basket_daily["BasketFwd5Pct"].mean()),
                "BasketAvgFwd10Pct": float(basket_daily["BasketFwd10Pct"].mean()),
                "BasketAvgExcess5Pct": float(basket_daily["BasketExcess5Pct"].mean()),
                "BasketAvgExcess10Pct": float(basket_daily["BasketExcess10Pct"].mean()),
                "BasketHit5Pct": float((basket_daily["BasketFwd5Pct"] > 0).mean() * 100.0),
                "BasketHit10Pct": float((basket_daily["BasketFwd10Pct"] > 0).mean() * 100.0),
                "PickAvgFwd5Pct": float(group["Fwd5Pct"].mean()),
                "PickAvgFwd10Pct": float(group["Fwd10Pct"].mean()),
                "PickAvgExcess5Pct": float(group["Excess5Pct"].mean()),
                "PickAvgExcess10Pct": float(group["Excess10Pct"].mean()),
                "PickHit5Pct": float((group["Fwd5Pct"] > 0).mean() * 100.0),
                "PickHit10Pct": float((group["Fwd10Pct"] > 0).mean() * 100.0),
                "PickAvgDrawdown10Pct": float(group["FutureDrawdown10Pct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["BasketAvgExcess10Pct", "BasketHit10Pct", "BasketAvgFwd10Pct"],
        ascending=[False, False, False],
    )


def build_current_market_snapshot(features: FeaturePack, market_signals: pd.DataFrame) -> pd.DataFrame:
    latest = market_signals.index[-1]
    rows: List[Dict[str, object]] = []
    for algorithm in market_signals.columns:
        rows.append(
            {
                "Date": latest.date().isoformat(),
                "Algorithm": algorithm,
                "CurrentBias": market_signals.loc[latest, algorithm],
                "IndexClose": float(features.index_close.loc[latest]),
                "IndexRange60": float(features.index_range60.loc[latest]),
                "Breadth20Pct": float(features.breadth20.loc[latest]),
                "Breadth50Pct": float(features.breadth50.loc[latest]),
                "BreadthPositive5Pct": float(features.breadth_positive5.loc[latest]),
            }
        )
    snapshot = pd.DataFrame(rows)
    counts = snapshot["CurrentBias"].value_counts()
    consensus = "WAIT"
    if counts.get("REDUCE", 0) >= 2:
        consensus = "REDUCE"
    elif counts.get("DEPLOY", 0) >= 2:
        consensus = "DEPLOY"
    snapshot["ConsensusBias"] = consensus
    return snapshot


def build_current_top_picks(
    features: FeaturePack,
    market_gate_name: str,
    market_gate: pd.Series,
    signal_bundles: Dict[str, SignalBundle],
    top_k: int,
) -> pd.DataFrame:
    latest = features.close.index[-1]
    rows: List[Dict[str, object]] = []
    for algorithm, bundle in signal_bundles.items():
        ranked = bundle.score.where(bundle.active).loc[latest].dropna().sort_values(ascending=False).head(top_k)
        for rank, (ticker, score) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "Date": latest.date().isoformat(),
                    "MarketGateAlgorithm": market_gate_name,
                    "MarketGateBias": market_gate.loc[latest],
                    "Algorithm": algorithm,
                    "Rank": rank,
                    "Ticker": ticker,
                    "Sector": features.sectors.get(ticker, "Unknown"),
                    "Score": float(score),
                    "Rel20Pct": float(features.rel20.loc[latest, ticker] * 100.0),
                    "Rel60Pct": float(features.rel60.loc[latest, ticker] * 100.0),
                    "DistSMA20Pct": float(features.dist20.loc[latest, ticker] * 100.0),
                    "DistSMA50Pct": float(features.dist50.loc[latest, ticker] * 100.0),
                    "SectorBreadth20Pct": float(features.sector_support.loc[latest, ticker]),
                    "VolumeRatio20": float(features.vol_ratio20.loc[latest, ticker]),
                }
            )
    return pd.DataFrame(rows)


def build_case_studies(
    features: FeaturePack,
    market_gate_name: str,
    market_gate: pd.Series,
    signal_bundles: Dict[str, SignalBundle],
    replay_df: pd.DataFrame,
    case_tickers: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    latest = features.close.index[-1]
    current_top = build_current_top_picks(features, market_gate_name, market_gate, signal_bundles, top_k)
    rows: List[Dict[str, object]] = []
    for ticker in case_tickers:
        if ticker not in features.close.columns:
            continue
        for algorithm, bundle in signal_bundles.items():
            score = bundle.score.loc[latest, ticker]
            active = bool(bundle.active.loc[latest, ticker])
            current_rows = current_top[(current_top["Algorithm"] == algorithm) & (current_top["Ticker"] == ticker)]
            history_rows = replay_df[(replay_df["Algorithm"] == algorithm) & (replay_df["Ticker"] == ticker)]
            rows.append(
                {
                    "Date": latest.date().isoformat(),
                    "Ticker": ticker,
                    "Sector": features.sectors.get(ticker, "Unknown"),
                    "MarketGateAlgorithm": market_gate_name,
                    "MarketGateBias": market_gate.loc[latest],
                    "Algorithm": algorithm,
                    "CurrentScore": float(score) if pd.notna(score) else float("nan"),
                    "CurrentlyActive": active,
                    "CurrentTopKRank": int(current_rows.iloc[0]["Rank"]) if not current_rows.empty else np.nan,
                    "CurrentRel20Pct": float(features.rel20.loc[latest, ticker] * 100.0),
                    "CurrentRel60Pct": float(features.rel60.loc[latest, ticker] * 100.0),
                    "CurrentDistSMA20Pct": float(features.dist20.loc[latest, ticker] * 100.0),
                    "CurrentDistSMA50Pct": float(features.dist50.loc[latest, ticker] * 100.0),
                    "CurrentSectorBreadth20Pct": float(features.sector_support.loc[latest, ticker]),
                    "HistoricalPickCount": int(history_rows.shape[0]),
                    "HistoricalAvgFwd5Pct": float(history_rows["Fwd5Pct"].mean()) if not history_rows.empty else float("nan"),
                    "HistoricalAvgFwd10Pct": float(history_rows["Fwd10Pct"].mean()) if not history_rows.empty else float("nan"),
                    "HistoricalAvgExcess10Pct": float(history_rows["Excess10Pct"].mean()) if not history_rows.empty else float("nan"),
                    "HistoricalHit10Pct": float((history_rows["Fwd10Pct"] > 0).mean() * 100.0)
                    if not history_rows.empty
                    else float("nan"),
                    "LastHistoricalPickDate": history_rows["Date"].max() if not history_rows.empty else "",
                }
            )
    return pd.DataFrame(rows)


def _print_stdout_summary(
    market_summary_df: pd.DataFrame,
    selected_market_algorithm: str,
    ticker_summary_df: pd.DataFrame,
    current_market_snapshot: pd.DataFrame,
    current_top_picks: pd.DataFrame,
    case_studies: pd.DataFrame,
) -> None:
    current_date = current_market_snapshot["Date"].iloc[0] if not current_market_snapshot.empty else ""
    consensus = current_market_snapshot["ConsensusBias"].iloc[0] if not current_market_snapshot.empty else "WAIT"
    selected_bias = ""
    if not current_market_snapshot.empty:
        selected_rows = current_market_snapshot[current_market_snapshot["Algorithm"] == selected_market_algorithm]
        if not selected_rows.empty:
            selected_bias = str(selected_rows.iloc[0]["CurrentBias"])

    print(f"SelectedMarketAlgorithm: {selected_market_algorithm}")
    print(f"CurrentDate: {current_date}")
    print(f"CurrentConsensusBias: {consensus}")
    if selected_bias:
        print(f"SelectedMarketCurrentBias: {selected_bias}")
    print()
    print("MarketAlgorithms")
    market_cols = [
        "Algorithm",
        "CurrentBias",
        "DEPLOYDays",
        "DEPLOYAvgRet10Pct",
        "DeployVsAll10Pct",
        "DEPLOYHit10Pct",
        "REDUCEDays",
        "REDUCEAvgRet10Pct",
        "DeployMinusReduce10Pct",
    ]
    print(market_summary_df[market_cols].to_string(index=False))
    print()
    print("TickerAlgorithms")
    ticker_cols = [
        "Algorithm",
        "SignalDays",
        "BasketAvgFwd10Pct",
        "BasketAvgExcess10Pct",
        "BasketHit10Pct",
        "PickAvgDrawdown10Pct",
    ]
    print(ticker_summary_df[ticker_cols].to_string(index=False))
    print()
    print("CurrentTopPicks")
    if current_top_picks.empty:
        print("No active picks today.")
    else:
        top_cols = [
            "Algorithm",
            "Rank",
            "Ticker",
            "Sector",
            "Score",
            "Rel20Pct",
            "Rel60Pct",
            "DistSMA20Pct",
            "SectorBreadth20Pct",
        ]
        print(current_top_picks[top_cols].to_string(index=False))
    print()
    print("CaseStudies")
    case_cols = [
        "Ticker",
        "Algorithm",
        "CurrentScore",
        "CurrentlyActive",
        "CurrentTopKRank",
        "CurrentRel20Pct",
        "CurrentRel60Pct",
        "CurrentDistSMA20Pct",
        "CurrentSectorBreadth20Pct",
        "HistoricalPickCount",
        "HistoricalAvgFwd10Pct",
        "HistoricalAvgExcess10Pct",
    ]
    print(case_studies[case_cols].to_string(index=False))


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    output_dir: Path,
    top_k: int,
    case_tickers: Sequence[str],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = build_feature_pack(history_dir, sector_map_path)
    market_signals = build_market_signal_table(features)
    market_summary_df = summarise_market_algorithms(features, market_signals)
    selected_market_algorithm = choose_market_algorithm(market_summary_df)
    market_summary_df["SelectedForTickerGate"] = market_summary_df["Algorithm"] == selected_market_algorithm
    selected_market_gate = market_signals[selected_market_algorithm]

    signal_bundles = build_ticker_signal_bundles(features, selected_market_gate)
    replay_df = _collect_replay_rows(features, selected_market_algorithm, selected_market_gate, signal_bundles, top_k)
    ticker_summary_df = summarise_ticker_algorithms(replay_df, top_k)
    current_market_snapshot = build_current_market_snapshot(features, market_signals)
    current_top_picks = build_current_top_picks(features, selected_market_algorithm, selected_market_gate, signal_bundles, top_k)
    case_studies = build_case_studies(
        features,
        selected_market_algorithm,
        selected_market_gate,
        signal_bundles,
        replay_df,
        case_tickers,
        top_k,
    )

    market_signals_out = market_signals.copy()
    market_signals_out.insert(0, "Date", [idx.date().isoformat() for idx in market_signals_out.index])
    market_signals_out.to_csv(output_dir / "market_bias_history.csv", index=False)
    market_summary_df.to_csv(output_dir / "market_algorithm_summary.csv", index=False)
    ticker_summary_df.to_csv(output_dir / "ticker_algorithm_summary.csv", index=False)
    replay_df.to_csv(output_dir / "ticker_replay_top_picks.csv", index=False)
    current_market_snapshot.to_csv(output_dir / "current_market_snapshot.csv", index=False)
    current_top_picks.to_csv(output_dir / "current_top_picks.csv", index=False)
    case_studies.to_csv(output_dir / "case_studies.csv", index=False)

    summary_payload = {
        "selected_market_algorithm": selected_market_algorithm,
        "selected_market_current_bias": selected_market_gate.iloc[-1],
        "current_consensus_bias": current_market_snapshot["ConsensusBias"].iloc[0] if not current_market_snapshot.empty else "WAIT",
        "market_algorithms_ranked": market_summary_df.to_dict(orient="records"),
        "ticker_algorithms_ranked": ticker_summary_df.to_dict(orient="records"),
        "case_tickers": list(case_tickers),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_stdout_summary(
        market_summary_df=market_summary_df,
        selected_market_algorithm=selected_market_algorithm,
        ticker_summary_df=ticker_summary_df,
        current_market_snapshot=current_market_snapshot,
        current_top_picks=current_top_picks,
        case_studies=case_studies,
    )

    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay deterministic market/ticker strategies on cached daily data.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV used for replay.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory to write evaluation reports.")
    parser.add_argument("--top-k", default=5, type=int, help="Top N tickers to evaluate per algorithm per day.")
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
        case_tickers=[_normalise_ticker(ticker) for ticker in args.case_tickers],
    )


if __name__ == "__main__":
    main()
