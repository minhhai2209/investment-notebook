from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.data_fetching.fetch_ticker_data import ensure_ohlc_cache


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class StrategyConfig:
    family: str
    label: str
    params: dict[str, float | int | bool | None]
    exit_ema_span: int
    max_hold_days: int
    target_pct: float | None
    stop_pct: float = -0.10
    min_hold_days: int = 3


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _require_columns(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _load_daily_ohlcv(ticker: str, history_dir: Path, history_calendar_days: int) -> pd.DataFrame:
    normalized = _normalise_ticker(ticker)
    ensure_ohlc_cache(normalized, outdir=str(history_dir), min_days=history_calendar_days, resolution="D")
    path = history_dir / f"{normalized}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached OHLCV for {normalized}: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, path, REQUIRED_OHLCV_COLUMNS)
    if "date_vn" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["date_vn"], errors="coerce")
    elif "t" in frame.columns:
        frame["Date"] = pd.to_datetime(pd.to_numeric(frame["t"], errors="coerce"), unit="s", errors="coerce")
    else:
        raise ValueError(f"Missing date column in {path}")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for column in REQUIRED_OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(REQUIRED_OHLCV_COLUMNS)).reset_index(drop=True)
    return frame


def _compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    high = out["high"]
    volume = out["volume"]
    out["ret1"] = close.pct_change()
    out["ret3"] = close.pct_change(3)
    out["ret5"] = close.pct_change(5)
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["ema5"] = close.ewm(span=5, adjust=False).mean()
    out["ema7"] = close.ewm(span=7, adjust=False).mean()
    out["ema10"] = close.ewm(span=10, adjust=False).mean()
    out["ema15"] = close.ewm(span=15, adjust=False).mean()
    out["vol_ratio20"] = volume / volume.rolling(20).mean()
    out["roll_high_5"] = high.shift(1).rolling(5).max()
    out["roll_high_10"] = high.shift(1).rolling(10).max()
    out["roll_high_15"] = high.shift(1).rolling(15).max()
    out["dd60"] = close / close.rolling(60).max() - 1.0
    out["dd120"] = close / close.rolling(120).max() - 1.0
    out["dist_sma20"] = close / out["sma20"] - 1.0
    out["dist_sma50"] = close / out["sma50"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    std20 = close.rolling(20).std()
    out["bb_lower_20_2"] = out["sma20"] - 2.0 * std20
    out["bb_upper_20_2"] = out["sma20"] + 2.0 * std20
    return out


def _strategy_catalog() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []

    for dd60_max in (-0.12, -0.18):
        for rsi_max in (35, 40):
            for require_prev_lower_bb in (True,):
                for ema_span in (7, 10):
                    for max_hold in (15,):
                        for target in (None, 0.10, 0.15):
                            params = {
                                "dd60_max": dd60_max,
                                "rsi_max": rsi_max,
                                "require_prev_lower_bb": require_prev_lower_bb,
                            }
                            label = (
                                f"washout_reclaim/dd60<={dd60_max:.0%}/"
                                f"rsi<={rsi_max}/prev_bb={'y' if require_prev_lower_bb else 'n'}"
                            )
                            configs.append(
                                StrategyConfig(
                                    family="washout_reclaim",
                                    label=label,
                                    params=params,
                                    exit_ema_span=ema_span,
                                    max_hold_days=max_hold,
                                    target_pct=target,
                                )
                            )

    for dist20_min in (-0.10, -0.07):
        for dist20_max in (0.0,):
            for rsi_min, rsi_max in ((40, 58), (45, 62)):
                for require_sma200 in (True,):
                    for ema_span in (10, 15):
                        for max_hold in (15, 25):
                            for target in (None, 0.08, 0.12):
                                params = {
                                    "dist20_min": dist20_min,
                                    "dist20_max": dist20_max,
                                    "rsi_min": rsi_min,
                                    "rsi_max": rsi_max,
                                    "require_sma200": require_sma200,
                                }
                                label = (
                                    f"trend_pullback/dist20[{dist20_min:.0%},{dist20_max:.0%}]/"
                                    f"rsi[{rsi_min},{rsi_max}]/sma200={'y' if require_sma200 else 'n'}"
                                )
                                configs.append(
                                    StrategyConfig(
                                        family="trend_pullback",
                                        label=label,
                                        params=params,
                                        exit_ema_span=ema_span,
                                        max_hold_days=max_hold,
                                        target_pct=target,
                                    )
                                )

    for dd60_max in (-0.08, -0.12):
        for lookback in (5, 10):
            for min_ret1 in (0.03, 0.045):
                for min_vol_ratio in (1.0, 1.3):
                    for ema_span in (5,):
                        for max_hold in (10, 15):
                            for target in (0.10, 0.15):
                                params = {
                                    "dd60_max": dd60_max,
                                    "lookback": lookback,
                                    "min_ret1": min_ret1,
                                    "min_vol_ratio": min_vol_ratio,
                                }
                                label = (
                                    f"breakout_followthrough/dd60<={dd60_max:.0%}/"
                                    f"look{lookback}/ret1>={min_ret1:.1%}/vol>={min_vol_ratio:.1f}"
                                )
                                configs.append(
                                    StrategyConfig(
                                        family="breakout_followthrough",
                                        label=label,
                                        params=params,
                                        exit_ema_span=ema_span,
                                        max_hold_days=max_hold,
                                        target_pct=target,
                                    )
                                )

    for lookback in (5, 10):
        for min_ret3 in (0.01, 0.02):
            for min_vol_ratio in (0.8, 1.0):
                for max_dist20 in (0.04,):
                    for ema_span in (10, 15):
                        for max_hold in (15, 25):
                            for target in (None, 0.10):
                                params = {
                                    "lookback": lookback,
                                    "min_ret3": min_ret3,
                                    "min_vol_ratio": min_vol_ratio,
                                    "max_dist20": max_dist20,
                                }
                                label = (
                                    f"trend_reacceleration/look{lookback}/ret3>={min_ret3:.0%}/"
                                    f"vol>={min_vol_ratio:.1f}/dist20<={max_dist20:.0%}"
                                )
                                configs.append(
                                    StrategyConfig(
                                        family="trend_reacceleration",
                                        label=label,
                                        params=params,
                                        exit_ema_span=ema_span,
                                        max_hold_days=max_hold,
                                        target_pct=target,
                                    )
                                )

    return configs


def _signal_for_config(frame: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    if config.family == "washout_reclaim":
        signal = (
            (frame["dd60"] <= float(config.params["dd60_max"]))
            & (frame["rsi14"] <= float(config.params["rsi_max"]))
            & (frame["ret1"] > 0.0)
        )
        if bool(config.params["require_prev_lower_bb"]):
            signal &= frame["close"].shift(1) < frame["bb_lower_20_2"].shift(1)
        return signal.fillna(False)

    if config.family == "trend_pullback":
        signal = (
            (frame["dist_sma20"] >= float(config.params["dist20_min"]))
            & (frame["dist_sma20"] <= float(config.params["dist20_max"]))
            & (frame["rsi14"] >= float(config.params["rsi_min"]))
            & (frame["rsi14"] <= float(config.params["rsi_max"]))
            & (frame["ret1"] > 0.0)
            & (frame["close"] > frame["ema10"])
        )
        if bool(config.params["require_sma200"]):
            signal &= frame["close"] >= frame["sma200"]
        return signal.fillna(False)

    if config.family == "breakout_followthrough":
        lookback = int(config.params["lookback"])
        breakout_col = f"roll_high_{lookback}"
        signal = (
            (frame["dd60"] <= float(config.params["dd60_max"]))
            & (frame["close"] > frame[breakout_col])
            & (frame["ret1"] >= float(config.params["min_ret1"]))
            & (frame["vol_ratio20"] >= float(config.params["min_vol_ratio"]))
        )
        return signal.fillna(False)

    if config.family == "trend_reacceleration":
        lookback = int(config.params["lookback"])
        breakout_col = f"roll_high_{lookback}"
        signal = (
            (frame["close"] >= frame["sma50"])
            & (frame["sma50"] >= frame["sma200"])
            & (frame["close"] > frame[breakout_col])
            & (frame["ret3"] >= float(config.params["min_ret3"]))
            & (frame["vol_ratio20"] >= float(config.params["min_vol_ratio"]))
            & (frame["dist_sma20"] <= float(config.params["max_dist20"]))
        )
        return signal.fillna(False)

    raise ValueError(f"Unsupported strategy family: {config.family}")


def _exit_ema_column(span: int) -> str:
    mapping = {5: "ema5", 7: "ema7", 10: "ema10", 15: "ema15"}
    if span not in mapping:
        raise ValueError(f"Unsupported exit EMA span: {span}")
    return mapping[span]


def _backtest_signals(
    frame: pd.DataFrame,
    signal: pd.Series,
    config: StrategyConfig,
    start_idx: int,
    end_idx: int | None = None,
) -> pd.DataFrame:
    stop_pct = float(config.stop_pct)
    target_pct = float(config.target_pct) if config.target_pct is not None else None
    max_hold = int(config.max_hold_days)
    min_hold = int(config.min_hold_days)
    exit_ema = frame[_exit_ema_column(config.exit_ema_span)]
    upper = len(frame) if end_idx is None else min(end_idx, len(frame))
    index = max(start_idx, 21)
    trades: list[dict[str, object]] = []

    while index < upper - 1:
        if not bool(signal.iloc[index]):
            index += 1
            continue
        entry_idx = index + 1
        entry_price = float(frame.at[entry_idx, "open"])
        exit_idx: int | None = None
        exit_price: float | None = None
        exit_reason: str | None = None
        probe_idx = entry_idx + min_hold - 1

        while probe_idx < min(upper, entry_idx + max_hold):
            if target_pct is not None and float(frame.at[probe_idx, "high"]) >= entry_price * (1.0 + target_pct):
                exit_idx = probe_idx
                exit_price = entry_price * (1.0 + target_pct)
                exit_reason = "target"
                break
            if float(frame.at[probe_idx, "low"]) <= entry_price * (1.0 + stop_pct):
                exit_idx = probe_idx
                exit_price = entry_price * (1.0 + stop_pct)
                exit_reason = "stop"
                break
            if float(frame.at[probe_idx, "close"]) < float(exit_ema.iloc[probe_idx]):
                exit_idx = probe_idx
                exit_price = float(frame.at[probe_idx, "close"])
                exit_reason = f"ema{config.exit_ema_span}"
                break
            probe_idx += 1

        if exit_idx is None or exit_price is None or exit_reason is None:
            exit_idx = min(upper - 1, entry_idx + max_hold - 1)
            exit_price = float(frame.at[exit_idx, "close"])
            exit_reason = "time"

        slice_frame = frame.iloc[entry_idx : exit_idx + 1]
        trades.append(
            {
                "SignalDate": frame.at[index, "Date"].date().isoformat(),
                "EntryDate": frame.at[entry_idx, "Date"].date().isoformat(),
                "ExitDate": frame.at[exit_idx, "Date"].date().isoformat(),
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "ReturnPct": (exit_price / entry_price - 1.0) * 100.0,
                "WorstDrawdownPct": (float(slice_frame["low"].min()) / entry_price - 1.0) * 100.0,
                "BestRunupPct": (float(slice_frame["high"].max()) / entry_price - 1.0) * 100.0,
                "HoldDays": int(exit_idx - entry_idx + 1),
                "ExitReason": exit_reason,
            }
        )
        index = exit_idx + 1

    return pd.DataFrame(trades)


def summarise_trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "Trades": 0.0,
            "WinRatePct": 0.0,
            "AvgReturnPct": np.nan,
            "MedianReturnPct": np.nan,
            "ProfitFactor": np.nan,
            "AvgHoldDays": np.nan,
            "WorstDrawdownPct": np.nan,
        }
    positive = trades.loc[trades["ReturnPct"] > 0.0, "ReturnPct"].sum()
    negative = -trades.loc[trades["ReturnPct"] <= 0.0, "ReturnPct"].sum()
    profit_factor = positive / negative if negative > 0.0 else np.nan
    return {
        "Trades": float(trades.shape[0]),
        "WinRatePct": float((trades["ReturnPct"] > 0.0).mean() * 100.0),
        "AvgReturnPct": float(trades["ReturnPct"].mean()),
        "MedianReturnPct": float(trades["ReturnPct"].median()),
        "ProfitFactor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "AvgHoldDays": float(trades["HoldDays"].mean()),
        "WorstDrawdownPct": float(trades["WorstDrawdownPct"].min()),
    }


def score_trade_metrics(metrics: dict[str, float]) -> float:
    trades = float(metrics.get("Trades", 0.0) or 0.0)
    if trades < 3.0:
        return -999.0
    avg_return = float(metrics.get("AvgReturnPct", np.nan))
    median_return = float(metrics.get("MedianReturnPct", np.nan))
    win_rate = float(metrics.get("WinRatePct", 0.0))
    profit_factor = float(metrics.get("ProfitFactor", np.nan))
    avg_hold = float(metrics.get("AvgHoldDays", np.nan))
    worst_drawdown = float(metrics.get("WorstDrawdownPct", np.nan))

    if np.isnan(avg_return) or np.isnan(median_return) or np.isnan(avg_hold) or np.isnan(worst_drawdown):
        return -999.0
    pf_term = min(profit_factor, 3.0) if not np.isnan(profit_factor) else 0.0
    return (
        avg_return
        + 0.35 * median_return
        + 0.12 * win_rate
        + 2.5 * pf_term
        - 0.12 * avg_hold
        + 0.25 * worst_drawdown
        + min(trades, 12.0)
    )


def select_best_playbook(config_results: pd.DataFrame) -> pd.DataFrame:
    if config_results.empty:
        return config_results.copy()
    result = config_results.sort_values(
        by=["Ticker", "RobustScore", "TestScore", "TrainScore", "AllScore"],
        ascending=[True, False, False, False, False],
    ).groupby("Ticker", as_index=False).head(1)
    return result.reset_index(drop=True)


def _robust_score(train_score: float, test_score: float, test_trades: float) -> float:
    if test_trades >= 3.0:
        return 0.45 * train_score + 0.55 * test_score
    if test_trades == 2.0:
        return 0.60 * train_score + 0.40 * test_score - 3.0
    if test_trades == 1.0:
        return 0.75 * train_score + 0.25 * test_score - 6.0
    return train_score - 12.0


def _current_signal_state(frame: pd.DataFrame, signal: pd.Series) -> tuple[bool, str | None]:
    if frame.empty:
        return False, None
    latest_idx = len(frame) - 1
    if latest_idx < 0:
        return False, None
    return bool(signal.iloc[latest_idx]), frame.at[latest_idx, "Date"].date().isoformat()


def build_playbook_report_for_ticker(
    ticker: str,
    history_dir: Path,
    history_calendar_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _compute_indicators(_load_daily_ohlcv(ticker, history_dir, history_calendar_days))
    split_idx = int(len(frame) * 0.70)
    configs = _strategy_catalog()
    result_rows: list[dict[str, object]] = []
    all_trades_rows: list[dict[str, object]] = []

    for config in configs:
        signal = _signal_for_config(frame, config)
        train_trades = _backtest_signals(frame, signal, config, start_idx=21, end_idx=split_idx)
        test_trades = _backtest_signals(frame, signal, config, start_idx=split_idx, end_idx=len(frame))
        all_trades = _backtest_signals(frame, signal, config, start_idx=21, end_idx=len(frame))

        train_metrics = summarise_trade_metrics(train_trades)
        test_metrics = summarise_trade_metrics(test_trades)
        all_metrics = summarise_trade_metrics(all_trades)
        train_score = score_trade_metrics(train_metrics)
        test_score = score_trade_metrics(test_metrics)
        all_score = score_trade_metrics(all_metrics)
        robust_score = _robust_score(train_score, test_score, test_metrics["Trades"])
        latest_signal, latest_signal_date = _current_signal_state(frame, signal)

        result_rows.append(
            {
                "Ticker": _normalise_ticker(ticker),
                "StrategyFamily": config.family,
                "StrategyLabel": config.label,
                "ParameterSummary": json.dumps(config.params, ensure_ascii=False, sort_keys=True),
                "ExitEMASpan": config.exit_ema_span,
                "MaxHoldDays": config.max_hold_days,
                "TargetPct": config.target_pct * 100.0 if config.target_pct is not None else np.nan,
                "StopPct": config.stop_pct * 100.0,
                "LatestSignal": latest_signal,
                "LatestSignalDate": latest_signal_date,
                "TrainScore": train_score,
                "TestScore": test_score,
                "AllScore": all_score,
                "RobustScore": robust_score,
                **{f"Train{key}": value for key, value in train_metrics.items()},
                **{f"Test{key}": value for key, value in test_metrics.items()},
                **{f"All{key}": value for key, value in all_metrics.items()},
            }
        )

        if not all_trades.empty:
            trades_copy = all_trades.copy()
            trades_copy.insert(0, "Ticker", _normalise_ticker(ticker))
            trades_copy.insert(1, "StrategyFamily", config.family)
            trades_copy.insert(2, "StrategyLabel", config.label)
            trades_copy.insert(3, "ParameterSummary", json.dumps(config.params, ensure_ascii=False, sort_keys=True))
            all_trades_rows.append(trades_copy)

    return pd.DataFrame(result_rows), pd.concat(all_trades_rows, ignore_index=True) if all_trades_rows else pd.DataFrame()


def build_human_summary(best_configs: pd.DataFrame) -> str:
    lines = ["# Ticker Playbook Summary", ""]
    for row in best_configs.to_dict("records"):
        ticker = str(row["Ticker"])
        lines.append(f"## {ticker}")
        lines.append(f"- Playbook: `{row['StrategyFamily']}`")
        lines.append(f"- Rule: `{row['StrategyLabel']}`")
        lines.append(
            "- Train/Test/All trades: "
            f"{int(row['TrainTrades'])}/{int(row['TestTrades'])}/{int(row['AllTrades'])}"
        )
        lines.append(
            "- Test profile: "
            f"win {row['TestWinRatePct']:.1f}%, avg {row['TestAvgReturnPct']:.2f}%, "
            f"median {row['TestMedianReturnPct']:.2f}%, hold {row['TestAvgHoldDays']:.1f} days, "
            f"worst dd {row['TestWorstDrawdownPct']:.2f}%"
        )
        lines.append(
            "- All-history profile: "
            f"win {row['AllWinRatePct']:.1f}%, avg {row['AllAvgReturnPct']:.2f}%, "
            f"median {row['AllMedianReturnPct']:.2f}%, hold {row['AllAvgHoldDays']:.1f} days, "
            f"worst dd {row['AllWorstDrawdownPct']:.2f}%"
        )
        lines.append(f"- Latest signal on snapshot: `{bool(row['LatestSignal'])}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [_normalise_ticker(part) for part in raw.split(",") if str(part).strip()]


def run_report(
    tickers: Sequence[str],
    history_dir: Path,
    output_dir: Path,
    history_calendar_days: int = 1600,
) -> dict[str, Path]:
    normalized_tickers = []
    for ticker in tickers:
        normalized = _normalise_ticker(ticker)
        if normalized and normalized not in normalized_tickers:
            normalized_tickers.append(normalized)
    if not normalized_tickers:
        raise ValueError("Expected at least one ticker")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_config_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for ticker in normalized_tickers:
        config_df, trades_df = build_playbook_report_for_ticker(
            ticker=ticker,
            history_dir=history_dir,
            history_calendar_days=history_calendar_days,
        )
        per_config_frames.append(config_df)
        if not trades_df.empty:
            trade_frames.append(trades_df)

    all_configs = pd.concat(per_config_frames, ignore_index=True)
    best_configs = select_best_playbook(all_configs)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_markdown = build_human_summary(best_configs)

    all_configs_path = output_dir / "ticker_playbook_all_configs.csv"
    best_configs_path = output_dir / "ticker_playbook_best_configs.csv"
    trades_path = output_dir / "ticker_playbook_all_trades.csv"
    summary_path = output_dir / "ticker_playbook_summary.md"

    all_configs.to_csv(all_configs_path, index=False)
    best_configs.to_csv(best_configs_path, index=False)
    all_trades.to_csv(trades_path, index=False)
    summary_path.write_text(summary_markdown, encoding="utf-8")

    return {
        "AllConfigs": all_configs_path,
        "BestConfigs": best_configs_path,
        "AllTrades": trades_path,
        "Summary": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest simple technical playbooks per ticker.")
    parser.add_argument("--tickers", help="Comma-separated ticker list")
    parser.add_argument("--history-dir", type=Path, default=Path("out/data_hose_ml_4y"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/analysis/ticker_playbooks"))
    parser.add_argument("--history-calendar-days", type=int, default=1600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tickers = _parse_tickers(args.tickers)
    outputs = run_report(
        tickers=tickers,
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        history_calendar_days=args.history_calendar_days,
    )
    print(f"Wrote {outputs['AllConfigs']}")
    print(f"Wrote {outputs['BestConfigs']}")
    print(f"Wrote {outputs['AllTrades']}")
    print(f"Wrote {outputs['Summary']}")


if __name__ == "__main__":
    main()
