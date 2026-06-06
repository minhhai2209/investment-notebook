from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "out" / "universe.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis" / "positions"
DEFAULT_PRICE_MULTIPLIER = 1_000.0


def _normalise_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _load_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    if "Ticker" in frame.columns:
        frame = frame.copy()
        frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    return frame


def _load_optional_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _load_csv(path, required, label)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _first_ticker_row(frame: pd.DataFrame | None, ticker: str) -> Dict[str, Any] | None:
    if frame is None:
        return None
    scoped = frame.loc[frame["Ticker"].eq(ticker)]
    if scoped.empty:
        return None
    return scoped.iloc[0].to_dict()


def _ticker_rows(frame: pd.DataFrame | None, ticker: str) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    scoped = frame.loc[frame["Ticker"].eq(ticker)].copy()
    if scoped.empty:
        return []
    if "Horizon" in scoped.columns:
        scoped["Horizon"] = pd.to_numeric(scoped["Horizon"], errors="coerce")
        scoped = scoped.sort_values("Horizon")
    return [row.to_dict() for _, row in scoped.iterrows()]


def _money(price_diff: float, quantity: int, price_multiplier: float) -> int:
    return int(round(price_diff * float(quantity) * float(price_multiplier)))


def _ceil_to_lot(quantity: float, lot_size: int) -> int:
    if quantity <= 0:
        return 0
    lots = math.ceil(quantity / float(lot_size))
    return int(lots * int(lot_size))


def _pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return (numerator / denominator) * 100.0


def _average_price(avg_price: float, quantity: int, entry_price: float, additional_quantity: int) -> float:
    total_quantity = int(quantity) + int(additional_quantity)
    if total_quantity <= 0:
        return avg_price
    return ((avg_price * int(quantity)) + (entry_price * int(additional_quantity))) / float(total_quantity)


def _required_additional_quantity(
    *,
    avg_price: float,
    quantity: int,
    entry_price: float,
    target_exit_price: float,
    target_profit_vnd: int,
    price_multiplier: float,
    lot_size: int,
) -> int | None:
    required_price_units = float(target_profit_vnd) / float(price_multiplier)
    current_target_pnl_units = (target_exit_price - avg_price) * float(quantity)
    remaining_units = required_price_units - current_target_pnl_units
    if remaining_units <= 0:
        return 0
    edge_per_added_share = target_exit_price - entry_price
    if edge_per_added_share <= 0:
        return None
    return _ceil_to_lot(remaining_units / edge_per_added_share, lot_size)


def _average_down_scenarios(
    forecast_rows: Sequence[Mapping[str, Any]],
    *,
    avg_price: float,
    quantity: int,
    current_price: float,
    candidate_context: Mapping[str, Any],
    target_profit_vnd: int,
    price_multiplier: float,
    lot_size: int,
) -> List[Dict[str, Any]]:
    raw_entry_prices = [
        current_price,
        _safe_float(candidate_context.get("PreferredBuyZoneLow")),
        _safe_float(candidate_context.get("PreferredBuyZoneHigh")),
    ]
    entry_prices: List[float] = []
    for raw_price in raw_entry_prices:
        if raw_price is None or raw_price <= 0:
            continue
        if raw_price not in entry_prices:
            entry_prices.append(raw_price)

    scenarios: List[Dict[str, Any]] = []
    for entry_price in sorted(entry_prices):
        for forecast in forecast_rows:
            horizon = forecast.get("Horizon")
            for target_name, target_price in (
                ("ForecastClose", _safe_float(forecast.get("ForecastClose"))),
                ("ForecastHigh", _safe_float(forecast.get("ForecastHigh"))),
            ):
                if target_price is None:
                    continue
                required_quantity = _required_additional_quantity(
                    avg_price=avg_price,
                    quantity=quantity,
                    entry_price=entry_price,
                    target_exit_price=target_price,
                    target_profit_vnd=target_profit_vnd,
                    price_multiplier=price_multiplier,
                    lot_size=lot_size,
                )
                if required_quantity is None:
                    scenarios.append(
                        {
                            "EntryPrice": _round_or_none(entry_price, 3),
                            "Target": target_name,
                            "ForecastWindow": forecast.get("ForecastWindow"),
                            "Horizon": horizon,
                            "TargetExitPrice": _round_or_none(target_price, 3),
                            "RequiredAdditionalQuantity": None,
                            "Feasible": False,
                            "Reason": "target_exit_price_not_above_entry_price",
                        }
                    )
                    continue
                resulting_avg = _average_price(avg_price, quantity, entry_price, required_quantity)
                total_quantity = int(quantity) + int(required_quantity)
                scenarios.append(
                    {
                        "EntryPrice": _round_or_none(entry_price, 3),
                        "Target": target_name,
                        "ForecastWindow": forecast.get("ForecastWindow"),
                        "Horizon": horizon,
                        "TargetExitPrice": _round_or_none(target_price, 3),
                        "RequiredAdditionalQuantity": int(required_quantity),
                        "RequiredCapitalVND": _money(entry_price, required_quantity, price_multiplier),
                        "ResultingAvgPrice": _round_or_none(resulting_avg, 3),
                        "TotalQuantity": total_quantity,
                        "TargetPnLVND": _money(target_price - resulting_avg, total_quantity, price_multiplier),
                        "Feasible": True,
                    }
                )
    return scenarios


def _forecast_position_row(
    row: Mapping[str, Any],
    *,
    avg_price: float,
    current_price: float,
    quantity: int,
    price_multiplier: float,
) -> Dict[str, Any]:
    forecast_close = _safe_float(row.get("ForecastClose"))
    forecast_high = _safe_float(row.get("ForecastHigh"))
    forecast_low = _safe_float(row.get("ForecastLow"))
    close_mae = _safe_float(row.get("CloseMAEPct"))
    horizon = _safe_float(row.get("Horizon"))

    close_error_abs = None
    if forecast_close is not None and close_mae is not None:
        close_error_abs = forecast_close * close_mae / 100.0

    forecast_close_pnl = _money(forecast_close - avg_price, quantity, price_multiplier) if forecast_close is not None else None
    forecast_low_pnl = _money(forecast_low - avg_price, quantity, price_multiplier) if forecast_low is not None else None
    forecast_high_pnl = _money(forecast_high - avg_price, quantity, price_multiplier) if forecast_high is not None else None
    close_error_band_low = forecast_close - close_error_abs if forecast_close is not None and close_error_abs is not None else None
    close_error_band_high = forecast_close + close_error_abs if forecast_close is not None and close_error_abs is not None else None

    return {
        "Horizon": int(horizon) if horizon is not None else None,
        "ForecastWindow": row.get("ForecastWindow"),
        "Model": row.get("Model"),
        "ModelFamily": row.get("ModelFamily"),
        "ModelClass": row.get("ModelClass"),
        "ForecastOpen": _round_or_none(row.get("ForecastOpen"), 3),
        "ForecastHigh": _round_or_none(forecast_high, 3),
        "ForecastLow": _round_or_none(forecast_low, 3),
        "ForecastClose": _round_or_none(forecast_close, 3),
        "ForecastCloseRetFromCurrentPct": _round_or_none(_pct((forecast_close or current_price) - current_price, current_price), 2)
        if forecast_close is not None
        else None,
        "CloseMAEPct": _round_or_none(close_mae, 2),
        "CloseDirHitPct": _round_or_none(row.get("CloseDirHitPct"), 2),
        "ForecastClosePnLVND": forecast_close_pnl,
        "ForecastLowPnLVND": forecast_low_pnl,
        "ForecastHighPnLVND": forecast_high_pnl,
        "CloseErrorBandLow": _round_or_none(close_error_band_low, 3),
        "CloseErrorBandHigh": _round_or_none(close_error_band_high, 3),
        "CloseErrorBandLowPnLVND": _money(close_error_band_low - avg_price, quantity, price_multiplier)
        if close_error_band_low is not None
        else None,
        "CloseErrorBandHighPnLVND": _money(close_error_band_high - avg_price, quantity, price_multiplier)
        if close_error_band_high is not None
        else None,
    }


def _timing_summary(row: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if row is None:
        return None
    return {
        "ForecastWindow": row.get("ForecastWindow"),
        "Model": row.get("Model"),
        "ModelFamily": row.get("ModelFamily"),
        "ModelClass": row.get("ModelClass"),
        "PredPeakRetPct": _round_or_none(row.get("PredPeakRetPct"), 2),
        "PredPeakDay": _round_or_none(row.get("PredPeakDay"), 1),
        "PredDrawdownPct": _round_or_none(row.get("PredDrawdownPct"), 2),
        "PredCloseRetPct": _round_or_none(row.get("PredCloseRetPct"), 2),
        "PredNetEdgePct": _round_or_none(row.get("PredNetEdgePct"), 2),
        "EvalRows": _round_or_none(row.get("EvalRows"), 0),
        "PeakRetMAEPct": _round_or_none(row.get("PeakRetMAEPct"), 2),
        "CloseMAEPct": _round_or_none(row.get("CloseMAEPct"), 2),
        "TradeScoreHitPct": _round_or_none(row.get("TradeScoreHitPct"), 2),
    }


def _best_timing_row(frame: pd.DataFrame | None, ticker: str) -> Dict[str, Any] | None:
    rows = _ticker_rows(frame, ticker)
    if not rows:
        return None
    scoped = pd.DataFrame(rows)
    scoped["PredNetEdgePct"] = pd.to_numeric(scoped["PredNetEdgePct"], errors="coerce")
    scoped = scoped.dropna(subset=["PredNetEdgePct"])
    if scoped.empty:
        return None
    return scoped.sort_values(["PredNetEdgePct", "Horizon"], ascending=[False, True]).iloc[0].to_dict()


def build_position_ml_review(
    *,
    ticker: str,
    quantity: int,
    avg_price: float,
    current_price: float | None,
    universe_csv: Path,
    analysis_dir: Path,
    output_dir: Path,
    price_multiplier: float = DEFAULT_PRICE_MULTIPLIER,
    target_profit_vnd: int = 0,
) -> Dict[str, Any]:
    normalized = _normalise_ticker(ticker)
    if not normalized:
        raise ValueError("ticker is required")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if avg_price <= 0:
        raise ValueError("avg_price must be positive")

    universe_df = _load_csv(universe_csv, ["Ticker", "Last"], "Universe snapshot")
    ohlc_multi_df = _load_csv(
        analysis_dir / "ml_ohlc_multi_session.csv",
        [
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "ForecastOpen",
            "ForecastHigh",
            "ForecastLow",
            "ForecastClose",
            "Model",
            "CloseMAEPct",
            "CloseDirHitPct",
        ],
        "Multi-session OHLC",
    )
    timing_df = _load_optional_csv(
        analysis_dir / "ml_single_name_timing.csv",
        [
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "Model",
            "PredPeakRetPct",
            "PredPeakDay",
            "PredDrawdownPct",
            "PredCloseRetPct",
            "PredNetEdgePct",
            "EvalRows",
            "PeakRetMAEPct",
            "CloseMAEPct",
            "TradeScoreHitPct",
        ],
        "Single-name timing",
    )
    candidate_df = _load_optional_csv(
        analysis_dir / "candidates" / "candidate_watchlist_full.csv",
        [
            "Ticker",
            "Decision",
            "ModelDecisionBasis",
            "PreferredBuyZoneLow",
            "PreferredBuyZoneHigh",
            "ZoneStatus",
            "NoChaseAbove",
            "InvalidationBelow",
            "OHLCMultiSessionSummary",
        ],
        "Candidate watchlist full",
    )

    universe_row = _first_ticker_row(universe_df, normalized)
    if universe_row is None:
        raise ValueError(f"{normalized} is not present in {universe_csv}")
    resolved_current = current_price if current_price is not None else _safe_float(universe_row.get("Last"))
    if resolved_current is None or resolved_current <= 0:
        raise ValueError("current_price must be positive or resolvable from universe snapshot")

    current_pnl = _money(resolved_current - avg_price, quantity, price_multiplier)
    lot_size = int(_safe_float(universe_row.get("LotSize")) or 100)
    forecast_rows = [
        _forecast_position_row(
            row,
            avg_price=avg_price,
            current_price=resolved_current,
            quantity=quantity,
            price_multiplier=price_multiplier,
        )
        for row in _ticker_rows(ohlc_multi_df, normalized)
    ]
    candidate_row = _first_ticker_row(candidate_df, normalized)

    candidate_context = {
        "Decision": (candidate_row or {}).get("Decision"),
        "ModelDecisionBasis": (candidate_row or {}).get("ModelDecisionBasis"),
        "PreferredBuyZoneLow": _round_or_none((candidate_row or {}).get("PreferredBuyZoneLow"), 3),
        "PreferredBuyZoneHigh": _round_or_none((candidate_row or {}).get("PreferredBuyZoneHigh"), 3),
        "ZoneStatus": (candidate_row or {}).get("ZoneStatus"),
        "NoChaseAbove": _round_or_none((candidate_row or {}).get("NoChaseAbove"), 3),
        "InvalidationBelow": _round_or_none((candidate_row or {}).get("InvalidationBelow"), 3),
    }

    report = {
        "Ticker": normalized,
        "Quantity": int(quantity),
        "AvgPrice": float(avg_price),
        "CurrentPrice": float(resolved_current),
        "LotSize": lot_size,
        "PriceMultiplier": float(price_multiplier),
        "TargetProfitVND": int(target_profit_vnd),
        "CurrentPnLVND": current_pnl,
        "CurrentPnLPct": _round_or_none(_pct(resolved_current - avg_price, avg_price), 2),
        "CandidateContext": candidate_context,
        "BestTimingModel": _timing_summary(_best_timing_row(timing_df, normalized)),
        "OHLCForecasts": forecast_rows,
        "AverageDownScenarios": _average_down_scenarios(
            forecast_rows,
            avg_price=avg_price,
            quantity=quantity,
            current_price=resolved_current,
            candidate_context=candidate_context,
            target_profit_vnd=int(target_profit_vnd),
            price_multiplier=price_multiplier,
            lot_size=lot_size,
        ),
        "Policy": "ml_only_no_manual_trim_or_stop_rules",
        "Interpretation": (
            "This report gives model-implied position scenarios only. "
            "It does not invent fixed sell quantities, stop levels, or averaging rules."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized.lower()}_position_ml_review.json"
    output_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    report["OutputJSON"] = str(output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an ML-only review for an existing stock position.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--quantity", type=int, required=True)
    parser.add_argument("--avg-price", type=float, required=True)
    parser.add_argument("--current-price", type=float, default=None)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-multiplier", type=float, default=DEFAULT_PRICE_MULTIPLIER)
    parser.add_argument("--target-profit-vnd", type=int, default=0)
    args = parser.parse_args(argv)

    report = build_position_ml_review(
        ticker=args.ticker,
        quantity=args.quantity,
        avg_price=args.avg_price,
        current_price=args.current_price,
        universe_csv=args.universe_csv,
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        price_multiplier=args.price_multiplier,
        target_profit_vnd=args.target_profit_vnd,
    )
    print(json.dumps(_json_safe(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
