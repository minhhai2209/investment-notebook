from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from scripts.analysis.ticker_color_overlay import (
    DEFAULT_TICKER_COLOR_DIR,
    extract_ticker_overlay,
    load_optional_overlay,
    summarise_ticker_overlay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "out" / "universe.csv"
DEFAULT_MARKET_SUMMARY_JSON = REPO_ROOT / "out" / "market_summary.json"
DEFAULT_SECTOR_SUMMARY_CSV = REPO_ROOT / "out" / "sector_summary.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_RESEARCH_DIR = REPO_ROOT / "research"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis" / "candidates"
DEFAULT_BUDGET_VND = 5_000_000_000

CORE_MODE = "core"
FULL_MODE = "full"
AUTO_MODE = "auto"


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _normalise_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    if "Ticker" in frame.columns:
        frame = frame.copy()
        frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    return frame


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(research_dir: Path, ticker: str) -> Dict[str, Any] | None:
    state_path = research_dir / "tickers" / ticker / "state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_mode(analysis_dir: Path, research_dir: Path, requested_mode: str) -> str:
    if requested_mode == AUTO_MODE:
        if _full_inputs_available(analysis_dir, research_dir):
            return FULL_MODE
        return CORE_MODE
    if requested_mode == FULL_MODE and not _full_inputs_available(analysis_dir, research_dir):
        missing = ", ".join(_missing_full_inputs(analysis_dir, research_dir))
        raise FileNotFoundError(f"Full-mode candidate watchlist requires missing inputs: {missing}")
    return requested_mode


def _missing_full_inputs(analysis_dir: Path, research_dir: Path) -> List[str]:
    expected = [
        analysis_dir / "ml_ohlc_next_session.csv",
        analysis_dir / "ml_single_name_timing.csv",
        analysis_dir / "ml_entry_ladder_eval.csv",
        analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv",
        research_dir / "manifest.json",
    ]
    return [_display_path(path) for path in expected if not path.exists()]


def _full_inputs_available(analysis_dir: Path, research_dir: Path) -> bool:
    return not _missing_full_inputs(analysis_dir, research_dir)


def _extract_timing_rows(frame: pd.DataFrame | None, ticker: str) -> tuple[Dict[str, Any] | None, float | None]:
    if frame is None:
        return None, None
    scoped = frame.loc[frame["Ticker"].eq(ticker)].copy()
    if scoped.empty:
        return None, None
    scoped["PredNetEdgePct"] = pd.to_numeric(scoped["PredNetEdgePct"], errors="coerce")
    scoped = scoped.dropna(subset=["PredNetEdgePct"])
    if scoped.empty:
        return None, None
    best_row = scoped.sort_values(["PredNetEdgePct", "Horizon"], ascending=[False, True]).iloc[0]
    t10_rows = scoped.loc[pd.to_numeric(scoped["Horizon"], errors="coerce").eq(10)]
    t10_edge = None
    if not t10_rows.empty:
        t10_edge = _safe_float(t10_rows.iloc[0]["PredNetEdgePct"])
    return best_row.to_dict(), t10_edge


def _extract_cycle_row(frame: pd.DataFrame | None, ticker: str) -> Dict[str, Any] | None:
    if frame is None:
        return None
    scoped = frame.loc[frame["Ticker"].eq(ticker)].copy()
    if scoped.empty:
        return None
    if "PredPeakRetPct" in scoped.columns:
        scoped["PredPeakRetPct"] = pd.to_numeric(scoped["PredPeakRetPct"], errors="coerce")
        scoped = scoped.dropna(subset=["PredPeakRetPct"])
        if scoped.empty:
            return None
        scoped = scoped.sort_values(["PredPeakRetPct", "ForecastWindow"], ascending=[False, True])
    return scoped.iloc[0].to_dict()


def _extract_entry_row(frame: pd.DataFrame | None, ticker: str) -> Dict[str, Any] | None:
    if frame is None:
        return None
    scoped = frame.loc[frame["Ticker"].eq(ticker)].copy()
    if scoped.empty:
        return None
    if "EntryAnchorCategory" in scoped.columns:
        historical = scoped.loc[
            scoped["EntryAnchorCategory"].astype(str).str.lower().isin({"historical", "mixed"})
        ].copy()
        if not historical.empty:
            scoped = historical
    scoped["EntryScoreRank"] = pd.to_numeric(scoped["EntryScoreRank"], errors="coerce")
    scoped = scoped.dropna(subset=["EntryScoreRank"])
    if scoped.empty:
        return None
    row = scoped.sort_values(["EntryScoreRank", "LimitPrice"], ascending=[True, True]).iloc[0]
    return row.to_dict()


def _preferred_zone(
    snapshot_row: pd.Series,
    *,
    state: Mapping[str, Any] | None,
    ladder_row: Mapping[str, Any] | None,
) -> tuple[float | None, float | None, str]:
    state_low = _safe_float(state.get("PreferredBuyZoneLow")) if state else None
    state_high = _safe_float(state.get("PreferredBuyZoneHigh")) if state else None
    if state_low is not None and state_high is not None and state_low > 0 and state_high > 0:
        low = min(state_low, state_high)
        high = max(state_low, state_high)
        return low, high, "state"

    if ladder_row is not None:
        limit_price = _safe_float(ladder_row.get("LimitPrice"))
        if limit_price is not None and limit_price > 0:
            tick = _safe_float(snapshot_row.get("TickSize")) or 0.0
            return max(limit_price - (2.0 * tick), 0.0), limit_price, "ladder"

    grid_prices = [
        _safe_float(snapshot_row.get("GridBelow_T1")),
        _safe_float(snapshot_row.get("GridBelow_T2")),
        _safe_float(snapshot_row.get("GridBelow_T3")),
    ]
    grid_prices = [value for value in grid_prices if value is not None and value > 0]
    if grid_prices:
        return min(grid_prices), max(grid_prices), "grid"
    return None, None, "unknown"


def _zone_status(last_price: float, zone_low: float | None, zone_high: float | None) -> tuple[str, float | None]:
    if zone_low is None or zone_high is None:
        return "unknown", None
    if zone_low <= last_price <= zone_high:
        return "inside", 0.0
    if last_price > zone_high:
        return "above", ((last_price / zone_high) - 1.0) * 100.0
    return "below", ((zone_low / last_price) - 1.0) * 100.0


def _reference_buy_price(
    *,
    decision: str,
    last_price: float,
    zone_low: float | None,
    zone_high: float | None,
    ladder_row: Mapping[str, Any] | None,
) -> float:
    if decision == "mua_ngay":
        if zone_high is not None and zone_low is not None and zone_low <= last_price <= zone_high:
            return last_price
    if zone_low is not None:
        return zone_low
    if ladder_row is not None:
        ladder_price = _safe_float(ladder_row.get("LimitPrice"))
        if ladder_price is not None and ladder_price > 0:
            return ladder_price
    return last_price


def _quantity_for_budget(budget_vnd: int, price_kvnd: float, lot_size: int) -> int:
    if price_kvnd <= 0 or lot_size <= 0:
        return 0
    lots = int(budget_vnd // (price_kvnd * 1_000 * lot_size))
    return lots * lot_size


def _budget_pct_adv(last_price: float, adtv_shares: float, budget_vnd: int) -> float | None:
    if last_price <= 0 or adtv_shares <= 0:
        return None
    adtv_value = last_price * 1_000 * adtv_shares
    return (budget_vnd / adtv_value) * 100.0


def _reference_budget_plan(
    session_tranches: Sequence[Mapping[str, Any]] | None,
    *,
    budget_vnd: int,
    lot_size: int,
    current_price: float,
) -> tuple[List[Dict[str, Any]], float | None, int | None, int]:
    if not session_tranches:
        return [], None, None, 0

    weights: List[float] = []
    normalised: List[Mapping[str, Any]] = []
    for tranche in session_tranches:
        limit_price = _safe_float(tranche.get("LimitPrice"))
        if limit_price is None or limit_price <= 0.0:
            continue
        weight = _safe_float(tranche.get("AllocatedCapitalPctOfPortfolio"))
        if weight is None or weight <= 0.0:
            weight = _safe_float(tranche.get("MaxCapitalPctOfPortfolio"))
        if weight is None or weight <= 0.0:
            weight = 1.0
        normalised.append(tranche)
        weights.append(weight)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return [], None, None, 0

    rows: List[Dict[str, Any]] = []
    deploy_now_pct = 0.0
    deploy_now_vnd = 0
    deploy_now_qty = 0
    for tranche, weight in zip(normalised, weights):
        limit_price = float(tranche["LimitPrice"])
        tranche_pct = (weight / total_weight) * 100.0
        capital_vnd = int(round(budget_vnd * (tranche_pct / 100.0)))
        quantity = _quantity_for_budget(capital_vnd, limit_price, lot_size)
        order_value_vnd = int(round(quantity * limit_price * 1_000))
        actionable_now = current_price <= limit_price
        if actionable_now:
            deploy_now_pct += tranche_pct
            deploy_now_vnd += order_value_vnd
            deploy_now_qty += quantity
        rows.append(
            {
                "Role": tranche.get("Role"),
                "LimitPrice": round(limit_price, 2),
                "CapitalPctOfRefBudget": round(tranche_pct, 2),
                "CapitalVND": capital_vnd,
                "OrderValueVND": order_value_vnd,
                "Quantity": int(quantity),
                "ActionableNow": actionable_now,
            }
        )
    return rows, round(deploy_now_pct, 2), deploy_now_vnd, deploy_now_qty


def _liquidity_score(budget_pct_adv: float | None) -> float:
    if budget_pct_adv is None:
        return -4.0
    if budget_pct_adv <= 5.0:
        return 10.0
    if budget_pct_adv <= 10.0:
        return 6.0
    if budget_pct_adv <= 15.0:
        return 1.5
    if budget_pct_adv <= 20.0:
        return -4.0
    return max(-16.0, -4.0 - ((budget_pct_adv - 20.0) * 0.6))


def _timing_summary(
    timing_row: Mapping[str, Any] | None,
    *,
    edge_pct: float | None,
) -> str | None:
    if timing_row is None:
        return None
    window = timing_row.get("ForecastWindow")
    peak_ret = _safe_float(timing_row.get("PredPeakRetPct"))
    peak_day = _safe_float(timing_row.get("PredPeakDay"))
    close_ret = _safe_float(timing_row.get("PredCloseRetPct"))
    if window is None and peak_ret is None and peak_day is None and close_ret is None and edge_pct is None:
        return None

    parts: List[str] = []
    if window:
        parts.append(str(window))
    if peak_ret is not None:
        if peak_day is not None:
            parts.append(f"peak {peak_ret:.2f}% trong ~{peak_day:.1f} phiên")
        else:
            parts.append(f"peak {peak_ret:.2f}%")
    if close_ret is not None:
        parts.append(f"close {close_ret:.2f}%")
    elif edge_pct is not None:
        parts.append(f"net edge {edge_pct:.2f}%")
    return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _validation_summary(timing_row: Mapping[str, Any] | None) -> str | None:
    if timing_row is None:
        return None
    eval_rows = _safe_float(timing_row.get("EvalRows"))
    hit_pct = _safe_float(timing_row.get("TradeScoreHitPct"))
    peak_mae = _safe_float(timing_row.get("PeakRetMAEPct"))
    close_mae = _safe_float(timing_row.get("CloseMAEPct"))
    if eval_rows is None and hit_pct is None and peak_mae is None and close_mae is None:
        return None

    parts: List[str] = []
    if eval_rows is not None:
        parts.append(f"{int(round(eval_rows))} mẫu")
    if hit_pct is not None:
        parts.append(f"hit {hit_pct:.1f}%")
    if peak_mae is not None:
        parts.append(f"peak MAE {peak_mae:.2f}%")
    if close_mae is not None:
        parts.append(f"close MAE {close_mae:.2f}%")
    return " | ".join(parts) if parts else None


def _cycle_summary(cycle_row: Mapping[str, Any] | None) -> str | None:
    if cycle_row is None:
        return None
    window = cycle_row.get("ForecastWindow")
    peak_ret = _safe_float(cycle_row.get("PredPeakRetPct"))
    peak_days = _safe_float(cycle_row.get("PredPeakDays"))
    if window is None and peak_ret is None and peak_days is None:
        return None

    parts: List[str] = []
    if window:
        parts.append(str(window))
    if peak_ret is not None:
        if peak_days is not None:
            parts.append(f"peak {peak_ret:.2f}% trong ~{peak_days:.1f} phiên")
        else:
            parts.append(f"peak {peak_ret:.2f}%")
    return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _score_candidate(
    snapshot_row: pd.Series,
    playbook_row: pd.Series,
    market_summary: Mapping[str, Any],
    *,
    budget_vnd: int,
    state: Mapping[str, Any] | None,
    timing_row: Mapping[str, Any] | None,
    t10_edge: float | None,
    cycle_row: Mapping[str, Any] | None,
    ohlc_row: Mapping[str, Any] | None,
    ladder_row: Mapping[str, Any] | None,
    color_overlay: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    ticker = _normalise_ticker(snapshot_row["Ticker"])
    last_price = float(snapshot_row["Last"])
    rsi14 = float(snapshot_row["RSI14"])
    dist_sma20 = float(snapshot_row["DistSMA20Pct"])
    ret20_vs_index = float(snapshot_row["Ret20dVsIndex"])
    ret60_vs_index = float(snapshot_row["Ret60dVsIndex"])
    sector_breadth_5d = _safe_float(snapshot_row.get("SectorBreadthPositive5dPct")) or 0.0
    adtv_shares = float(snapshot_row["ADTV20_shares"])
    lot_size = int(snapshot_row["LotSize"])
    pos52w = _safe_float(snapshot_row.get("Pos52wPct"))

    zone_low, zone_high, zone_source = _preferred_zone(snapshot_row, state=state, ladder_row=ladder_row)
    anchored_zone = zone_source in {"state", "ladder"}
    zone_status, zone_gap_pct = _zone_status(last_price, zone_low, zone_high)
    budget_pct_adv = _budget_pct_adv(last_price, adtv_shares, budget_vnd)

    all_score = _safe_float(playbook_row.get("AllScore")) or 0.0
    robust_score = _safe_float(playbook_row.get("RobustScore")) or 0.0
    latest_signal = bool(playbook_row.get("LatestSignal"))
    forecast_close_ret = _safe_float(ohlc_row.get("ForecastCloseRetPct")) if ohlc_row is not None else None
    best_timing_edge = _safe_float(timing_row.get("PredNetEdgePct")) if timing_row is not None else _safe_float(state.get("BestTimingNetEdgePct") if state else None)
    best_timing_peak_ret = _safe_float(timing_row.get("PredPeakRetPct")) if timing_row is not None else None
    best_timing_peak_day = _safe_float(timing_row.get("PredPeakDay")) if timing_row is not None else None
    best_timing_close_ret = _safe_float(timing_row.get("PredCloseRetPct")) if timing_row is not None else None
    best_timing_drawdown = _safe_float(timing_row.get("PredDrawdownPct")) if timing_row is not None else None
    best_timing_cap_efficiency = _safe_float(timing_row.get("PredCapitalEfficiencyPctPerDay")) if timing_row is not None else None
    best_timing_eval_rows = _safe_float(timing_row.get("EvalRows")) if timing_row is not None else None
    best_timing_peak_mae = _safe_float(timing_row.get("PeakRetMAEPct")) if timing_row is not None else None
    best_timing_close_mae = _safe_float(timing_row.get("CloseMAEPct")) if timing_row is not None else None
    best_timing_drawdown_mae = _safe_float(timing_row.get("DrawdownMAEPct")) if timing_row is not None else None
    best_timing_hit_pct = _safe_float(timing_row.get("TradeScoreHitPct")) if timing_row is not None else None
    cycle_window = cycle_row.get("ForecastWindow") if cycle_row is not None else None
    cycle_peak_ret = _safe_float(cycle_row.get("PredPeakRetPct")) if cycle_row is not None else None
    cycle_peak_days = _safe_float(cycle_row.get("PredPeakDays")) if cycle_row is not None else None
    entry_score = _safe_float(ladder_row.get("EntryScore")) if ladder_row is not None else None
    suggested_new_capital_pct = _safe_float(state.get("SuggestedNewCapitalPct")) if state else None
    deferred_build_pct = _safe_float(state.get("DeferredBuildPct")) if state else None
    persistent_weakness_bid = bool(state.get("PersistentWeaknessBid")) if state else False
    invalidation_below = _safe_float(state.get("DamageBelow")) if state else None
    breakout_confirm_above = _safe_float(state.get("BullishConfirmAbove")) if state else None
    trend_overlay_score = int(color_overlay.get("OverlayScore") or 0) if color_overlay else 0
    trend_overlay_summary = color_overlay.get("Summary") if color_overlay else None

    market_crowded = (
        (_safe_float(market_summary.get("IndexRangePos20")) or 0.0) >= 0.95
        and (_safe_float(market_summary.get("BreadthAboveSMA20Pct")) or 0.0) >= 75.0
        and (_safe_float(market_summary.get("BreadthPositive5dPct")) or 0.0) >= 55.0
    )

    score = 50.0
    score += _clamp(all_score * 0.30, -12.0, 12.0)
    score += _clamp(robust_score * 0.30, -15.0, 15.0)
    score += _clamp(ret20_vs_index * 0.25, -8.0, 8.0)
    score += _clamp(ret60_vs_index * 0.15, -5.0, 5.0)
    score += _clamp((sector_breadth_5d - 50.0) * 0.12, -6.0, 6.0)
    score += _liquidity_score(budget_pct_adv)

    foreign_20d_bn = None
    if "NetBuySellForeign_kVND_20d" in snapshot_row.index:
        raw_foreign_20d = _safe_float(snapshot_row["NetBuySellForeign_kVND_20d"])
        if raw_foreign_20d is not None:
            foreign_20d_bn = raw_foreign_20d / 1_000_000.0
            score += _clamp(foreign_20d_bn / 200.0, -4.0, 4.0)

    if latest_signal:
        score += 6.0

    if anchored_zone:
        if zone_status == "inside":
            score += 10.0
        elif zone_status == "above":
            gap = zone_gap_pct or 0.0
            if gap <= 3.0:
                score += 2.0
            elif gap <= 8.0:
                score -= 2.0
            else:
                score -= 6.0
        elif zone_status == "below":
            gap = zone_gap_pct or 0.0
            if gap <= 2.0:
                score += 1.0
            else:
                score -= 4.0

    if dist_sma20 > 8.0:
        score -= _clamp((dist_sma20 - 8.0) * 1.3, 0.0, 16.0)
    if rsi14 > 65.0:
        score -= _clamp((rsi14 - 65.0) * 0.9, 0.0, 12.0)
    if pos52w is not None and pos52w >= 0.75:
        score -= _clamp((pos52w - 0.75) * 36.0, 0.0, 9.0)

    if best_timing_edge is not None:
        score += _clamp(best_timing_edge * 0.8, -12.0, 12.0)
    if t10_edge is not None:
        score += _clamp(t10_edge * 0.3, -6.0, 6.0)
    if forecast_close_ret is not None:
        score += _clamp(forecast_close_ret * 1.3, -6.0, 6.0)
    if entry_score is not None:
        score += _clamp(entry_score * 1.5, 0.0, 10.0)
    score += _clamp(float(trend_overlay_score), -6.0, 6.0)

    if market_crowded:
        score -= 4.0
        if zone_status == "above":
            score -= 4.0

    score = _clamp(score, 0.0, 100.0)

    very_stretched = dist_sma20 >= 18.0 or rsi14 >= 74.0
    stretched = dist_sma20 >= 12.0 or rsi14 >= 68.0
    timing_bad = best_timing_edge is not None and best_timing_edge < -1.5 and (t10_edge is None or t10_edge < 0.0)
    ohlc_bad = forecast_close_ret is not None and forecast_close_ret < -1.0
    liquidity_bad = budget_pct_adv is not None and budget_pct_adv > 20.0
    playbook_bad = robust_score < 0.0 and all_score < 20.0

    if liquidity_bad or (very_stretched and timing_bad) or (playbook_bad and timing_bad):
        decision = "không_mua"
    elif zone_status == "inside":
        if anchored_zone and score >= 62.0 and not very_stretched and not timing_bad and not ohlc_bad:
            decision = "mua_ngay"
        elif anchored_zone and score >= 58.0 and best_timing_edge is None and forecast_close_ret is None and not stretched:
            decision = "mua_ngay"
        elif score >= 52.0:
            decision = "chờ"
        else:
            decision = "không_mua"
    elif zone_status == "above":
        decision = "chờ" if score >= 48.0 and not (very_stretched and timing_bad) else "không_mua"
    elif zone_status == "below":
        decision = "chờ" if score >= 55.0 else "không_mua"
    else:
        if score >= 65.0 and not stretched and (latest_signal or (best_timing_edge or 0.0) > 4.0):
            decision = "mua_ngay"
        elif score >= 52.0:
            decision = "chờ"
        else:
            decision = "không_mua"

    if market_crowded and decision == "mua_ngay" and zone_status != "inside":
        decision = "chờ"
    if decision == "mua_ngay" and not anchored_zone:
        decision = "chờ"
    if very_stretched and decision == "mua_ngay":
        decision = "chờ"
    if timing_bad and decision == "mua_ngay":
        decision = "chờ"

    reference_price = _reference_buy_price(
        decision=decision,
        last_price=last_price,
        zone_low=zone_low,
        zone_high=zone_high,
        ladder_row=ladder_row,
    )
    reference_quantity = _quantity_for_budget(budget_vnd, reference_price, lot_size)
    reference_budget_plan, ref_deploy_now_pct, ref_deploy_now_vnd, ref_deploy_now_qty = _reference_budget_plan(
        state.get("SessionBuyTranches") if state else None,
        budget_vnd=budget_vnd,
        lot_size=lot_size,
        current_price=last_price,
    )
    recommended_deploy_pct = None
    if suggested_new_capital_pct is not None and decision != "không_mua":
        recommended_deploy_pct = _clamp(suggested_new_capital_pct, 0.0, 100.0)
    recommended_deploy_vnd = int(round(budget_vnd * (recommended_deploy_pct / 100.0))) if recommended_deploy_pct is not None else None
    recommended_deploy_quantity = (
        _quantity_for_budget(recommended_deploy_vnd, reference_price, lot_size)
        if recommended_deploy_vnd is not None
        else 0
    )
    no_chase_above = _round_or_none(zone_high, 2) if zone_high is not None else None
    validation_summary = _validation_summary(timing_row)
    conservative_peak_ret = None
    if best_timing_peak_ret is not None and best_timing_peak_mae is not None:
        conservative_peak_ret = best_timing_peak_ret - best_timing_peak_mae
    conservative_close_ret = None
    if best_timing_close_ret is not None and best_timing_close_mae is not None:
        conservative_close_ret = best_timing_close_ret - best_timing_close_mae
    conservative_drawdown = None
    if best_timing_drawdown is not None and best_timing_drawdown_mae is not None:
        conservative_drawdown = best_timing_drawdown - best_timing_drawdown_mae
    reference_budget_plan_summary = None
    if reference_budget_plan:
        reference_budget_plan_summary = ", ".join(
            f"{item['LimitPrice']}: {item['CapitalPctOfRefBudget']:.2f}% ngân sách"
            for item in reference_budget_plan
        )

    reasons: List[str] = []
    if zone_status == "inside":
        if anchored_zone:
            reasons.append("giá đang nằm trong vùng mua tham chiếu")
        else:
            reasons.append("giá đang chạm fallback grid ngắn hạn, chưa phải buy zone neo theo research/ladder")
    elif zone_status == "above":
        if anchored_zone:
            reasons.append("giá đang đứng trên vùng mua, nên chờ pullback")
        else:
            reasons.append("giá đang đứng trên fallback grid ngắn hạn, chưa đủ để coi là vùng mua chắc")
    elif zone_status == "below":
        if anchored_zone:
            reasons.append("giá đang dưới vùng chờ; cần xem phản ứng hồi lại")
        else:
            reasons.append("giá đang dưới fallback grid ngắn hạn, nhưng chưa có vùng mua neo đủ chắc")
    else:
        reasons.append("chưa xác định được vùng mua chuẩn từ artifact hiện có")

    if market_crowded:
        reasons.append("thị trường chung đang sát mép trên ngắn hạn")
    if very_stretched:
        reasons.append("mã đang kéo dãn mạnh so với nền ngắn hạn")
    elif stretched:
        reasons.append("mã đã hơi nóng, không phù hợp đuổi giá")
    if best_timing_edge is not None:
        reasons.append(f"timing edge tốt nhất {best_timing_edge:.2f}%")
    timing_summary = _timing_summary(timing_row, edge_pct=best_timing_edge)
    if timing_summary is not None:
        reasons.append(f"kỳ vọng ngắn hạn {timing_summary}")
    if forecast_close_ret is not None:
        reasons.append(f"OHLC T+1 kỳ vọng close {forecast_close_ret:.2f}%")
    cycle_summary = _cycle_summary(cycle_row)
    if cycle_summary is not None:
        reasons.append(f"cycle {cycle_summary}")
    if budget_pct_adv is not None:
        reasons.append(f"budget 5 tỷ tương đương {budget_pct_adv:.2f}% ADV20")
    if trend_overlay_summary:
        reasons.append(f"trend persistence {trend_overlay_summary}")

    return {
        "Ticker": ticker,
        "Sector": snapshot_row["Sector"],
        "Decision": decision,
        "CandidateScore": round(score, 2),
        "ReasonSummary": "; ".join(reasons[:4]),
        "CurrentPrice": round(last_price, 2),
        "PreferredBuyZoneLow": _round_or_none(zone_low, 2),
        "PreferredBuyZoneHigh": _round_or_none(zone_high, 2),
        "PreferredBuyZoneSource": zone_source,
        "AnchoredBuyZone": anchored_zone,
        "ZoneStatus": zone_status,
        "ZoneGapPct": _round_or_none(zone_gap_pct, 2),
        "ReferenceBuyPrice": _round_or_none(reference_price, 2),
        "ReferenceQuantity": int(reference_quantity),
        "ReferenceBudgetPctADV20": _round_or_none(budget_pct_adv, 2),
        "RSI14": _round_or_none(rsi14, 2),
        "DistSMA20Pct": _round_or_none(dist_sma20, 2),
        "Ret20dVsIndex": _round_or_none(ret20_vs_index, 2),
        "Ret60dVsIndex": _round_or_none(ret60_vs_index, 2),
        "SectorBreadthPositive5dPct": _round_or_none(sector_breadth_5d, 2),
        "PlaybookAllScore": _round_or_none(all_score, 2),
        "PlaybookRobustScore": _round_or_none(robust_score, 2),
        "PlaybookLatestSignal": latest_signal,
        "BestTimingWindow": (timing_row or {}).get("ForecastWindow") or (state or {}).get("BestTimingWindow"),
        "BestTimingNetEdgePct": _round_or_none(best_timing_edge, 2),
        "BestTimingPeakRetPct": _round_or_none(best_timing_peak_ret, 2),
        "BestTimingPeakDay": _round_or_none(best_timing_peak_day, 1),
        "BestTimingCloseRetPct": _round_or_none(best_timing_close_ret, 2),
        "BestTimingDrawdownPct": _round_or_none(best_timing_drawdown, 2),
        "BestTimingCapitalEfficiencyPctPerDay": _round_or_none(best_timing_cap_efficiency, 3),
        "TimingProfitSummary": timing_summary,
        "BacktestEvalRows": _round_or_none(best_timing_eval_rows, 0),
        "BacktestTradeScoreHitPct": _round_or_none(best_timing_hit_pct, 2),
        "BacktestPeakRetMAEPct": _round_or_none(best_timing_peak_mae, 2),
        "BacktestCloseMAEPct": _round_or_none(best_timing_close_mae, 2),
        "BacktestDrawdownMAEPct": _round_or_none(best_timing_drawdown_mae, 2),
        "ValidationSummary": validation_summary,
        "ConservativePeakRetPct": _round_or_none(conservative_peak_ret, 2),
        "ConservativeCloseRetPct": _round_or_none(conservative_close_ret, 2),
        "ConservativeDrawdownPct": _round_or_none(conservative_drawdown, 2),
        "T10NetEdgePct": _round_or_none(t10_edge, 2),
        "BestCycleWindow": cycle_window,
        "BestCyclePeakRetPct": _round_or_none(cycle_peak_ret, 2),
        "BestCyclePeakDays": _round_or_none(cycle_peak_days, 1),
        "CycleProfitSummary": cycle_summary,
        "ForecastCloseRetPctT1": _round_or_none(forecast_close_ret, 2),
        "ForecastCandleBias": (ohlc_row or {}).get("ForecastCandleBias"),
        "TopEntryScore": _round_or_none(entry_score, 4),
        "TopEntryLimitPrice": _round_or_none((ladder_row or {}).get("LimitPrice"), 2),
        "TopEntryFillScore": _round_or_none((ladder_row or {}).get("FillScoreComposite"), 2),
        "SuggestedNewCapitalPct": _round_or_none(suggested_new_capital_pct, 2),
        "DeferredBuildPct": _round_or_none(deferred_build_pct, 2),
        "RecommendedDeployPctOfRefBudget": _round_or_none(recommended_deploy_pct, 2),
        "RecommendedDeployVND": recommended_deploy_vnd,
        "RecommendedDeployQuantity": int(recommended_deploy_quantity),
        "ReferenceBudgetFullPlanPct": 100.0 if reference_budget_plan and decision != "không_mua" else None,
        "ReferenceBudgetDeployNowPct": _round_or_none(ref_deploy_now_pct, 2) if decision != "không_mua" else None,
        "ReferenceBudgetDeployNowVND": ref_deploy_now_vnd if decision != "không_mua" else None,
        "ReferenceBudgetDeployNowQuantity": int(ref_deploy_now_qty) if decision != "không_mua" else 0,
        "ReferenceBudgetPlanSummary": reference_budget_plan_summary,
        "NoChaseAbove": no_chase_above,
        "InvalidationBelow": _round_or_none(invalidation_below, 2),
        "BreakoutConfirmAbove": _round_or_none(breakout_confirm_above, 2),
        "PersistentWeaknessBid": persistent_weakness_bid,
        "SessionBuyPlanSummary": (state or {}).get("SessionBuyPlanSummary"),
        "TrendPersistenceOverlayScore": int(trend_overlay_score),
        "TrendPersistenceSummary": trend_overlay_summary,
        "MarketCrowded": market_crowded,
        "ForeignNetValue20DBnVND": _round_or_none(foreign_20d_bn, 2),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    grouped: Dict[str, List[Mapping[str, Any]]] = {key: [] for key in ("mua_ngay", "chờ", "không_mua")}
    for row in report["Rows"]:
        grouped.setdefault(row["Decision"], []).append(row)

    lines: List[str] = []
    lines.append(f"# Candidate Watchlist ({report['Mode']})")
    lines.append("")
    lines.append(f"- SnapshotDate: `{report['SnapshotDate']}`")
    lines.append(f"- Budget assumption: `{report['BudgetVND']:,} VND`")
    lines.append(f"- Market crowded: `{report['MarketCrowded']}`")
    lines.append("")

    for decision in ("mua_ngay", "chờ", "không_mua"):
        lines.append(f"## {decision}")
        rows = grouped.get(decision) or []
        if not rows:
            lines.append("- Không có mã.")
            lines.append("")
            continue
        for row in rows:
            extras: List[str] = []
            if row.get("TimingProfitSummary"):
                extras.append(f"timing `{row['TimingProfitSummary']}`")
            if row.get("CycleProfitSummary"):
                extras.append(f"cycle `{row['CycleProfitSummary']}`")
            if row.get("ValidationSummary"):
                extras.append(f"verify `{row['ValidationSummary']}`")
            if row.get("TrendPersistenceSummary"):
                extras.append(f"trend `{row['TrendPersistenceSummary']}`")
            if row.get("RecommendedDeployPctOfRefBudget") is not None:
                extras.append(
                    f"deploy `~{row['RecommendedDeployPctOfRefBudget']}%` ref budget"
                )
            if row.get("ReferenceBudgetFullPlanPct") is not None:
                extras.append(
                    f"full-plan `{row['ReferenceBudgetFullPlanPct']}%` ref budget"
                )
            if row.get("ReferenceBudgetDeployNowPct") is not None:
                extras.append(
                    f"now `{row['ReferenceBudgetDeployNowPct']}%` ref budget"
                )
            if row.get("NoChaseAbove") is not None:
                extras.append(f"no-chase `>{row['NoChaseAbove']}`")
            if row.get("InvalidationBelow") is not None:
                extras.append(f"invalid `<{row['InvalidationBelow']}`")
            extra_summary = " | ".join(extras)
            lines.append(
                "- "
                f"{row['Ticker']} | score `{row['CandidateScore']}` | "
                f"zone `{row['PreferredBuyZoneLow']} - {row['PreferredBuyZoneHigh']}` ({row['PreferredBuyZoneSource']}) | "
                f"ref `{row['ReferenceBuyPrice']}` / `{row['ReferenceQuantity']}` cp | "
                f"{extra_summary + ' | ' if extra_summary else ''}"
                f"{row['ReasonSummary']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_watchlist(
    *,
    mode: str,
    budget_vnd: int,
    universe_csv: Path,
    market_summary_json: Path,
    sector_summary_csv: Path,
    analysis_dir: Path,
    research_dir: Path,
    output_dir: Path,
    ticker_color_dir: Path = DEFAULT_TICKER_COLOR_DIR,
) -> Dict[str, Any]:
    resolved_mode = _resolve_mode(analysis_dir, research_dir, mode)

    universe_df = _load_csv(
        universe_csv,
        [
            "Ticker",
            "Sector",
            "Last",
            "RSI14",
            "DistSMA20Pct",
            "Ret20dVsIndex",
            "Ret60dVsIndex",
            "ADTV20_shares",
            "TickSize",
            "LotSize",
            "GridBelow_T1",
            "GridBelow_T2",
            "GridBelow_T3",
            "SectorBreadthPositive5dPct",
        ],
        "Universe snapshot",
    )
    _load_csv(sector_summary_csv, ["Sector", "TickerCount"], "Sector summary")
    market_summary = _load_json(market_summary_json, "Market summary")

    playbook_df = _load_csv(
        analysis_dir / "ticker_playbooks_live" / "ticker_playbook_best_configs.csv",
        ["Ticker", "LatestSignal", "AllScore", "RobustScore"],
        "Ticker playbook best configs",
    )
    color_comparison_df, color_current_df = load_optional_overlay(ticker_color_dir)

    ohlc_df = None
    timing_df = None
    ladder_df = None
    cycle_df = None
    if resolved_mode == FULL_MODE:
        ohlc_df = _load_csv(
            analysis_dir / "ml_ohlc_next_session.csv",
            ["Ticker", "ForecastCloseRetPct", "ForecastCandleBias"],
            "Next-session OHLC",
        )
        timing_df = _load_csv(
            analysis_dir / "ml_single_name_timing.csv",
            [
                "Ticker",
                "Horizon",
                "ForecastWindow",
                "PredNetEdgePct",
                "PredPeakRetPct",
                "PredPeakDay",
                "PredDrawdownPct",
                "PredCloseRetPct",
                "PredCapitalEfficiencyPctPerDay",
                "EvalRows",
                "PeakRetMAEPct",
                "DrawdownMAEPct",
                "CloseMAEPct",
                "TradeScoreHitPct",
            ],
            "Single-name timing",
        )
        ladder_df = _load_csv(
            analysis_dir / "ml_entry_ladder_eval.csv",
            ["Ticker", "EntryScoreRank", "LimitPrice", "FillScoreComposite", "EntryScore"],
            "Entry ladder",
        )
        cycle_df = _load_optional_csv(
            analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv",
            ["Ticker", "ForecastWindow", "PredPeakRetPct", "PredPeakDays", "PredDrawdownPct"],
            "Cycle best horizon",
        )
        _load_optional_json(research_dir / "manifest.json")

    merged = universe_df.loc[~universe_df["Ticker"].eq("VNINDEX")].merge(playbook_df, on="Ticker", how="left")
    if merged.empty:
        raise RuntimeError("Universe snapshot does not contain any reportable tickers")

    rows: List[Dict[str, Any]] = []
    for _, snapshot_row in merged.iterrows():
        ticker = _normalise_ticker(snapshot_row["Ticker"])
        state = _load_state(research_dir, ticker) if resolved_mode == FULL_MODE else None

        timing_row, t10_edge = _extract_timing_rows(timing_df, ticker)
        cycle_row = _extract_cycle_row(cycle_df, ticker)
        ladder_row = _extract_entry_row(ladder_df, ticker)
        ohlc_row = None
        if ohlc_df is not None:
            scoped_ohlc = ohlc_df.loc[ohlc_df["Ticker"].eq(ticker)]
            if not scoped_ohlc.empty:
                ohlc_row = scoped_ohlc.iloc[0].to_dict()
        color_overlay = summarise_ticker_overlay(
            extract_ticker_overlay(ticker, color_comparison_df, color_current_df)
        )

        rows.append(
            _score_candidate(
                snapshot_row,
                snapshot_row,
                market_summary,
                budget_vnd=budget_vnd,
                state=state,
                timing_row=timing_row,
                t10_edge=t10_edge,
                cycle_row=cycle_row,
                ohlc_row=ohlc_row,
                ladder_row=ladder_row,
                color_overlay=color_overlay,
            )
        )

    decision_order = {"mua_ngay": 0, "chờ": 1, "không_mua": 2}
    report_df = pd.DataFrame(rows)
    report_df["DecisionOrder"] = report_df["Decision"].map(decision_order).fillna(9)
    report_df = report_df.sort_values(["DecisionOrder", "CandidateScore", "Ticker"], ascending=[True, False, True]).reset_index(drop=True)
    report_df["DecisionRank"] = report_df.groupby("Decision").cumcount() + 1
    report_df = report_df.drop(columns=["DecisionOrder"])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"candidate_watchlist_{resolved_mode}.csv"
    json_path = output_dir / f"candidate_watchlist_{resolved_mode}.json"
    md_path = output_dir / f"candidate_watchlist_{resolved_mode}.md"
    report_df.to_csv(csv_path, index=False)

    report = {
        "Mode": resolved_mode,
        "BudgetVND": int(budget_vnd),
        "SnapshotDate": market_summary.get("GeneratedAt"),
        "MarketCrowded": bool(
            (_safe_float(market_summary.get("IndexRangePos20")) or 0.0) >= 0.95
            and (_safe_float(market_summary.get("BreadthAboveSMA20Pct")) or 0.0) >= 75.0
            and (_safe_float(market_summary.get("BreadthPositive5dPct")) or 0.0) >= 55.0
        ),
        "Rows": [{column: _json_safe(value) for column, value in row.items()} for row in report_df.to_dict(orient="records")],
        "OutputCSV": _display_path(csv_path),
        "OutputJSON": _display_path(json_path),
        "OutputMarkdown": _display_path(md_path),
    }
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a ranked candidate watchlist from the current notebook artifacts."
    )
    parser.add_argument("--mode", choices=[AUTO_MODE, CORE_MODE, FULL_MODE], default=AUTO_MODE)
    parser.add_argument("--budget-vnd", type=int, default=DEFAULT_BUDGET_VND)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--market-summary-json", type=Path, default=DEFAULT_MARKET_SUMMARY_JSON)
    parser.add_argument("--sector-summary-csv", type=Path, default=DEFAULT_SECTOR_SUMMARY_CSV)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticker-color-dir", type=Path, default=DEFAULT_TICKER_COLOR_DIR)
    args = parser.parse_args(argv)

    report = build_candidate_watchlist(
        mode=args.mode,
        budget_vnd=args.budget_vnd,
        universe_csv=args.universe_csv,
        market_summary_json=args.market_summary_json,
        sector_summary_csv=args.sector_summary_csv,
        analysis_dir=args.analysis_dir,
        research_dir=args.research_dir,
        output_dir=args.output_dir,
        ticker_color_dir=args.ticker_color_dir,
    )
    print(json.dumps(_json_safe(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
