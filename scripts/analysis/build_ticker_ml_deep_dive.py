from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "out" / "universe.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_RESEARCH_DIR = REPO_ROOT / "research"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "deep_dive"
DEFAULT_BUDGET_VND = 5_000_000_000
OHLC_STATE_COLUMNS = [
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
    "IndexColorStreakState",
    "VN30ColorStreakState",
]


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _normalise_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _load_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    if "Ticker" in frame.columns:
        frame = frame.copy()
        frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    return frame


def _select_single_row(frame: pd.DataFrame, ticker: str, label: str) -> pd.Series:
    scoped = frame.loc[frame["Ticker"].eq(ticker)]
    if scoped.empty:
        raise RuntimeError(f"{label} does not contain ticker {ticker}")
    return scoped.iloc[0]


def _load_state(research_dir: Path, ticker: str) -> Dict[str, Any]:
    state_path = research_dir / "tickers" / ticker / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Ticker state not found for {ticker}: {state_path}. Run ./broker.sh prime first.")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
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


def _validation_summary(row: Mapping[str, Any]) -> str | None:
    eval_rows = _safe_float(row.get("EvalRows"))
    hit_pct = _safe_float(row.get("TradeScoreHitPct"))
    peak_mae = _safe_float(row.get("PeakRetMAEPct"))
    close_mae = _safe_float(row.get("CloseMAEPct"))
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


def _state_regime_label(value: Any, *, positive: str, negative: str) -> str | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric >= 0.5:
        return positive
    if numeric <= -0.5:
        return negative
    return "neutral"


def _summarise_ohlc_state(ohlc_row: pd.Series) -> str | None:
    labels = [
        _state_regime_label(ohlc_row.get("TickerColorStreakState"), positive="green streak", negative="red streak"),
        _state_regime_label(ohlc_row.get("TickerLimitProxyState"), positive="ceiling proxy", negative="floor proxy"),
        _state_regime_label(ohlc_row.get("TickerShockState1D"), positive="shock up", negative="shock down"),
        _state_regime_label(ohlc_row.get("TickerImpulseState3D"), positive="3-day impulse up", negative="3-day impulse down"),
        _state_regime_label(ohlc_row.get("TickerWideRangeState"), positive="wide-range expansion", negative="wide-range breakdown"),
        _state_regime_label(ohlc_row.get("TickerTrendRegimeState"), positive="hot trend regime", negative="hot down regime"),
        _state_regime_label(ohlc_row.get("TickerCompressionState"), positive="compression above base", negative="compression below base"),
        _state_regime_label(ohlc_row.get("TickerReclaimState"), positive="sma20 reclaim", negative="sma20 loss"),
        _state_regime_label(ohlc_row.get("TickerRelativeRotationState"), positive="relative rotation up", negative="relative rotation down"),
        _state_regime_label(ohlc_row.get("TickerExhaustionState"), positive="upside exhaustion", negative="downside capitulation"),
        _state_regime_label(ohlc_row.get("IndexColorStreakState"), positive="VNINDEX green streak", negative="VNINDEX red streak"),
        _state_regime_label(ohlc_row.get("VN30ColorStreakState"), positive="VN30 green streak", negative="VN30 red streak"),
    ]
    active = [label for label in labels if label and label != "neutral"]
    if active:
        return ", ".join(active)
    if any(label == "neutral" for label in labels):
        return "neutral"
    return None


def _reference_budget_plan(
    session_tranches: Sequence[Mapping[str, Any]],
    *,
    budget_vnd: int,
    lot_size: int,
    current_price: float,
) -> tuple[List[Dict[str, Any]], float, int, int]:
    if not session_tranches:
        return [], 0.0, 0, 0

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
        return [], 0.0, 0, 0

    rows: List[Dict[str, Any]] = []
    deploy_now_pct = 0.0
    deploy_now_vnd = 0
    deploy_now_qty = 0
    for tranche, weight in zip(normalised, weights):
        limit_price = float(tranche["LimitPrice"])
        tranche_pct = (weight / total_weight) * 100.0
        capital_vnd = int(round(budget_vnd * (tranche_pct / 100.0)))
        qty_lots = int(capital_vnd // (limit_price * 1_000 * lot_size))
        quantity = qty_lots * lot_size
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
                "Reason": tranche.get("Reason"),
                "EntryScoreRank": _round_or_none(tranche.get("EntryScoreRank"), 2),
            }
        )
    return rows, round(deploy_now_pct, 2), deploy_now_vnd, deploy_now_qty


def _build_budget_plan(
    session_tranches: Sequence[Mapping[str, Any]],
    *,
    deploy_budget_vnd: int,
    lot_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not session_tranches or deploy_budget_vnd <= 0:
        return rows

    weights: List[float] = []
    for tranche in session_tranches:
        weight = float(tranche.get("AllocatedCapitalPctOfPortfolio") or 0.0)
        if weight <= 0:
            weight = float(tranche.get("MaxCapitalPctOfPortfolio") or 0.0)
        if weight <= 0:
            weight = 1.0
        weights.append(weight)
    total_weight = sum(weights)
    if total_weight <= 0:
        return rows

    for tranche, weight in zip(session_tranches, weights):
        limit_price = float(tranche["LimitPrice"])
        capital_vnd = deploy_budget_vnd * (weight / total_weight)
        qty_lots = int(capital_vnd // (limit_price * 1_000 * lot_size))
        quantity = qty_lots * lot_size
        order_value_vnd = int(round(quantity * limit_price * 1_000))
        rows.append(
            {
                "Role": tranche.get("Role"),
                "LimitPrice": limit_price,
                "CapitalPctOfBudget": round((weight / total_weight) * 100.0, 2),
                "CapitalVND": int(round(capital_vnd)),
                "OrderValueVND": order_value_vnd,
                "Quantity": int(quantity),
                "Reason": tranche.get("Reason"),
                "EntryScoreRank": _round_or_none(tranche.get("EntryScoreRank"), 2),
            }
        )
    return rows


def _derive_verdict(
    *,
    snapshot: pd.Series,
    state: Mapping[str, Any],
    ohlc_row: pd.Series,
    timing_rows: pd.DataFrame,
) -> tuple[str, List[str]]:
    last_price = float(snapshot["Last"])
    dist_sma20 = float(snapshot["DistSMA20Pct"])
    rsi14 = float(snapshot["RSI14"])
    forecast_close_ret = float(ohlc_row["ForecastCloseRetPct"])
    has_session_plan = bool(state.get("SessionBuyTranches"))

    best_timing_edge = float(state.get("BestTimingNetEdgePct") or timing_rows["PredNetEdgePct"].max())
    t10_rows = timing_rows.loc[timing_rows["Horizon"].astype(int).eq(10)]
    t10_edge = float(state.get("T10NetEdgePct") or (t10_rows["PredNetEdgePct"].iloc[0] if not t10_rows.empty else 0.0))

    preferred_low = _round_or_none(state.get("PreferredBuyZoneLow"), 2)
    preferred_high = _round_or_none(state.get("PreferredBuyZoneHigh"), 2)
    in_preferred_zone = (
        preferred_low is not None
        and preferred_high is not None
        and preferred_low <= last_price <= preferred_high
    )

    deeply_extended = dist_sma20 >= 15.0 or rsi14 >= 74.0
    extended = dist_sma20 >= 10.0 or rsi14 >= 68.0
    long_timing_negative = best_timing_edge < 0.0 and t10_edge < 0.0

    reasons: List[str] = []
    if has_session_plan:
        reasons.append("research state có session buy tranches rõ ràng")
        if in_preferred_zone and best_timing_edge > 0 and forecast_close_ret >= -0.5:
            verdict = "BUY_NOW"
            reasons.append("giá đang nằm trong vùng ưu tiên và timing ngắn hạn chưa phủ định")
        else:
            verdict = "RESTING_BUY_ONLY"
            reasons.append("nên đặt resting bids theo ladder thay vì mua đuổi ở giá hiện tại")
    else:
        if deeply_extended:
            verdict = "NO_BUY_NOW"
            reasons.append("giá đang quá kéo dãn so với nền ngắn hạn")
        elif long_timing_negative:
            verdict = "NO_BUY_NOW"
            reasons.append("timing tốt nhất và T+10 đều âm, không phù hợp dồn vốn lớn")
        elif extended and forecast_close_ret < 0:
            verdict = "NO_BUY_NOW"
            reasons.append("giá đang nóng trong khi OHLC T+1 không ủng hộ")
        elif preferred_high is not None and last_price > preferred_high:
            verdict = "WAIT_FOR_PULLBACK"
            reasons.append("giá đã vượt vùng chờ tốt hơn do research state gợi ý")
        else:
            verdict = "WATCH"
            reasons.append("chưa có đủ xác nhận để xem như điểm vào lớn")

    reasons.append(f"BestTimingNetEdge {best_timing_edge:.2f}% | T10NetEdge {t10_edge:.2f}%")
    reasons.append(f"ForecastCloseRet T+1 {forecast_close_ret:.2f}% | DistSMA20 {dist_sma20:.2f}% | RSI14 {rsi14:.2f}")
    return verdict, reasons


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# {report['Ticker']} ML Deep Dive")
    lines.append("")
    lines.append(f"- Verdict: `{report['Verdict']}`")
    lines.append(f"- Last: `{report['Snapshot']['Last']}`")
    lines.append(f"- Sector: `{report['Snapshot']['Sector']}`")
    lines.append(f"- Budget assumption: `{report['BudgetVND']:,} VND`")
    lines.append("")
    lines.append("## Why")
    for reason in report["VerdictReasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Snapshot")
    snapshot = report["Snapshot"]
    for key in [
        "Ret5d",
        "Ret20dVsIndex",
        "Ret60dVsIndex",
        "RSI14",
        "DistSMA20Pct",
        "Pos52wPct",
        "ADTV20BnVND",
        "ForeignFlow5dBnVND",
        "ForeignFlow20dBnVND",
    ]:
        lines.append(f"- {key}: `{snapshot.get(key)}`")
    lines.append("")
    lines.append("## Timing")
    for row in report["TimingRows"]:
        lines.append(
            "- "
            f"{row['ForecastWindow']}: peak `{row['PredPeakRetPct']}%`, drawdown `{row['PredDrawdownPct']}%`, "
            f"close `{row['PredCloseRetPct']}%`, conservative close `{row['ConservativeCloseRetPct']}%`, "
            f"net edge `{row['PredNetEdgePct']}%`, RR `{row['PredRewardRisk']}`, verify `{row['ValidationSummary']}`"
        )
    lines.append("")
    lines.append("## OHLC T+1")
    ohlc = report["NextSessionOHLC"]
    lines.append(
        f"- Open `{ohlc['ForecastOpen']}`, High `{ohlc['ForecastHigh']}`, Low `{ohlc['ForecastLow']}`, "
        f"Close `{ohlc['ForecastClose']}` ({ohlc['ForecastCloseRetPct']}%), bias `{ohlc['ForecastCandleBias']}`"
    )
    state_signals = report["OHLCStateSignals"]
    lines.append(f"- Regime summary: `{state_signals['Summary']}`")
    for key in OHLC_STATE_COLUMNS:
        lines.append(f"- {key}: `{state_signals.get(key)}`")
    lines.append("")
    lines.append("## Entry Ladder")
    if report["TopLadderRows"]:
        for row in report["TopLadderRows"]:
            lines.append(
                "- "
                f"{row['LimitPrice']} via `{row['EntryAnchor']}`: entry vs last `{row['EntryVsLastPct']}%`, "
                f"net close `{row['NetRetToNextClosePct']}%`, net high `{row['NetRetToNextHighPct']}%`, "
                f"timing `{row['BestTimingWindow']}` / edge `{row['BestTimingNetEdgePct']}%`, "
                f"cycle `{row['CycleNetEdgePct']}%`, rank `{row['EntryScoreRank']}`"
            )
    else:
        lines.append("- No entry ladder rows.")
    lines.append("")
    lines.append("## Research State")
    state = report["State"]
    for key in [
        "Archetype",
        "PreferredHoldWindow",
        "DailySummary",
        "WeeklySummary",
        "SessionBuyPlanSummary",
        "PreferredBuyZoneLow",
        "PreferredBuyZoneHigh",
        "DamageBelow",
        "BullishConfirmAbove",
    ]:
        lines.append(f"- {key}: `{state.get(key)}`")
    lines.append("")
    lines.append("## Risk And Sizing")
    sizing = report["Sizing"]
    for key in [
        "RecommendedDeployPctOfRefBudget",
        "RecommendedDeployVND",
        "ReferenceBudgetFullPlanPct",
        "ReferenceBudgetDeployNowPct",
        "ReferenceBudgetDeployNowVND",
        "ReferenceBudgetDeployNowQuantity",
        "NoChaseAbove",
        "InvalidationBelow",
        "BreakoutConfirmAbove",
    ]:
        lines.append(f"- {key}: `{sizing.get(key)}`")
    lines.append("")
    lines.append("## Reference Budget Plan")
    if report["ReferenceBudgetPlan"]:
        for row in report["ReferenceBudgetPlan"]:
            lines.append(
                "- "
                f"{row['Role']} at `{row['LimitPrice']}`: `{row['Quantity']}` cp, "
                f"`{row['OrderValueVND']}` VND, `{row['CapitalPctOfRefBudget']}%` ref budget, actionable now `{row['ActionableNow']}`"
            )
    else:
        lines.append("- No full reference-budget plan from session tranches.")
    lines.append("")
    lines.append("## Budget Plan")
    if report["BudgetPlan"]:
        for row in report["BudgetPlan"]:
            lines.append(
                "- "
                f"{row['Role']} at `{row['LimitPrice']}`: `{row['Quantity']}` cp, "
                f"`{row['OrderValueVND']}` VND, `{row['CapitalPctOfBudget']}%` deploy budget"
            )
    else:
        lines.append("- No model-approved session budget plan for this ticker.")
    return "\n".join(lines) + "\n"


def build_deep_dive(
    *,
    ticker: str,
    budget_vnd: int,
    universe_csv: Path,
    analysis_dir: Path,
    research_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    ticker = _normalise_ticker(ticker)
    if not ticker:
        raise ValueError("Ticker is required")

    universe_df = _load_csv(
        universe_csv,
        [
            "Ticker",
            "Sector",
            "Last",
            "Ret5d",
            "Ret20dVsIndex",
            "Ret60dVsIndex",
            "RSI14",
            "DistSMA20Pct",
            "Pos52wPct",
            "ADTV20_shares",
            "TickSize",
            "LotSize",
        ],
        "Universe snapshot",
    )
    ohlc_df = _load_csv(
        analysis_dir / "ml_ohlc_next_session.csv",
        [
            "Ticker",
            "ForecastOpen",
            "ForecastHigh",
            "ForecastLow",
            "ForecastClose",
            "ForecastCloseRetPct",
            "ForecastCandleBias",
        ],
        "Next-session OHLC",
    )
    timing_df = _load_csv(
        analysis_dir / "ml_single_name_timing.csv",
        [
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "EvalRows",
            "PeakRetMAEPct",
            "PeakDayMAE",
            "DrawdownMAEPct",
            "CloseMAEPct",
            "TradeScoreHitPct",
            "PredPeakRetPct",
            "PredDrawdownPct",
            "PredCloseRetPct",
            "PredPeakPrice",
            "PredDrawdownPrice",
            "PredClosePrice",
            "PredRewardRisk",
            "PredNetEdgePct",
        ],
        "Single-name timing",
    )
    range_recent_df = _load_csv(
        analysis_dir / "ml_range_predictions_recent_focus.csv",
        ["Ticker", "Horizon", "ForecastWindow", "Low", "Mid", "High", "PredLowRetPct", "PredMidRetPct", "PredHighRetPct"],
        "Recent-focus range",
    )
    range_full_df = _load_csv(
        analysis_dir / "ml_range_predictions_full_2y.csv",
        ["Ticker", "Horizon", "ForecastWindow", "Low", "Mid", "High", "PredLowRetPct", "PredMidRetPct", "PredHighRetPct"],
        "Full-2Y range",
    )
    cycle_df = _load_csv(
        analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv",
        ["Ticker", "ForecastWindow", "PredPeakRetPct", "PredPeakDays", "PredPeakPrice", "PredDrawdownPct", "PredDrawdownPrice"],
        "Cycle best horizon",
    )
    ladder_df = _load_csv(
        analysis_dir / "ml_entry_ladder_eval.csv",
        [
            "Ticker",
            "EntryScoreRank",
            "LimitPrice",
            "EntryAnchor",
            "EntryVsLastPct",
            "NetRetToNextClosePct",
            "NetRetToNextHighPct",
            "BestTimingWindow",
            "BestTimingNetEdgePct",
            "BestTimingRewardRisk",
            "CycleNetEdgePct",
            "CycleRewardRisk",
            "FillScoreComposite",
            "EntryScore",
        ],
        "Entry ladder",
    )
    playbook_df = _load_csv(
        analysis_dir / "ticker_playbooks_live" / "ticker_playbook_best_configs.csv",
        ["Ticker", "StrategyFamily", "LatestSignal", "TestScore", "AllScore", "RobustScore", "TestWinRatePct", "TestAvgReturnPct", "AllAvgReturnPct", "AllWorstDrawdownPct"],
        "Ticker playbook best configs",
    )

    snapshot_row = _select_single_row(universe_df, ticker, "Universe snapshot")
    ohlc_row = _select_single_row(ohlc_df, ticker, "Next-session OHLC")
    cycle_row = _select_single_row(cycle_df, ticker, "Cycle best horizon")
    playbook_row = _select_single_row(playbook_df, ticker, "Ticker playbook best configs")
    state = _load_state(research_dir, ticker)

    ticker_timing = timing_df.loc[timing_df["Ticker"].eq(ticker)].sort_values("Horizon").reset_index(drop=True)
    if ticker_timing.empty:
        raise RuntimeError(f"Single-name timing does not contain ticker {ticker}")
    ticker_ladder = ladder_df.loc[ladder_df["Ticker"].eq(ticker)].sort_values(["EntryScoreRank", "LimitPrice"]).reset_index(drop=True)
    if ticker_ladder.empty:
        raise RuntimeError(f"Entry ladder does not contain ticker {ticker}")
    range_recent = range_recent_df.loc[range_recent_df["Ticker"].eq(ticker)].sort_values("Horizon")
    range_full = range_full_df.loc[range_full_df["Ticker"].eq(ticker)].sort_values("Horizon")

    verdict, verdict_reasons = _derive_verdict(
        snapshot=snapshot_row,
        state=state,
        ohlc_row=ohlc_row,
        timing_rows=ticker_timing,
    )

    budget_plan = _build_budget_plan(
        state.get("SessionBuyTranches") or [],
        deploy_budget_vnd=int(
            round(
                budget_vnd
                * (
                    min(max(_safe_float(state.get("SuggestedNewCapitalPct")) or 0.0, 0.0), 100.0)
                    / 100.0
                )
            )
        ),
        lot_size=int(snapshot_row["LotSize"]),
    )
    foreign_flow_5d = _safe_float(snapshot_row.get("NetBuySellForeign_kVND_5d"))
    foreign_flow_20d = _safe_float(snapshot_row.get("NetBuySellForeign_kVND_20d"))
    recommended_deploy_pct = min(max(_safe_float(state.get("SuggestedNewCapitalPct")) or 0.0, 0.0), 100.0)
    recommended_deploy_vnd = int(round(budget_vnd * (recommended_deploy_pct / 100.0)))
    no_chase_above = _round_or_none(state.get("PreferredBuyZoneHigh"), 2)
    invalidation_below = _round_or_none(state.get("DamageBelow"), 2)
    breakout_confirm_above = _round_or_none(state.get("BullishConfirmAbove"), 2)
    reference_budget_plan, reference_deploy_now_pct, reference_deploy_now_vnd, reference_deploy_now_qty = _reference_budget_plan(
        state.get("SessionBuyTranches") or [],
        budget_vnd=budget_vnd,
        lot_size=int(snapshot_row["LotSize"]),
        current_price=float(snapshot_row["Last"]),
    )

    report: Dict[str, Any] = {
        "Ticker": ticker,
        "BudgetVND": int(budget_vnd),
        "Verdict": verdict,
        "VerdictReasons": verdict_reasons,
        "Snapshot": {
            "Sector": snapshot_row["Sector"],
            "Last": _round_or_none(snapshot_row["Last"], 2),
            "Ret5d": _round_or_none(snapshot_row["Ret5d"], 2),
            "Ret20dVsIndex": _round_or_none(snapshot_row["Ret20dVsIndex"], 2),
            "Ret60dVsIndex": _round_or_none(snapshot_row["Ret60dVsIndex"], 2),
            "RSI14": _round_or_none(snapshot_row["RSI14"], 2),
            "DistSMA20Pct": _round_or_none(snapshot_row["DistSMA20Pct"], 2),
            "Pos52wPct": _round_or_none(snapshot_row["Pos52wPct"], 2),
            "ADTV20BnVND": _round_or_none((float(snapshot_row["ADTV20_shares"]) * float(snapshot_row["Last"])) / 1_000_000.0, 2),
            "ForeignFlow5dBnVND": _round_or_none(foreign_flow_5d / 1_000_000.0 if foreign_flow_5d is not None else None, 2),
            "ForeignFlow20dBnVND": _round_or_none(foreign_flow_20d / 1_000_000.0 if foreign_flow_20d is not None else None, 2),
            "TickSize": _round_or_none(snapshot_row["TickSize"], 4),
            "LotSize": int(snapshot_row["LotSize"]),
        },
        "NextSessionOHLC": {
            key: _json_safe(_round_or_none(ohlc_row[key], 4) if key != "ForecastCandleBias" else ohlc_row[key])
            for key in [
                "ForecastOpen",
                "ForecastHigh",
                "ForecastLow",
                "ForecastClose",
                "ForecastCloseRetPct",
                "ForecastCandleBias",
                *[column for column in OHLC_STATE_COLUMNS if column in ohlc_row.index],
            ]
        },
        "OHLCStateSignals": {
            "Summary": _summarise_ohlc_state(ohlc_row),
            **{
                column: _round_or_none(ohlc_row.get(column), 2)
                for column in OHLC_STATE_COLUMNS
            },
        },
        "TimingRows": [
            {
                "Horizon": int(row["Horizon"]),
                "ForecastWindow": row["ForecastWindow"],
                "EvalRows": _round_or_none(row["EvalRows"], 0),
                "PeakRetMAEPct": _round_or_none(row["PeakRetMAEPct"], 2),
                "PeakDayMAE": _round_or_none(row["PeakDayMAE"], 2),
                "DrawdownMAEPct": _round_or_none(row["DrawdownMAEPct"], 2),
                "CloseMAEPct": _round_or_none(row["CloseMAEPct"], 2),
                "TradeScoreHitPct": _round_or_none(row["TradeScoreHitPct"], 2),
                "PredPeakRetPct": _round_or_none(row["PredPeakRetPct"], 2),
                "PredDrawdownPct": _round_or_none(row["PredDrawdownPct"], 2),
                "PredCloseRetPct": _round_or_none(row["PredCloseRetPct"], 2),
                "ConservativePeakRetPct": _round_or_none(
                    (_safe_float(row["PredPeakRetPct"]) or 0.0) - (_safe_float(row["PeakRetMAEPct"]) or 0.0),
                    2,
                ),
                "ConservativeCloseRetPct": _round_or_none(
                    (_safe_float(row["PredCloseRetPct"]) or 0.0) - (_safe_float(row["CloseMAEPct"]) or 0.0),
                    2,
                ),
                "PredPeakPrice": _round_or_none(row["PredPeakPrice"], 2),
                "PredDrawdownPrice": _round_or_none(row["PredDrawdownPrice"], 2),
                "PredClosePrice": _round_or_none(row["PredClosePrice"], 2),
                "PredRewardRisk": _round_or_none(row["PredRewardRisk"], 2),
                "PredNetEdgePct": _round_or_none(row["PredNetEdgePct"], 2),
                "ValidationSummary": _validation_summary(row),
            }
            for _, row in ticker_timing.iterrows()
        ],
        "RangeRecentRows": [
            {
                "Horizon": int(row["Horizon"]),
                "ForecastWindow": row["ForecastWindow"],
                "Low": _round_or_none(row["Low"], 2),
                "Mid": _round_or_none(row["Mid"], 2),
                "High": _round_or_none(row["High"], 2),
                "PredLowRetPct": _round_or_none(row["PredLowRetPct"], 2),
                "PredMidRetPct": _round_or_none(row["PredMidRetPct"], 2),
                "PredHighRetPct": _round_or_none(row["PredHighRetPct"], 2),
            }
            for _, row in range_recent.iterrows()
        ],
        "RangeFullRows": [
            {
                "Horizon": int(row["Horizon"]),
                "ForecastWindow": row["ForecastWindow"],
                "Low": _round_or_none(row["Low"], 2),
                "Mid": _round_or_none(row["Mid"], 2),
                "High": _round_or_none(row["High"], 2),
                "PredLowRetPct": _round_or_none(row["PredLowRetPct"], 2),
                "PredMidRetPct": _round_or_none(row["PredMidRetPct"], 2),
                "PredHighRetPct": _round_or_none(row["PredHighRetPct"], 2),
            }
            for _, row in range_full.iterrows()
        ],
        "Cycle": {
            "ForecastWindow": cycle_row["ForecastWindow"],
            "PredPeakRetPct": _round_or_none(cycle_row["PredPeakRetPct"], 2),
            "PredPeakDays": _round_or_none(cycle_row["PredPeakDays"], 2),
            "PredPeakPrice": _round_or_none(cycle_row["PredPeakPrice"], 2),
            "PredDrawdownPct": _round_or_none(cycle_row["PredDrawdownPct"], 2),
            "PredDrawdownPrice": _round_or_none(cycle_row["PredDrawdownPrice"], 2),
        },
        "TopLadderRows": [
            {
                "LimitPrice": _round_or_none(row["LimitPrice"], 2),
                "EntryAnchor": row["EntryAnchor"],
                "EntryVsLastPct": _round_or_none(row["EntryVsLastPct"], 2),
                "NetRetToNextClosePct": _round_or_none(row["NetRetToNextClosePct"], 2),
                "NetRetToNextHighPct": _round_or_none(row["NetRetToNextHighPct"], 2),
                "BestTimingWindow": row["BestTimingWindow"],
                "BestTimingNetEdgePct": _round_or_none(row["BestTimingNetEdgePct"], 2),
                "BestTimingRewardRisk": _round_or_none(row["BestTimingRewardRisk"], 2),
                "CycleNetEdgePct": _round_or_none(row["CycleNetEdgePct"], 2),
                "CycleRewardRisk": _round_or_none(row["CycleRewardRisk"], 2),
                "FillScoreComposite": _round_or_none(row["FillScoreComposite"], 2),
                "EntryScoreRank": int(row["EntryScoreRank"]),
                "EntryScore": _round_or_none(row["EntryScore"], 4),
            }
            for _, row in ticker_ladder.head(5).iterrows()
        ],
        "Playbook": {
            "StrategyFamily": playbook_row["StrategyFamily"],
            "LatestSignal": bool(playbook_row["LatestSignal"]),
            "TestScore": _round_or_none(playbook_row["TestScore"], 2),
            "AllScore": _round_or_none(playbook_row["AllScore"], 2),
            "RobustScore": _round_or_none(playbook_row["RobustScore"], 2),
            "TestWinRatePct": _round_or_none(playbook_row["TestWinRatePct"], 2),
            "TestAvgReturnPct": _round_or_none(playbook_row["TestAvgReturnPct"], 2),
            "AllAvgReturnPct": _round_or_none(playbook_row["AllAvgReturnPct"], 2),
            "AllWorstDrawdownPct": _round_or_none(playbook_row["AllWorstDrawdownPct"], 2),
        },
        "State": {
            key: _json_safe(state.get(key))
            for key in [
                "Archetype",
                "PreferredHoldWindow",
                "DailySummary",
                "WeeklySummary",
                "BestTimingWindow",
                "BestTimingNetEdgePct",
                "T10NetEdgePct",
                "SuggestedNewCapitalPct",
                "DeferredBuildPct",
                "SessionBuyPlanSummary",
                "SessionBuyTranches",
                "PreferredBuyZoneLow",
                "PreferredBuyZoneHigh",
                "BullishConfirmAbove",
                "DamageBelow",
                "ExecutionBias",
                "BurstExecutionBias",
                "TrimAggression",
                "MustSellFractionPct",
            ]
        },
        "Sizing": {
            "RecommendedDeployPctOfRefBudget": _round_or_none(recommended_deploy_pct, 2),
            "RecommendedDeployVND": recommended_deploy_vnd,
            "ReferenceBudgetFullPlanPct": 100.0 if reference_budget_plan else None,
            "ReferenceBudgetDeployNowPct": _round_or_none(reference_deploy_now_pct, 2) if reference_budget_plan else None,
            "ReferenceBudgetDeployNowVND": reference_deploy_now_vnd if reference_budget_plan else None,
            "ReferenceBudgetDeployNowQuantity": int(reference_deploy_now_qty) if reference_budget_plan else 0,
            "NoChaseAbove": no_chase_above,
            "InvalidationBelow": invalidation_below,
            "BreakoutConfirmAbove": breakout_confirm_above,
        },
        "ReferenceBudgetPlan": reference_budget_plan,
        "BudgetPlan": budget_plan,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{ticker}_ml_deep_dive.json"
    md_path = output_dir / f"{ticker}_ml_deep_dive.md"
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    report["OutputJSON"] = _display_path(json_path)
    report["OutputMarkdown"] = _display_path(md_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deep ML + research synthesis report for one ticker from the current notebook artifacts."
    )
    parser.add_argument("--ticker", required=True, help="Ticker to analyse.")
    parser.add_argument("--budget-vnd", type=int, default=DEFAULT_BUDGET_VND, help="Budget assumption used for sizing output.")
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    report = build_deep_dive(
        ticker=args.ticker,
        budget_vnd=args.budget_vnd,
        universe_csv=args.universe_csv,
        analysis_dir=args.analysis_dir,
        research_dir=args.research_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
