from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _base_overlay(state: Mapping[str, Any] | None) -> Dict[str, Any]:
    return {
        "Archetype": state.get("Archetype") if state else None,
        "PreferredHoldWindow": state.get("PreferredHoldWindow") if state else None,
        "Regime": None,
        "ActionBias": None,
        "OverlayScore": 0,
        "Summary": None,
        "Signals": [],
        "Metrics": {},
    }


def summarise_specialized_ticker_setup(
    ticker: str,
    state: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    overlay = _base_overlay(state)
    if not state:
        return overlay

    archetype = str(state.get("Archetype") or "").strip().lower()
    best_timing_edge = _safe_float(state.get("BestTimingNetEdgePct"))
    t10_edge = _safe_float(state.get("T10NetEdgePct"))
    burst_age = _safe_float(state.get("LatestBurstSignalAge"))
    next_day_positive_rate = _safe_float(state.get("BurstNextDayPositiveRate"))
    next_day_strong_rate = _safe_float(state.get("BurstNextDayStrongRate"))
    third_day_negative_rate = _safe_float(state.get("BurstThirdDayNegativeRate"))
    avg_three_day_drawdown = _safe_float(state.get("BurstAvgThreeDayDrawdownPct"))
    burst_sample_count = _safe_float(state.get("BurstSampleCount"))
    execution_bias = str(state.get("ExecutionBias") or "").strip().lower()
    burst_execution_bias = str(state.get("BurstExecutionBias") or "").strip().lower()
    execution_note = state.get("ExecutionNote")

    signals: List[str] = []
    regime = "generic"
    action_bias = "neutral"

    if best_timing_edge is not None:
        signals.append(f"best timing {best_timing_edge:+.2f}%")
    if t10_edge is not None:
        signals.append(f"T+10 {t10_edge:+.2f}%")
    if burst_age is not None:
        signals.append(f"burst age D+{int(round(burst_age))}")

    if archetype == "momentum_high_beta":
        regime = "momentum_high_beta"
        action_bias = "read_ml_forecast_not_overlay_score"
        if burst_age is not None and burst_age <= 1:
            signals.append("burst còn rất mới")
        if burst_age is not None and 2 <= burst_age <= 4:
            regime = "post_burst_t25_supply"
            signals.append("đang ở cửa sổ cung T+2.5")
        if execution_bias == "distribution":
            signals.append("tape đang phân phối")
        if burst_execution_bias == "failed_day2_followthrough":
            signals.append("fail follow-through sau burst")
        if burst_execution_bias == "respect_t25_supply":
            regime = "post_burst_t25_supply"
            signals.append("phải tôn trọng cung T+2.5")
        if next_day_positive_rate is not None and next_day_positive_rate >= 65.0:
            signals.append(f"xác suất tăng ngày kế {next_day_positive_rate:.1f}%")
        if next_day_strong_rate is not None and next_day_strong_rate >= 40.0:
            signals.append(f"xác suất vượt ngưỡng positive ngày kế {next_day_strong_rate:.1f}%")
        if avg_three_day_drawdown is not None and avg_three_day_drawdown <= -2.0:
            signals.append(f"burst drawdown 3 ngày {avg_three_day_drawdown:.2f}%")

    elif archetype == "cyclical_beta":
        regime = "cycle_pullback_build"
        action_bias = "read_ml_forecast_not_overlay_score"
        if burst_age is not None and burst_age >= 10:
            signals.append("đã qua pha burst đầu")
        if burst_sample_count is not None and burst_sample_count < 5:
            signals.append("mẫu burst lịch sử còn ít")

    elif archetype == "quality_trend":
        regime = "trend_persistence_pullback_add"
        action_bias = "read_ml_forecast_not_overlay_score"
        if burst_age is not None and burst_age >= 10:
            signals.append("trend ngoài cửa sổ burst gần nhất")

    elif archetype == "special_situation":
        regime = "event_swing_only"
        action_bias = "read_ml_forecast_not_overlay_score"
        if burst_age is not None and burst_age <= 10:
            signals.append("vẫn còn trong vùng hậu burst")
        if third_day_negative_rate is not None and third_day_negative_rate >= 35.0:
            signals.append(f"xác suất âm lại T+3 {third_day_negative_rate:.1f}%")
        if avg_three_day_drawdown is not None and avg_three_day_drawdown <= -2.0:
            signals.append(f"drawdown burst 3 ngày {avg_three_day_drawdown:.2f}%")

    headline = None
    if archetype == "momentum_high_beta":
        if regime == "post_burst_t25_supply":
            headline = (
                f"{ticker} đang ở nhịp hậu burst; quyết định mua/bán phải lấy từ model riêng của mã, không lấy từ overlay"
            )
        else:
            headline = (
                f"{ticker} là momentum high beta; overlay chỉ mô tả trạng thái, không chấm điểm quyết định"
            )
    elif archetype == "cyclical_beta":
        headline = (
            f"{ticker}: cyclical_beta archetype; overlay chỉ mô tả context, không chấm điểm quyết định"
        )
    elif archetype == "quality_trend":
        headline = (
            f"{ticker}: quality_trend archetype; action phải lấy từ forecast/zone/action-sizing"
        )
    elif archetype == "special_situation":
        headline = (
            f"{ticker}: special_situation archetype; action phải lấy từ forecast/zone/action-sizing"
        )

    summary_parts: List[str] = []
    if headline:
        summary_parts.append(headline)
    if best_timing_edge is not None or t10_edge is not None:
        timing_bits: List[str] = []
        if best_timing_edge is not None:
            timing_bits.append(f"best timing {best_timing_edge:+.2f}%")
        if t10_edge is not None:
            timing_bits.append(f"T+10 {t10_edge:+.2f}%")
        summary_parts.append(", ".join(timing_bits))
    if burst_age is not None:
        summary_parts.append(f"burst age D+{int(round(burst_age))}")
    if execution_note:
        summary_parts.append(str(execution_note))

    overlay["Regime"] = regime
    overlay["ActionBias"] = action_bias
    overlay["OverlayScore"] = 0
    overlay["Summary"] = "; ".join(summary_parts[:4]) if summary_parts else None
    overlay["Signals"] = signals[:8]
    overlay["Metrics"] = {
        "BestTimingNetEdgePct": _round_or_none(best_timing_edge, 2),
        "T10NetEdgePct": _round_or_none(t10_edge, 2),
        "BurstSampleCount": _round_or_none(burst_sample_count, 0),
        "BurstNextDayPositiveRate": _round_or_none(next_day_positive_rate, 2),
        "BurstNextDayStrongRate": _round_or_none(next_day_strong_rate, 2),
        "BurstThirdDayNegativeRate": _round_or_none(third_day_negative_rate, 2),
        "BurstAvgThreeDayDrawdownPct": _round_or_none(avg_three_day_drawdown, 2),
        "LatestBurstSignalAge": _round_or_none(burst_age, 0),
        "ExecutionBias": state.get("ExecutionBias"),
        "BurstExecutionBias": state.get("BurstExecutionBias"),
    }
    return overlay
