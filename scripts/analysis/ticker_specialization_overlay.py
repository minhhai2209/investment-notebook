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
    score = 0.0
    regime = "generic"
    action_bias = "neutral"

    if best_timing_edge is not None:
        signals.append(f"best timing {best_timing_edge:+.2f}%")
    if t10_edge is not None:
        signals.append(f"T+10 {t10_edge:+.2f}%")
    if burst_age is not None:
        signals.append(f"burst age D+{int(round(burst_age))}")

    if archetype == "momentum_high_beta":
        regime = "fresh_burst_distribution"
        action_bias = "khong_duoi_burst_cho_deep_pullback"
        if burst_age is not None and burst_age <= 1:
            score -= 2.0
            signals.append("burst còn rất mới")
        if burst_age is not None and 2 <= burst_age <= 4:
            regime = "post_burst_t25_supply"
            action_bias = "ton_trong_cung_t25_trim_chu_dong"
            score -= 1.0
            signals.append("đang ở cửa sổ cung T+2.5")
        if execution_bias == "distribution":
            score -= 2.0
            signals.append("tape đang phân phối")
        if burst_execution_bias == "failed_day2_followthrough":
            score -= 2.0
            signals.append("fail follow-through sau burst")
        if burst_execution_bias == "respect_t25_supply":
            regime = "post_burst_t25_supply"
            action_bias = "ton_trong_cung_t25_trim_chu_dong"
            score -= 2.0
            signals.append("phải tôn trọng cung T+2.5")
        if next_day_positive_rate is not None and next_day_positive_rate >= 65.0:
            signals.append(f"xác suất tăng ngày kế {next_day_positive_rate:.1f}%")
        if next_day_strong_rate is not None and next_day_strong_rate >= 40.0:
            signals.append(f"xác suất tăng mạnh ngày kế {next_day_strong_rate:.1f}%")
        if t10_edge is not None and t10_edge < 0.0:
            score -= 2.0
        if avg_three_day_drawdown is not None and avg_three_day_drawdown <= -2.0:
            score -= 1.0
            signals.append(f"burst drawdown 3 ngày {avg_three_day_drawdown:.2f}%")

    elif archetype == "cyclical_beta":
        regime = "cycle_pullback_build"
        action_bias = "uu_tien_add_on_pullback_khong_duoi_breakout"
        if best_timing_edge is not None:
            if best_timing_edge >= 3.0:
                score += 3.0
                signals.append("timing chu kỳ đủ mạnh")
            elif best_timing_edge > 0.0:
                score += 1.0
        if t10_edge is not None and t10_edge >= 0.0:
            score += 1.0
        if burst_age is not None and burst_age >= 10:
            score += 1.0
            signals.append("đã qua pha burst đầu")
        if next_day_positive_rate is not None and next_day_positive_rate >= 70.0:
            score += 1.0
        if avg_three_day_drawdown is not None and avg_three_day_drawdown > -1.0:
            score += 1.0
        if burst_sample_count is not None and burst_sample_count < 5:
            signals.append("mẫu burst lịch sử còn ít")

    elif archetype == "quality_trend":
        regime = "trend_persistence_pullback_add"
        action_bias = "giu_trend_add_co_chon_loc"
        if best_timing_edge is not None and best_timing_edge > 0.0:
            score += 1.0
        if t10_edge is not None and t10_edge > 0.0:
            score += 2.0
        if burst_age is not None and burst_age >= 10:
            score += 1.0
            signals.append("trend đang đứng ngoài pha burst nóng")
        if next_day_positive_rate is not None and next_day_positive_rate >= 75.0:
            score += 1.0
        if avg_three_day_drawdown is not None and avg_three_day_drawdown > -1.0:
            score += 1.0

    elif archetype == "special_situation":
        regime = "event_swing_only"
        action_bias = "chi_hop_swing_ngan_mua_sau_trim_hung_phan"
        if best_timing_edge is not None and best_timing_edge >= 2.0:
            score += 2.0
        if t10_edge is not None and t10_edge < 0.0:
            score -= 3.0
            signals.append("khung giữ dài không đẹp")
        if burst_age is not None and burst_age <= 10:
            score -= 1.0
            signals.append("vẫn còn trong vùng hậu burst")
        if third_day_negative_rate is not None and third_day_negative_rate >= 35.0:
            score -= 1.0
            signals.append(f"xác suất âm lại T+3 {third_day_negative_rate:.1f}%")
        if avg_three_day_drawdown is not None and avg_three_day_drawdown <= -2.0:
            score -= 1.0
            signals.append(f"drawdown burst 3 ngày {avg_three_day_drawdown:.2f}%")

    else:
        regime = "generic"
        action_bias = "neutral"
        if best_timing_edge is not None:
            score += _clamp(best_timing_edge * 0.3, -2.0, 2.0)
        if t10_edge is not None:
            score += _clamp(t10_edge * 0.2, -2.0, 2.0)

    score = _clamp(score, -6.0, 6.0)

    headline = None
    if archetype == "momentum_high_beta":
        if regime == "post_burst_t25_supply":
            headline = (
                f"{ticker} đang ở nhịp hậu burst dễ gặp cung T+2.5, nên đọc như event tape hơn là trend bền"
            )
        else:
            headline = (
                f"{ticker} đang ở pha burst rất mới, hợp đọc như setup tốc độ chứ không phải trend sạch"
            )
    elif archetype == "cyclical_beta":
        headline = (
            f"{ticker} hợp đánh theo nhịp chu kỳ và pullback hơn là chase breakout"
        )
    elif archetype == "quality_trend":
        headline = (
            f"{ticker} phù hợp logic giữ trend và add có chọn lọc ở pullback sạch"
        )
    elif archetype == "special_situation":
        headline = (
            f"{ticker} nên đọc như special situation, ưu tiên swing ngắn thay vì hold máy móc"
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
    overlay["OverlayScore"] = int(round(score))
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
