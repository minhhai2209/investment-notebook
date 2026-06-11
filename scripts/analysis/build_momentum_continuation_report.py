from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES_JSON = REPO_ROOT / "out" / "analysis" / "candidates" / "candidate_watchlist_full.json"
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "out" / "universe.csv"
DEFAULT_MARKET_SUMMARY_JSON = REPO_ROOT / "out" / "market_summary.json"
DEFAULT_EVENTS_JSON = REPO_ROOT / "config" / "market_events.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis" / "momentum"
DEFAULT_TICKERS = ""


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


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe snapshot not found: {path}")
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        [
            "Ticker",
            "Last",
            "ChangePct",
            "RSI14",
            "DistSMA20Pct",
            "Ret5d",
            "Ret20d",
            "Ret20dVsIndex",
            "Ret60dVsIndex",
        ],
        "Universe snapshot",
    )
    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    return frame


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_snapshot_date(value: Any) -> date:
    if value is None:
        return date.today()
    raw = str(value).strip()
    if not raw:
        return date.today()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return date.fromisoformat(raw[:10])


def _parse_tickers(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    tickers: List[str] = []
    for part in parts:
        ticker = str(part).strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _load_events(path: Path, snapshot_date: date, lookahead_days: int) -> tuple[List[Dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("Events", payload if isinstance(payload, list) else [])
    if not isinstance(events, list):
        raise ValueError(f"Events file must contain an Events list: {path}")

    scoped: List[Dict[str, Any]] = []
    until = snapshot_date + timedelta(days=lookahead_days)
    for event in events:
        if not isinstance(event, Mapping):
            continue
        raw_date = str(event.get("Date", "")).strip()
        if not raw_date:
            continue
        event_date = date.fromisoformat(raw_date[:10])
        if snapshot_date <= event_date <= until:
            row = dict(event)
            row["DaysAway"] = (event_date - snapshot_date).days
            row["BusinessDaysAway"] = _business_days_between(snapshot_date, event_date)
            scoped.append(row)
    scoped.sort(key=lambda item: (item.get("Date", ""), str(item.get("Label", ""))))
    return scoped, True


def _business_days_between(start: date, end: date) -> int:
    days = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor += timedelta(days=1)
    return days


def _event_risk(events: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if not events:
        return "none", "không có sự kiện local trong cửa sổ theo dõi"
    closed = [event for event in events if bool(event.get("MarketClosed"))]
    high = [
        event
        for event in events
        if str(event.get("Impact", "")).lower() in {"high", "cao"}
        or str(event.get("Risk", "")).lower() in {"high", "cao"}
    ]
    nearest = min(events, key=lambda event: int(event.get("BusinessDaysAway") or event.get("DaysAway") or 999))
    label = str(nearest.get("Label") or nearest.get("Name") or nearest.get("Date"))
    distance = nearest.get("BusinessDaysAway")
    if closed and (closed[0].get("BusinessDaysAway") or 999) <= 5:
        return "holiday_gap", f"sắp có nghỉ/gap lịch: {label} trong ~{distance} phiên làm việc"
    if high and (high[0].get("BusinessDaysAway") or 999) <= 5:
        return "event_risk", f"sắp có sự kiện rủi ro: {label} trong ~{distance} phiên làm việc"
    return "watch", f"có sự kiện cần theo dõi: {label} trong ~{distance} phiên làm việc"


def _score_row(candidate: Mapping[str, Any], universe_row: Mapping[str, Any], market: Mapping[str, Any]) -> tuple[float, List[str]]:
    reasons: List[str] = []

    timing_edge = _safe_float(candidate.get("BestTimingNetEdgePct")) or 0.0
    t10_edge = _safe_float(candidate.get("T10NetEdgePct")) or 0.0
    t1_close = _safe_float(candidate.get("ForecastCloseRetPctT1")) or 0.0
    peak_ret = _safe_float(candidate.get("BestTimingPeakRetPct"))
    close_ret = _safe_float(candidate.get("BestTimingCloseRetPct"))
    peak_mae = _safe_float(candidate.get("BacktestPeakRetMAEPct"))
    close_mae = _safe_float(candidate.get("BacktestCloseMAEPct"))
    conservative_peak = peak_ret - peak_mae if peak_ret is not None and peak_mae is not None else peak_ret
    conservative_close = close_ret - close_mae if close_ret is not None and close_mae is not None else close_ret
    score_candidates = [
        value
        for value in (conservative_close, conservative_peak, close_ret, timing_edge, t10_edge, t1_close)
        if value is not None
    ]
    score = max(score_candidates) if score_candidates else 0.0

    reasons.append(f"timing edge {timing_edge:.2f}%")
    if conservative_peak is not None:
        reasons.append(f"conservative peak {conservative_peak:.2f}%")
    if conservative_close is not None:
        reasons.append(f"conservative close {conservative_close:.2f}%")
    if t10_edge < 0:
        reasons.append(f"T+10 âm {t10_edge:.2f}%")
    if str(candidate.get("ForecastConsistencyStatus") or "") == "conflict":
        summary = str(candidate.get("ForecastConsistencySummary") or "").strip()
        reasons.append(f"forecast conflict {summary}" if summary else "forecast conflict timing vs OHLC")

    zone_status = str(candidate.get("ZoneStatus") or "")
    if zone_status == "inside":
        reasons.append("giá đang trong vùng mua")
    elif zone_status == "above":
        gap = _safe_float(candidate.get("ZoneGapPct")) or 0.0
        reasons.append(f"giá đang trên vùng mua {gap:.2f}%")
    elif zone_status == "below":
        reasons.append("giá dưới vùng mua, cần xác nhận hồi")

    change_pct = _safe_float(universe_row.get("ChangePct"))

    if change_pct is not None and change_pct <= -2.0:
        reasons.append(f"snapshot intraday change {change_pct:.2f}%")

    breadth_1d = _safe_float(market.get("BreadthPositive1dPct"))
    adv_ratio = _safe_float(market.get("AdvanceDeclineRatio"))
    index_pos20 = _safe_float(market.get("IndexRangePos20"))
    if breadth_1d is not None and breadth_1d < 35:
        reasons.append(f"market breadth positive {breadth_1d:.1f}%")
    if index_pos20 is not None and index_pos20 > 0.9:
        reasons.append(f"VNINDEX range position 20d {index_pos20:.2f}")

    return round(score, 3), reasons


def _direction_from_score(score: float, candidate: Mapping[str, Any]) -> str:
    if str(candidate.get("ForecastConsistencyStatus") or "") == "conflict":
        return "model conflict / chờ xác nhận"
    close_ret = _safe_float(candidate.get("BestTimingCloseRetPct"))
    close_mae = _safe_float(candidate.get("BacktestCloseMAEPct"))
    conservative_close = close_ret - close_mae if close_ret is not None and close_mae is not None else close_ret
    t10_edge = _safe_float(candidate.get("T10NetEdgePct"))
    t1_close = _safe_float(candidate.get("ForecastCloseRetPctT1"))
    if conservative_close is not None and conservative_close > 0.0:
        return "tăng tiếp"
    if close_ret is None and t10_edge is not None and t10_edge < 0.0 and t1_close is not None and t1_close < 0.0:
        return "giảm hoặc rủi ro giảm"
    if close_ret is not None and close_ret > 0.0 and score > 0.0:
        return "đi ngang nghiêng tăng"
    if close_ret is None and score > 0.0 and (t10_edge is None or t10_edge >= 0.0):
        return "đi ngang nghiêng tăng"
    if score > 0.0:
        return "đi ngang / còn rung"
    if t10_edge is not None and t10_edge > 0.0:
        return "đi ngang / còn rung"
    return "giảm hoặc rủi ro giảm"


def _urgency(candidate: Mapping[str, Any], direction: str, event_risk_code: str) -> str:
    zone_status = str(candidate.get("ZoneStatus") or "")
    decision = str(candidate.get("Decision") or "")
    if str(candidate.get("ForecastConsistencyStatus") or "") == "conflict":
        return "model action: timing/OHLC conflict, wait for path confirmation"
    if decision == "không_mua" or direction == "giảm hoặc rủi ro giảm":
        return "model action: no new buy"
    if zone_status == "above":
        return "model action: wait for artifact zone"
    if direction == "tăng tiếp":
        if event_risk_code in {"holiday_gap", "event_risk"}:
            return "model action: positive forecast but event-risk size cap required"
        return "model action: eligible only inside artifact zone"
    if direction == "đi ngang nghiêng tăng":
        return "model action: marginal positive, require ladder sizing"
    return "model action: require better forecast/zone confirmation"


def _interpretation(candidate: Mapping[str, Any], direction: str, urgency: str) -> str:
    ticker = candidate.get("Ticker")
    no_chase = candidate.get("NoChaseAbove")
    zone_low = candidate.get("PreferredBuyZoneLow")
    zone_high = candidate.get("PreferredBuyZoneHigh")
    interpretation = (
        f"{ticker}: {direction}; {urgency}. "
        f"Artifact zone {zone_low}-{zone_high}, no-chase artifact {no_chase}."
    )
    if str(candidate.get("ForecastConsistencyStatus") or "") == "conflict":
        summary = str(candidate.get("ForecastConsistencySummary") or "").strip()
        if summary:
            interpretation += f" Forecast conflict: {summary}."
    return interpretation


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Momentum Continuation")
    lines.append("")
    lines.append(f"- SnapshotDate: `{report['SnapshotDate']}`")
    lines.append(f"- Event risk: `{report['EventRisk']}` - {report['EventRiskSummary']}")
    lines.append("- Live news: phải check riêng ngay lúc trả lời, không coi report này là lớp news live.")
    lines.append("")
    for row in report["Rows"]:
        lines.append(
            "- "
            f"{row['Ticker']} | `{row['DirectionCall']}` | urgency `{row['Urgency']}` | "
            f"score `{row['ContinuationScore']}` | price `{row['CurrentPrice']}` | "
            f"zone `{row['PreferredBuyZoneLow']} - {row['PreferredBuyZoneHigh']}` | "
            f"no-chase `>{row['NoChaseAbove']}` | {row['Interpretation']}"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_momentum_continuation_report(
    *,
    candidates_json: Path,
    universe_csv: Path,
    market_summary_json: Path,
    events_json: Path,
    output_dir: Path,
    tickers: Sequence[str],
    event_lookahead_days: int = 14,
) -> Dict[str, Any]:
    candidates = _load_json(candidates_json, "Candidate watchlist")
    universe_df = _load_universe(universe_csv)
    market_summary = _load_json(market_summary_json, "Market summary")
    snapshot_date = _parse_snapshot_date(candidates.get("SnapshotDate") or market_summary.get("GeneratedAt"))
    events, events_loaded = _load_events(events_json, snapshot_date, event_lookahead_days)
    event_risk_code, event_risk_summary = _event_risk(events)

    universe_by_ticker = {row["Ticker"]: row for row in universe_df.to_dict(orient="records")}
    candidate_by_ticker = {
        _normalise_ticker(row.get("Ticker")): row
        for row in candidates.get("Rows", [])
        if _normalise_ticker(row.get("Ticker"))
    }
    parsed_tickers = _parse_tickers(tickers)
    if not parsed_tickers:
        parsed_tickers = list(candidate_by_ticker.keys())
    wanted = set(parsed_tickers)

    rows: List[Dict[str, Any]] = []
    for ticker in sorted(wanted):
        if ticker not in candidate_by_ticker:
            raise ValueError(f"Ticker missing from candidate watchlist: {ticker}")
        if ticker not in universe_by_ticker:
            raise ValueError(f"Ticker missing from universe snapshot: {ticker}")
        candidate = candidate_by_ticker[ticker]
        universe_row = universe_by_ticker[ticker]
        score, reasons = _score_row(candidate, universe_row, market_summary)
        direction = _direction_from_score(score, candidate)
        urgency = _urgency(candidate, direction, event_risk_code)
        row = {
            "Ticker": ticker,
            "Decision": candidate.get("Decision"),
            "CurrentPrice": _round_or_none(candidate.get("CurrentPrice")),
            "PreferredBuyZoneLow": candidate.get("PreferredBuyZoneLow"),
            "PreferredBuyZoneHigh": candidate.get("PreferredBuyZoneHigh"),
            "NoChaseAbove": candidate.get("NoChaseAbove"),
            "InvalidationBelow": candidate.get("InvalidationBelow"),
            "ContinuationScore": score,
            "DirectionCall": direction,
            "Urgency": urgency,
            "ReasonBullets": reasons,
            "Interpretation": _interpretation(candidate, direction, urgency),
            "BestTimingWindow": candidate.get("BestTimingWindow"),
            "BestTimingNetEdgePct": candidate.get("BestTimingNetEdgePct"),
            "T10NetEdgePct": candidate.get("T10NetEdgePct"),
            "ForecastCloseRetPctT1": candidate.get("ForecastCloseRetPctT1"),
            "ForecastCandleBias": candidate.get("ForecastCandleBias"),
            "ForecastConsistencyStatus": candidate.get("ForecastConsistencyStatus"),
            "ForecastConsistencySummary": candidate.get("ForecastConsistencySummary"),
            "ForecastConsistencyGapPct": candidate.get("ForecastConsistencyGapPct"),
            "ReferenceBudgetPlanSummary": candidate.get("ReferenceBudgetPlanSummary"),
            "SessionBuyPlanSummary": candidate.get("SessionBuyPlanSummary"),
            "SpecializedRegime": candidate.get("SpecializedRegime"),
            "SpecializedActionBias": candidate.get("SpecializedActionBias"),
            "ChangePct": _round_or_none(universe_row.get("ChangePct")),
            "RSI14": _round_or_none(universe_row.get("RSI14")),
            "DistSMA20Pct": _round_or_none(universe_row.get("DistSMA20Pct")),
            "Ret5d": _round_or_none(universe_row.get("Ret5d")),
            "Ret20dVsIndex": _round_or_none(universe_row.get("Ret20dVsIndex")),
            "Ret60dVsIndex": _round_or_none(universe_row.get("Ret60dVsIndex")),
        }
        rows.append(row)

    direction_order = {
        "tăng tiếp": 0,
        "đi ngang nghiêng tăng": 1,
        "model conflict / chờ xác nhận": 2,
        "đi ngang / còn rung": 3,
        "giảm hoặc rủi ro giảm": 4,
    }
    rows.sort(
        key=lambda item: (
            direction_order.get(str(item["DirectionCall"]), 9),
            -float(item["ContinuationScore"]),
            item["Ticker"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "momentum_continuation.csv"
    json_path = output_dir / "momentum_continuation.json"
    md_path = output_dir / "momentum_continuation.md"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = {
        "SchemaVersion": 1,
        "SnapshotDate": candidates.get("SnapshotDate") or market_summary.get("GeneratedAt"),
        "Tickers": sorted(wanted),
        "EventCalendarLoaded": events_loaded,
        "EventRisk": event_risk_code,
        "EventRiskSummary": event_risk_summary,
        "UpcomingEvents": events,
        "LiveNewsRequired": True,
        "Rows": rows,
        "OutputCSV": _display_path(csv_path),
        "OutputJSON": _display_path(json_path),
        "OutputMarkdown": _display_path(md_path),
    }
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify whether focused tickers still have continuation, sideways, or downside risk."
    )
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--market-summary-json", type=Path, default=DEFAULT_MARKET_SUMMARY_JSON)
    parser.add_argument("--events-json", type=Path, default=DEFAULT_EVENTS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--event-lookahead-days", type=int, default=14)
    args = parser.parse_args(argv)

    report = build_momentum_continuation_report(
        candidates_json=args.candidates_json,
        universe_csv=args.universe_csv,
        market_summary_json=args.market_summary_json,
        events_json=args.events_json,
        output_dir=args.output_dir,
        tickers=_parse_tickers(args.tickers),
        event_lookahead_days=args.event_lookahead_days,
    )
    print(json.dumps(_json_safe(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
