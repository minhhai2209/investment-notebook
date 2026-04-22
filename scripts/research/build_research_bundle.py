from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "out" / "universe.csv"
DEFAULT_MARKET_SUMMARY_JSON = REPO_ROOT / "out" / "market_summary.json"
DEFAULT_SECTOR_SUMMARY_CSV = REPO_ROOT / "out" / "sector_summary.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_INTRADAY_DIR = REPO_ROOT / "out" / "data" / "intraday_5m"
DEFAULT_RESEARCH_DIR = REPO_ROOT / "research"
DEFAULT_HUMAN_NOTES_PATH = REPO_ROOT / "human_notes.md"

REQUIRED_UNIVERSE_COLUMNS = [
    "Ticker",
    "EngineRunAt",
    "Sector",
    "Last",
    "Ref",
    "ChangePct",
    "SMA20",
    "SMA50",
    "SMA200",
    "EMA20",
    "RSI14",
    "ATR14Pct",
    "Vol20Pct",
    "Vol60Pct",
    "MACD",
    "MACDSignal",
    "Beta60_Index",
    "Corr60_Index",
    "Corr20_Index",
    "Beta20_Index",
    "Ret5d",
    "Ret20d",
    "Ret60d",
    "Ret20dVsIndex",
    "Ret60dVsIndex",
    "Ret20dVsSector",
    "Ret60dVsSector",
    "Pos52wPct",
    "High52w",
    "Low52w",
    "ADTV20_shares",
    "IntradayVol_shares",
    "IntradayValue_kVND",
    "IntradayPctADV20",
    "ValidBid1",
    "ValidAsk1",
    "GridBelow_T1",
    "GridBelow_T2",
    "GridBelow_T3",
    "GridAbove_T1",
    "GridAbove_T2",
    "GridAbove_T3",
    "DistSMA20Pct",
    "PE_fwd",
    "PB",
    "ROE",
    "PositionMarketValue_kVND",
    "PositionWeightPct",
    "EnginePortfolioMarketValue_kVND",
]
REQUIRED_SECTOR_COLUMNS = [
    "Sector",
    "SectorBreadthAboveSMA20Pct",
    "SectorBreadthAboveSMA50Pct",
    "SectorBreadthPositive5dPct",
    "SectorMedianRet20dVsIndex",
    "SectorMedianRet60dVsIndex",
]
REQUIRED_MARKET_KEYS = [
    "BreadthAboveSMA20Pct",
    "BreadthAboveSMA50Pct",
    "BreadthPositive5dPct",
    "NewHigh20Pct",
    "AdvanceDeclineRatio",
    "VNINDEX_ATR14PctRank",
]
REQUIRED_TIMING_COLUMNS = [
    "Ticker",
    "ForecastWindow",
    "PredPeakRetPct",
    "PredPeakDay",
    "PredDrawdownPct",
    "PredCloseRetPct",
    "PredRewardRisk",
    "PredTradeScore",
    "PredNetEdgePct",
    "PredCapitalEfficiencyPctPerDay",
]
REQUIRED_LADDER_COLUMNS = [
    "Ticker",
    "EntryScoreRank",
    "LimitPrice",
    "EntryScore",
    "BestTimingWindow",
    "BestTimingNetEdgePct",
    "BestTimingCloseRetPct",
    "BestTimingRewardRisk",
    "CycleNetEdgePct",
    "CycleRewardRisk",
    "FillScoreComposite",
]

TACTICAL_LADDER_ANCHORS = {
    "valid_bid1",
    "grid_below_t1",
    "grid_below_t2",
    "grid_below_t3",
}
HISTORICAL_LADDER_ANCHORS = {
    "forecast_low_t1",
    "range_low_blend_t5",
    "range_low_blend_t10",
    "cycle_drawdown",
    "atr_1x_below",
    "atr_1_5x_below",
}
REQUIRED_OHLC_COLUMNS = [
    "Ticker",
    "ForecastOpen",
    "ForecastHigh",
    "ForecastLow",
    "ForecastClose",
    "ForecastCloseRetPct",
    "ForecastRangePct",
    "ForecastCandleBias",
]
REQUIRED_PLAYBOOK_COLUMNS = [
    "Ticker",
    "StrategyFamily",
    "StrategyLabel",
    "LatestSignal",
    "LatestSignalDate",
    "TestScore",
    "RobustScore",
    "TestAvgReturnPct",
    "TestWorstDrawdownPct",
]
REQUIRED_CYCLE_BEST_COLUMNS = [
    "Ticker",
    "HorizonMonths",
    "ForecastWindow",
    "Variant",
    "Model",
    "SelectionScore",
    "PredPeakRetPct",
    "PredPeakDays",
    "PredPeakPrice",
    "PredDrawdownPct",
    "PredDrawdownPrice",
]
REQUIRED_RANGE_COLUMNS = [
    "Ticker",
    "ForecastWindow",
    "PredLowRetPct",
    "PredMidRetPct",
    "PredHighRetPct",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
]
REQUIRED_INTRADAY_REPORT_COLUMNS = [
    "Ticker",
    "SnapshotTimeBucket",
    "PredLowRetPct",
    "PredMidRetPct",
    "PredHighRetPct",
    "SelectionScore",
]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    if frame.empty:
        raise RuntimeError(f"{label} is empty: {path}")
    return frame


def _read_optional_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    return frame


def _read_json(path: Path, required_keys: Iterable[str], label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")
    return payload


def _normalise_ticker(value: object) -> str:
    return str(value).strip().upper()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _fmt_pct(value: float | None, digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "không có dữ liệu"
    if signed:
        return f"{value:+.{digits}f}%"
    return f"{value:.{digits}f}%"


def _fmt_price(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "không có dữ liệu"
    return f"{value:.{digits}f}"


def _fmt_ratio(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "không có dữ liệu"
    return f"{value:.{digits}f}"


def _bool_text(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _load_existing_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_human_notes(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    overlays: Dict[str, Dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        ticker_match = re.search(r"`([A-Z0-9]+)`", line)
        if ticker_match is None:
            continue
        ticker = _normalise_ticker(ticker_match.group(1))
        if not ticker:
            continue
        target_match = re.search(r"target giá\s+`?(\d+(?:\.\d+)?)`?", line, flags=re.IGNORECASE)
        year_match = re.search(r"năm\s+(\d{4})", line, flags=re.IGNORECASE)
        overlay = {
            "raw_line": line.lstrip("-").strip(),
            "target_price": float(target_match.group(1)) if target_match else None,
            "target_year": int(year_match.group(1)) if year_match else None,
        }
        overlays[ticker] = overlay
    return overlays


def _market_regime(payload: Mapping[str, Any]) -> str:
    breadth20 = _to_float(payload.get("BreadthAboveSMA20Pct")) or 0.0
    breadth50 = _to_float(payload.get("BreadthAboveSMA50Pct")) or 0.0
    breadth5d = _to_float(payload.get("BreadthPositive5dPct")) or 0.0
    new_high = _to_float(payload.get("NewHigh20Pct")) or 0.0
    if breadth20 >= 65.0 and breadth50 >= 55.0 and breadth5d >= 55.0 and new_high >= 15.0:
        return "risk_on"
    if breadth20 <= 35.0 and breadth50 <= 30.0 and breadth5d <= 40.0:
        return "risk_off"
    return "balanced"


def _market_regime_vi(regime: str) -> str:
    return {
        "risk_on": "risk-on",
        "risk_off": "risk-off",
        "balanced": "cân bằng",
    }.get(regime, regime)


def _sector_regime(row: Mapping[str, Any]) -> str:
    ticker_count = _to_float(row.get("TickerCount"))
    if ticker_count is not None and ticker_count <= 1.0:
        return "mixed"
    breadth20 = _to_float(row.get("SectorBreadthAboveSMA20Pct")) or 0.0
    breadth50 = _to_float(row.get("SectorBreadthAboveSMA50Pct")) or 0.0
    ret20 = _to_float(row.get("SectorMedianRet20dVsIndex")) or 0.0
    if breadth20 >= 60.0 and breadth50 >= 50.0 and ret20 >= 5.0:
        return "leader"
    if breadth20 <= 40.0 and breadth50 <= 35.0 and ret20 <= -3.0:
        return "laggard"
    return "mixed"


def _market_breadth_source_line(payload: Mapping[str, Any]) -> str:
    source = str(payload.get("BreadthSource") or "working_universe").strip().lower()
    breadth_count = _to_float(payload.get("BreadthUniverseTickerCount"))
    if source == "benchmark_basket":
        if breadth_count is not None:
            return f"nguồn breadth: benchmark basket `{int(breadth_count)}` mã"
        return "nguồn breadth: benchmark basket"
    if breadth_count is not None:
        return f"nguồn breadth: working universe `{int(breadth_count)}` mã"
    return "nguồn breadth: working universe"


def _classify_archetype(row: Mapping[str, Any]) -> str:
    sector = str(row.get("Sector", "")).strip().lower()
    roe = _to_float(row.get("ROE")) or 0.0
    pb = _to_float(row.get("PB")) or 0.0
    pe = _to_float(row.get("PE_fwd")) or 0.0
    vol20 = _to_float(row.get("Vol20Pct")) or 0.0
    atr_pct = _to_float(row.get("ATR14Pct")) or 0.0
    ret20_vs_idx = _to_float(row.get("Ret20dVsIndex")) or 0.0
    beta60 = _to_float(row.get("Beta60_Index")) or 0.0

    if "bất động sản" in sector and (roe <= 8.0 or pb <= 1.0) and (vol20 >= 3.0 or atr_pct >= 4.0):
        return "special_situation"
    if "tài chính" in sector and roe >= 18.0 and pb >= 1.5:
        return "quality_financial_trend"
    if "nguyên vật liệu" in sector and beta60 >= 0.8:
        return "cyclical_beta"
    if roe >= 18.0 and pe > 0 and pe <= 18.0 and vol20 <= 3.0:
        return "quality_trend"
    if ret20_vs_idx >= 10.0 and atr_pct >= 3.0:
        return "momentum_high_beta"
    return "balanced_swing"


def _archetype_template(archetype: str) -> Dict[str, Any]:
    templates: Dict[str, Dict[str, Any]] = {
        "special_situation": {
            "label": "special situation / turnaround",
            "style": "swing theo catalyst, mua sâu và trim vào hưng phấn",
            "hold_window": "3-10 phiên, chỉ kéo dài nếu follow-through sạch",
            "allow_add_on_strength": False,
            "allow_add_on_weakness": True,
            "profile_lines": [
                "Mã này không phù hợp với logic nắm giữ thụ động chờ tăng trưởng đều.",
                "Luận điểm dài hạn phụ thuộc nhiều hơn vào tiến độ xử lý bảng cân đối, tái cấu trúc, nguồn vốn và headline doanh nghiệp.",
                "Khi giá tăng mạnh, cung treo và lượng hàng trading thường xuất hiện sớm; chất lượng tape quan trọng hơn câu chuyện tổng quát.",
            ],
            "focus_lines": [
                "Tiến độ tái cấu trúc đang tiến triển thật hay chỉ dừng ở headline.",
                "Các nguồn vốn mới, bảo lãnh, tái cấu trúc nợ, chuyển đổi trái phiếu hay pha loãng mới.",
                "Dự án hoặc tài sản nào thực sự mở khóa dòng tiền trong 1-2 quý tới.",
                "Nhịp hồi hiện tại là rerating bền hơn hay chỉ là short squeeze nối tiếp.",
            ],
            "order_lines": [
                "Không chase open mạnh.",
                "Ưu tiên mua pullback sâu hoặc sau khi hấp thụ cung xong.",
                "Nếu không follow-through trong 3-5 phiên, phải hạ conviction.",
                "Có thể trim vào các nhịp kéo intraday nếu close quality yếu.",
            ],
        },
        "quality_financial_trend": {
            "label": "quality financial trend",
            "style": "ưu tiên pullback sạch, có thể add-on-strength khi tín hiệu bền",
            "hold_window": "2-8 tuần",
            "allow_add_on_strength": True,
            "allow_add_on_weakness": True,
            "profile_lines": [
                "Đây là nhóm tài chính có chất lượng lợi nhuận và ROE đủ tốt để theo dõi như một trend vehicle thay vì special situation.",
                "Điểm quan trọng là độ bền của xu hướng và chất lượng dòng tiền hơn là một headline đơn lẻ.",
                "Nếu thị trường thuận và mã giữ được nền giá, có thể cho size lớn hơn nhóm đầu cơ.",
            ],
            "focus_lines": [
                "Độ bền của tăng trưởng lợi nhuận và chất lượng vốn.",
                "Sức mạnh tương đối so với nhóm tài chính và VNINDEX.",
                "Dòng tiền ngoại/tự doanh có ủng hộ liên tục hay không.",
                "Chất lượng pullback: cạn cung hay chỉ là break tạm.",
            ],
            "order_lines": [
                "Ưu tiên mua ở pullback hoặc nền tích lũy chặt.",
                "Cho phép add-on-strength nếu playbook và timing đều ủng hộ.",
                "Không cần ép về ladder quá sâu nếu thị trường đang risk-on rõ.",
            ],
        },
        "cyclical_beta": {
            "label": "cyclical beta",
            "style": "đánh theo chu kỳ và nhịp trend, không ôm thụ động",
            "hold_window": "1-6 tuần tùy chu kỳ",
            "allow_add_on_strength": True,
            "allow_add_on_weakness": True,
            "profile_lines": [
                "Mã thuộc nhóm nhạy chu kỳ và thường phản ứng mạnh với pha risk-on/risk-off của thị trường.",
                "Khi xu hướng thuận, biên lợi nhuận thường đến từ continuation và reacceleration; khi chu kỳ gãy, downside có thể đến nhanh.",
                "Cần nhìn đồng thời diễn biến ngành, thị trường và tape của chính mã.",
            ],
            "focus_lines": [
                "Tín hiệu chu kỳ ngành và sức mạnh tương đối với chỉ số.",
                "Dòng tiền ngoại/tự doanh nếu có ảnh hưởng đáng kể.",
                "Khả năng giữ nền sau các nhịp tăng ngắn.",
                "Biên drawdown nếu chu kỳ đảo chiều sớm hơn kỳ vọng.",
            ],
            "order_lines": [
                "Có thể mua pullback hoặc add khi breakout giữ được close quality.",
                "Nếu mất relative strength so với ngành và chỉ số, nên hạ nhanh kỳ vọng.",
            ],
        },
        "quality_trend": {
            "label": "quality trend",
            "style": "ưu tiên giữ trend, add có chọn lọc ở pullback hoặc breakout chất lượng",
            "hold_window": "2-8 tuần",
            "allow_add_on_strength": True,
            "allow_add_on_weakness": True,
            "profile_lines": [
                "Đây là nhóm có chất lượng cơ bản và hành vi giá đủ tốt để vận hành như một trend compounder trong khung swing-trung hạn.",
                "Điểm quan trọng là không mua đuổi quá xa nền giá khi RSI và khoảng cách với SMA đã cao.",
            ],
            "focus_lines": [
                "Độ sạch của nhịp pullback.",
                "Khả năng duy trì relative strength.",
                "Chất lượng close sau các phiên breakout.",
            ],
            "order_lines": [
                "Ưu tiên vào ở pullback hoặc breakout giữ nền.",
                "Có thể cho phép add-on-strength nhưng vẫn cần quản lý khoảng cách tới nền.",
            ],
        },
        "momentum_high_beta": {
            "label": "momentum high beta",
            "style": "swing nhanh, ưu tiên tốc độ và close quality",
            "hold_window": "3-10 phiên",
            "allow_add_on_strength": True,
            "allow_add_on_weakness": False,
            "profile_lines": [
                "Mã đang vận hành như high-beta momentum play hơn là câu chuyện tích lũy chậm.",
                "Điểm mạnh là tốc độ; điểm yếu là đảo chiều cũng rất nhanh nếu mất follow-through.",
            ],
            "focus_lines": [
                "Close quality và relative strength từng phiên.",
                "Khả năng giữ thành quả sau các nhịp kéo.",
            ],
            "order_lines": [
                "Ưu tiên continuation có xác nhận.",
                "Tránh trung bình giá xuống khi tape chuyển xấu rõ.",
            ],
        },
        "balanced_swing": {
            "label": "balanced swing",
            "style": "đánh cân bằng theo pullback và vùng giá, không quá lệch breakout hay bắt đáy",
            "hold_window": "1-4 tuần",
            "allow_add_on_strength": False,
            "allow_add_on_weakness": True,
            "profile_lines": [
                "Mã chưa nghiêng rõ về compounder chất lượng hay special situation cực đoan.",
                "Do đó cách đánh phù hợp là swing cân bằng, để giá quyết định thay vì đặt thesis quá cứng.",
            ],
            "focus_lines": [
                "Vùng giá hỗ trợ/kháng cự gần.",
                "Timing ngắn hạn và playbook nào đang thắng.",
                "Độ lệch so với ngành và chỉ số.",
            ],
            "order_lines": [
                "Ưu tiên ladder theo giá trị kỳ vọng tốt nhất.",
                "Chỉ add mạnh khi nhiều lớp cùng đồng thuận.",
            ],
        },
    }
    if archetype not in templates:
        raise KeyError(f"Unsupported archetype: {archetype}")
    return templates[archetype]


def _build_objective_profile_bullets(row: Mapping[str, Any], sector_regime: str, cycle_best: Mapping[str, Any] | None) -> List[str]:
    bullets = [
        f"Ngành: `{row.get('Sector', 'không rõ')}`; chế độ ngành hiện tại: `{sector_regime}`.",
        f"Định giá snapshot: `PE_fwd {_fmt_ratio(_to_float(row.get('PE_fwd')))} / PB {_fmt_ratio(_to_float(row.get('PB')))} / ROE {_fmt_pct(_to_float(row.get('ROE')), signed=False)}`.",
        f"Sức mạnh giá: `Ret20d {_fmt_pct(_to_float(row.get('Ret20d')))} / Ret20dVsIndex {_fmt_pct(_to_float(row.get('Ret20dVsIndex')))} / RSI14 {_fmt_ratio(_to_float(row.get('RSI14')))} / ATR14Pct {_fmt_pct(_to_float(row.get('ATR14Pct')), signed=False)}`.",
        f"Vị thế trong chu kỳ giá: `Pos52wPct {_fmt_ratio(_to_float(row.get('Pos52wPct')))} / High52w {_fmt_price(_to_float(row.get('High52w')))} / Low52w {_fmt_price(_to_float(row.get('Low52w')))}.`",
    ]
    if cycle_best is not None:
        bullets.append(
            "Cycle fit hiện tại: "
            f"`{cycle_best.get('ForecastWindow', 'n/a')}` với đỉnh kỳ vọng "
            f"`{_fmt_pct(_to_float(cycle_best.get('PredPeakRetPct')))}` "
            f"trong khoảng `{_fmt_ratio(_to_float(cycle_best.get('PredPeakDays')))} phiên` và drawdown kỳ vọng `{_fmt_pct(_to_float(cycle_best.get('PredDrawdownPct')))}.`"
        )
    return bullets


def _pick_sector_row(sector_df: pd.DataFrame, sector: str) -> Mapping[str, Any]:
    match = sector_df.loc[sector_df["Sector"].astype(str) == str(sector)]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _pick_first_by_ticker(frame: pd.DataFrame | None, ticker: str) -> Mapping[str, Any] | None:
    if frame is None:
        return None
    match = frame.loc[frame["Ticker"].astype(str).str.upper() == ticker]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _pick_rows_by_ticker(frame: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    match = frame.loc[frame["Ticker"].astype(str).str.upper() == ticker].copy()
    return match.reset_index(drop=True)


def _summarise_range(range_df: pd.DataFrame, forecast_window: str) -> str:
    if range_df.empty:
        return "không có dữ liệu"
    match = range_df.loc[range_df["ForecastWindow"].astype(str) == forecast_window]
    if match.empty:
        return "không có dữ liệu"
    row = match.iloc[0]
    return (
        f"low/mid/high = `{_fmt_pct(_to_float(row.get('PredLowRetPct')))} / "
        f"{_fmt_pct(_to_float(row.get('PredMidRetPct')))} / {_fmt_pct(_to_float(row.get('PredHighRetPct')))}`"
    )


def _summarise_best_ladders(ladder_df: pd.DataFrame) -> Dict[str, Any]:
    if ladder_df.empty:
        return {
            "summary": "không có dữ liệu ladder",
            "buy_zone_low": None,
            "buy_zone_high": None,
            "top_levels": [],
        }

    def is_historical_row(row: pd.Series) -> bool:
        category = str(row.get("EntryAnchorCategory") or "").strip().lower()
        if category in {"historical", "mixed"}:
            return True
        if category == "tactical":
            return False
        anchors = {
            str(part).strip()
            for part in str(row.get("EntryAnchor") or "").split("|")
            if str(part).strip()
        }
        if not anchors:
            return True
        has_historical = any(anchor in HISTORICAL_LADDER_ANCHORS for anchor in anchors)
        has_tactical = any(anchor in TACTICAL_LADDER_ANCHORS for anchor in anchors)
        return has_historical or not has_tactical

    historical = ladder_df[ladder_df.apply(is_historical_row, axis=1)].copy()
    ranked_source = historical if not historical.empty else ladder_df
    ranked = ranked_source.sort_values(["EntryScoreRank", "PriceRank"]).head(3).copy()
    levels = [f"{_fmt_price(_to_float(row['LimitPrice']))} (rank {int(row['EntryScoreRank'])})" for _, row in ranked.iterrows()]
    prices = [_to_float(value) for value in ranked["LimitPrice"].tolist()]
    prices = [value for value in prices if value is not None]
    return {
        "summary": ", ".join(levels),
        "buy_zone_low": min(prices) if prices else None,
        "buy_zone_high": max(prices) if prices else None,
        "top_levels": [
            {
                "LimitPrice": _to_float(level.get("LimitPrice")),
                "EntryScoreRank": _to_float(level.get("EntryScoreRank")),
                "EntryScore": _to_float(level.get("EntryScore")),
                "EntryAnchorCategory": str(level.get("EntryAnchorCategory") or ""),
                "FillScoreComposite": _to_float(level.get("FillScoreComposite")),
            }
            for _, level in ranked.iterrows()
        ],
    }


def _derive_session_buy_tranche_blueprint(
    *,
    row: Mapping[str, Any],
    ladder_df: pd.DataFrame,
    persistent_weakness_bid: bool,
    allow_add_on_strength: bool,
    burst_summary: Mapping[str, Any],
    execution_guidance: Mapping[str, Any],
) -> Dict[str, Any]:
    if ladder_df.empty:
        return {
            "summary": "không có buy-tranche blueprint",
            "tranches": [],
        }

    ranked = ladder_df.sort_values(["EntryScoreRank", "PriceRank"]).copy()
    if "EntryAnchorCategory" in ranked.columns:
        historical_ranked = ranked[
            ranked["EntryAnchorCategory"].astype(str).str.lower().isin({"historical", "mixed"})
        ].copy()
        if not historical_ranked.empty:
            ranked = historical_ranked
    core = ranked.head(3).reset_index(drop=True).copy()
    if core.empty:
        return {
            "summary": "không có buy-tranche blueprint",
            "tranches": [],
        }

    last_price = _to_float(row.get("Last"))
    latest_signal_age = burst_summary.get("latest_signal_age")
    burst_bias = str(execution_guidance.get("burst_execution_bias") or "").strip().lower()
    trim_aggression = str(execution_guidance.get("trim_aggression") or "").strip().lower()

    core_prices = pd.to_numeric(core["LimitPrice"], errors="coerce").dropna().tolist()
    core_high = max(core_prices) if core_prices else None
    continuation_price = None
    continuation_share_pct = 0.0
    bridge_row: Mapping[str, Any] | None = None
    bridge_share_pct = 0.0

    continuation_candidates = [
        _to_float(row.get("ValidAsk1")),
        _to_float(row.get("GridAbove_T1")),
    ]
    continuation_candidates = [
        price
        for price in continuation_candidates
        if price is not None and price > 0.0 and (core_high is None or price > core_high)
    ]
    if continuation_candidates:
        continuation_price = min(continuation_candidates)

    if (
        (persistent_weakness_bid or allow_add_on_strength)
        and last_price is not None
        and core_high is not None
        and core_high > 0.0
        and last_price > core_high * 1.02
    ):
        if continuation_price is not None and continuation_price <= last_price * 1.02:
            if burst_bias in {"respect_day2_followthrough", "keep_core_trim_selectively"}:
                continuation_share_pct = 9.0 if allow_add_on_strength else 7.0
            elif burst_bias == "normal_tactical_management":
                continuation_share_pct = 7.0 if allow_add_on_strength else 5.5
            elif burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
                continuation_share_pct = 4.0 if persistent_weakness_bid else 0.0
            if trim_aggression == "high":
                continuation_share_pct = max(continuation_share_pct - 0.5, 0.0)
            continuation_share_pct = min(continuation_share_pct, 10.0)

        above_core = ranked.loc[pd.to_numeric(ranked["LimitPrice"], errors="coerce") > core_high].copy()
        if not above_core.empty:
            above_core = above_core.sort_values(["LimitPrice", "EntryScoreRank"])
            bridge_row = above_core.iloc[0].to_dict()
            bridge_share_pct = 7.5
            if latest_signal_age is not None and latest_signal_age <= 1:
                bridge_share_pct += 1.5
            fill_advantage = (_to_float(bridge_row.get("FillScoreComposite")) or 0.0) - max(
                (_to_float(value) or 0.0) for value in core["FillScoreComposite"].tolist()
            )
            if fill_advantage >= 15.0:
                bridge_share_pct += 1.0
            if ((last_price / core_high) - 1.0) * 100.0 >= 4.0:
                bridge_share_pct += 1.0
            if trim_aggression == "high" and burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
                bridge_share_pct += 0.5
            bridge_share_pct = min(bridge_share_pct, 10.0)

    core_budget_share_pct = max(100.0 - bridge_share_pct - continuation_share_pct, 0.0)
    core_raw_scores: List[tuple[int, float]] = []
    for idx, (_, core_row) in enumerate(core.iterrows()):
        entry_score = max(_to_float(core_row.get("EntryScore")) or 0.0, 0.0)
        fill_score = max(_to_float(core_row.get("FillScoreComposite")) or 0.0, 0.0)
        rank = max(_to_float(core_row.get("EntryScoreRank")) or float(idx + 1), 1.0)
        raw_score = entry_score * (1.0 + 0.015 * min(fill_score, 100.0)) / (rank ** 0.5)
        core_raw_scores.append((idx, raw_score))
    raw_total = sum(score for _, score in core_raw_scores)
    if raw_total <= 0.0:
        equal_share = core_budget_share_pct / len(core_raw_scores)
        core_share_map = {idx: equal_share for idx, _ in core_raw_scores}
    else:
        core_share_map = {idx: core_budget_share_pct * score / raw_total for idx, score in core_raw_scores}

    tranche_rows: List[Dict[str, Any]] = []
    if continuation_price is not None and continuation_share_pct > 0.0:
        continuation_base_score = max(
            max((_to_float(value) or 0.0) for value in core["EntryScore"].tolist()) * 0.72,
            0.01,
        )
        if allow_add_on_strength:
            continuation_base_score *= 1.08
        if burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
            continuation_base_score *= 0.86
        tranche_rows.append(
            {
                "Role": "continuation_reserve",
                "LimitPrice": continuation_price,
                "EntryScoreRank": 0.0,
                "PlanSharePctOfTickerSession": round(continuation_share_pct, 2),
                "BaseTrancheScore": round(continuation_base_score, 6),
                "MandatoryFloorPctOfTickerCap": 0.0,
                "Reason": "giữ một reserve nhỏ gần giá hiện tại để không lỡ nhịp nếu mã không pullback mà vẫn giữ đà",
            }
        )

    if bridge_row is not None and bridge_share_pct > 0.0:
        bridge_entry_score = max(_to_float(bridge_row.get("EntryScore")) or 0.0, 0.0)
        bridge_fill_score = max(_to_float(bridge_row.get("FillScoreComposite")) or 0.0, 0.0)
        bridge_base_score = bridge_entry_score * (1.0 + 0.01 * min(bridge_fill_score, 100.0))
        bridge_base_score *= 0.92 if allow_add_on_strength else 0.82
        if trim_aggression == "high":
            bridge_base_score *= 0.92
        tranche_rows.append(
            {
                "Role": "bridge",
                "LimitPrice": _to_float(bridge_row.get("LimitPrice")),
                "EntryScoreRank": _to_float(bridge_row.get("EntryScoreRank")),
                "PlanSharePctOfTickerSession": round(bridge_share_pct, 2),
                "BaseTrancheScore": round(bridge_base_score, 6),
                "MandatoryFloorPctOfTickerCap": 0.0,
                "Reason": "giữ một probe gần hơn để không bỏ lỡ phiên chạy khi giá đứng quá xa core ladder",
            }
        )

    core_desc = core.sort_values("LimitPrice", ascending=False).reset_index().rename(columns={"index": "CoreIdx"})
    role_names = ["shallow_core", "mid_core", "deep_core"]
    for desc_idx, (_, desc_row) in enumerate(core_desc.iterrows()):
        original_idx = int(desc_row["CoreIdx"])
        share_pct = round(core_share_map.get(original_idx, 0.0), 2)
        entry_score = max(_to_float(desc_row.get("EntryScore")) or 0.0, 0.0)
        fill_score = max(_to_float(desc_row.get("FillScoreComposite")) or 0.0, 0.0)
        base_score = entry_score * (1.0 + 0.015 * min(fill_score, 100.0))
        role = role_names[min(desc_idx, len(role_names) - 1)]
        if role == "shallow_core":
            base_score *= 0.95 if not allow_add_on_strength else 1.02
            if trim_aggression == "high" and burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
                base_score *= 0.92
        elif role == "deep_core":
            base_score *= 1.08
            if persistent_weakness_bid:
                base_score *= 1.12
        mandatory_floor_pct = 0.0
        if persistent_weakness_bid and role == "deep_core":
            mandatory_floor_pct = min(max(share_pct * 0.35, 12.0), share_pct)
        tranche_rows.append(
            {
                "Role": role,
                "LimitPrice": _to_float(desc_row.get("LimitPrice")),
                "EntryScoreRank": _to_float(desc_row.get("EntryScoreRank")),
                "PlanSharePctOfTickerSession": share_pct,
                "BaseTrancheScore": round(base_score, 6),
                "MandatoryFloorPctOfTickerCap": round(mandatory_floor_pct, 2),
                "Reason": "core ladder theo EV sau khi điều chỉnh bởi xác suất chạm",
            }
        )

    share_sum = sum((_to_float(tranche.get("PlanSharePctOfTickerSession")) or 0.0) for tranche in tranche_rows)
    drift = round(100.0 - share_sum, 2)
    if tranche_rows and abs(drift) >= 0.01:
        tranche_rows[-1]["PlanSharePctOfTickerSession"] = round(
            (_to_float(tranche_rows[-1].get("PlanSharePctOfTickerSession")) or 0.0) + drift,
            2,
        )

    summary = ", ".join(
        f"{_fmt_price(_to_float(tranche.get('LimitPrice')))}: {_fmt_ratio(_to_float(tranche.get('PlanSharePctOfTickerSession')))}% ticker-cap ({tranche.get('Role')})"
        for tranche in tranche_rows
    )
    return {
        "summary": summary or "không có buy-tranche blueprint",
        "tranches": tranche_rows,
    }


def _build_session_buy_tranches(
    *,
    row: Mapping[str, Any],
    ladder_df: pd.DataFrame,
    allocation_target: Mapping[str, Any],
    burst_summary: Mapping[str, Any],
    execution_guidance: Mapping[str, Any],
) -> Dict[str, Any]:
    suggested_new_capital_pct = _to_float(allocation_target.get("SuggestedNewCapitalPct")) or 0.0
    if ladder_df.empty or suggested_new_capital_pct <= 0.0:
        return {
            "summary": "không có session buy plan riêng",
            "tranches": [],
        }
    blueprint = _derive_session_buy_tranche_blueprint(
        row=row,
        ladder_df=ladder_df,
        persistent_weakness_bid=bool(allocation_target.get("PersistentWeaknessBid")),
        allow_add_on_strength=bool(allocation_target.get("AddOnStrengthAllowed")),
        burst_summary=burst_summary,
        execution_guidance=execution_guidance,
    )
    tranche_rows: List[Dict[str, Any]] = []
    for tranche in blueprint["tranches"]:
        share_pct = _to_float(tranche.get("PlanSharePctOfTickerSession")) or 0.0
        tranche_rows.append(
            {
                "Role": tranche.get("Role"),
                "LimitPrice": tranche.get("LimitPrice"),
                "EntryScoreRank": tranche.get("EntryScoreRank"),
                "BudgetSharePctOfSession": round(share_pct, 2),
                "CapitalPctOfPortfolio": round(suggested_new_capital_pct * share_pct / 100.0, 2),
                "Reason": tranche.get("Reason"),
            }
        )
    summary = ", ".join(
        f"{_fmt_price(_to_float(tranche.get('LimitPrice')))}: {_fmt_ratio(_to_float(tranche.get('BudgetSharePctOfSession')))}% session ({tranche.get('Role')})"
        for tranche in tranche_rows
    )
    return {
        "summary": summary or "không có session buy plan riêng",
        "tranches": tranche_rows,
    }


def _summarise_timing(timing_df: pd.DataFrame) -> Dict[str, Any]:
    if timing_df.empty:
        return {
            "summary": "không có dữ liệu timing",
            "best_window": None,
            "best_net_edge": None,
            "t10_net_edge": None,
        }
    ranked = timing_df.sort_values("PredNetEdgePct", ascending=False).copy()
    best = ranked.iloc[0]
    t10 = timing_df.loc[timing_df["ForecastWindow"].astype(str) == "T+10"]
    t10_net_edge = _to_float(t10.iloc[0]["PredNetEdgePct"]) if not t10.empty else None
    lines = []
    for _, row in timing_df.sort_values("Horizon").iterrows():
        lines.append(
            f"{row['ForecastWindow']}: net edge {_fmt_pct(_to_float(row.get('PredNetEdgePct')))}, "
            f"peak {_fmt_pct(_to_float(row.get('PredPeakRetPct')))} trong {_fmt_ratio(_to_float(row.get('PredPeakDay')))} phiên"
        )
    return {
        "summary": "; ".join(lines),
        "best_window": str(best.get("ForecastWindow")),
        "best_net_edge": _to_float(best.get("PredNetEdgePct")),
        "t10_net_edge": t10_net_edge,
    }


def _summarise_intraday_report(intraday_df: pd.DataFrame) -> str:
    if intraday_df.empty:
        return "không có lớp intraday rest-of-session"
    row = intraday_df.iloc[0]
    return (
        f"bucket `{row.get('SnapshotTimeBucket')}`: low/mid/high = "
        f"`{_fmt_pct(_to_float(row.get('PredLowRetPct')))} / {_fmt_pct(_to_float(row.get('PredMidRetPct')))} / "
        f"{_fmt_pct(_to_float(row.get('PredHighRetPct')))}`"
    )


def _load_daily_history(history_dir: Path, ticker: str) -> pd.DataFrame:
    path = history_dir / f"{ticker}_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = ["open", "high", "low", "close", "volume", "date_vn"]
    _require_columns(frame, required, str(path))
    frame["Date"] = pd.to_datetime(frame["date_vn"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


def _summarise_burst_patterns(history_dir: Path, ticker: str) -> Dict[str, Any]:
    frame = _load_daily_history(history_dir, ticker)
    if frame.empty or len(frame) < 25:
        return {
            "summary": "không đủ lịch sử daily để đo burst/follow-through",
            "sample_count": 0,
            "next_day_positive_rate": None,
            "next_day_strong_rate": None,
            "third_day_negative_rate": None,
            "avg_three_day_drawdown_pct": None,
            "latest_signal_age": None,
        }
    frame["Ret1Pct"] = frame["close"].pct_change() * 100.0
    frame["Ret2Pct"] = (frame["close"] / frame["close"].shift(2) - 1.0) * 100.0
    records: List[Dict[str, Any]] = []
    latest_signal_index: int | None = None
    for idx, row in frame.iterrows():
        is_signal = ((_to_float(row.get("Ret1Pct")) or 0.0) >= 6.5) or ((_to_float(row.get("Ret2Pct")) or 0.0) >= 10.0)
        if not is_signal:
            continue
        latest_signal_index = idx
        signal_close = _to_float(row.get("close"))
        if signal_close is None or signal_close <= 0:
            continue
        next_day_ret = None
        two_day_ret = None
        three_day_ret = None
        min_low_3d = None
        if idx + 1 < len(frame):
            next_day_ret = ((frame.iloc[idx + 1]["close"] / signal_close) - 1.0) * 100.0
        if idx + 2 < len(frame):
            two_day_ret = ((frame.iloc[idx + 2]["close"] / signal_close) - 1.0) * 100.0
        if idx + 3 < len(frame):
            three_day_ret = ((frame.iloc[idx + 3]["close"] / signal_close) - 1.0) * 100.0
            min_low_3d = ((frame.iloc[idx + 1 : idx + 4]["low"].min() / signal_close) - 1.0) * 100.0
        records.append(
            {
                "next_day_ret": next_day_ret,
                "two_day_ret": two_day_ret,
                "three_day_ret": three_day_ret,
                "min_low_3d": min_low_3d,
            }
        )
    if not records:
        return {
            "summary": "chưa có mẫu burst gần-trần đủ rõ trong lịch sử daily hiện có",
            "sample_count": 0,
            "next_day_positive_rate": None,
            "next_day_strong_rate": None,
            "third_day_negative_rate": None,
            "avg_three_day_drawdown_pct": None,
            "latest_signal_age": None,
        }
    next_day = [item["next_day_ret"] for item in records if item["next_day_ret"] is not None]
    day_three = [item["three_day_ret"] for item in records if item["three_day_ret"] is not None]
    min_low_3d = [item["min_low_3d"] for item in records if item["min_low_3d"] is not None]
    latest_signal_age = None
    if latest_signal_index is not None:
        latest_signal_age = (len(frame) - 1) - latest_signal_index
    next_day_positive_rate = (sum(value > 0.0 for value in next_day) / len(next_day) * 100.0) if next_day else None
    next_day_strong_rate = (sum(value >= 3.0 for value in next_day) / len(next_day) * 100.0) if next_day else None
    third_day_negative_rate = (sum(value < 0.0 for value in day_three) / len(day_three) * 100.0) if day_three else None
    avg_three_day_drawdown_pct = (sum(min_low_3d) / len(min_low_3d)) if min_low_3d else None
    summary = (
        f"Mẫu burst gần-trần lịch sử `{len(records)}` lần; "
        f"phiên kế tiếp còn tăng `{_fmt_pct(next_day_positive_rate, signed=False)}`, "
        f"tăng mạnh >3% `{_fmt_pct(next_day_strong_rate, signed=False)}`; "
        f"đến T+3 sau burst, xác suất đóng âm lại `{_fmt_pct(third_day_negative_rate, signed=False)}` "
        f"và drawdown thấp nhất trung bình trong 3 phiên là `{_fmt_pct(avg_three_day_drawdown_pct)}`."
    )
    if latest_signal_age is not None:
        summary += f" Burst gần nhất hiện đang ở tuổi `D+{latest_signal_age}`."
    return {
        "summary": summary,
        "sample_count": len(records),
        "next_day_positive_rate": next_day_positive_rate,
        "next_day_strong_rate": next_day_strong_rate,
        "third_day_negative_rate": third_day_negative_rate,
        "avg_three_day_drawdown_pct": avg_three_day_drawdown_pct,
        "latest_signal_age": latest_signal_age,
    }


def _load_intraday_session(history_dir: Path, ticker: str) -> pd.DataFrame:
    path = history_dir / f"{ticker}_5m.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = ["t", "open", "high", "low", "close", "volume", "date_vn"]
    _require_columns(frame, required, str(path))
    frame["Timestamp"] = pd.to_datetime(pd.to_numeric(frame["t"], errors="coerce"), unit="s", utc=True, errors="coerce")
    frame = frame.dropna(subset=["Timestamp"]).copy()
    frame["Timestamp"] = frame["Timestamp"].dt.tz_convert("Asia/Ho_Chi_Minh")
    frame = frame.sort_values("Timestamp").reset_index(drop=True)
    if frame.empty:
        return frame
    latest_date = str(frame["date_vn"].dropna().astype(str).iloc[-1])
    return frame.loc[frame["date_vn"].astype(str) == latest_date].reset_index(drop=True)


def _intraday_checkpoints(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    checkpoints = ["09:15", "09:45", "10:30", "11:00", "13:15", "14:00", "14:25", "14:45"]
    session_open = _to_float(frame.iloc[0]["open"]) or _to_float(frame.iloc[0]["close"]) or 0.0
    rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        timestamp = pd.Timestamp(f"{str(frame.iloc[-1]['date_vn'])} {checkpoint}:00", tz="Asia/Ho_Chi_Minh")
        subset = frame.loc[frame["Timestamp"] <= timestamp]
        if subset.empty:
            continue
        row = subset.iloc[-1]
        close_value = _to_float(row["close"]) or 0.0
        rows.append(
            {
                "time": checkpoint,
                "close": close_value,
                "ret_from_open_pct": ((close_value / session_open) - 1.0) * 100.0 if session_open else 0.0,
                "cum_volume": float(subset["volume"].sum()),
            }
        )
    return rows


def _summarise_intraday_tape(history_dir: Path, ticker: str) -> Dict[str, Any]:
    ticker_frame = _load_intraday_session(history_dir, ticker)
    index_frame = _load_intraday_session(history_dir, "VNINDEX")
    if ticker_frame.empty or index_frame.empty:
        return {
            "summary": "không có tape intraday 5m để tóm tắt",
            "damage_hint": None,
            "execution_bias": "unknown",
            "opening_squeeze_failure": False,
            "close_in_range_pct": None,
            "fade_from_high_pct": None,
            "close_vs_vwap_pct": None,
            "pm_return_pct": None,
            "pm_excess_vs_index_pct": None,
            "burst_execution_bias": "neutral",
            "trim_aggression": "moderate",
        }

    ticker_checkpoints = _intraday_checkpoints(ticker_frame)
    index_checkpoints = _intraday_checkpoints(index_frame)
    session_open = _to_float(ticker_frame.iloc[0]["open"]) or 0.0
    session_close = _to_float(ticker_frame.iloc[-1]["close"]) or 0.0
    session_high = _to_float(ticker_frame["high"].max())
    session_low = _to_float(ticker_frame["low"].min())
    session_range = (session_high - session_low) if session_high is not None and session_low is not None else None
    high_at = ticker_frame.loc[ticker_frame["high"].idxmax(), "Timestamp"].strftime("%H:%M")
    low_at = ticker_frame.loc[ticker_frame["low"].idxmin(), "Timestamp"].strftime("%H:%M")
    high_ts = ticker_frame.loc[ticker_frame["high"].idxmax(), "Timestamp"]
    low_ts = ticker_frame.loc[ticker_frame["low"].idxmin(), "Timestamp"]
    typical_price = (ticker_frame["high"] + ticker_frame["low"] + ticker_frame["close"]) / 3.0
    volume_sum = float(ticker_frame["volume"].sum())
    approx_vwap = float((typical_price * ticker_frame["volume"]).sum() / volume_sum) if volume_sum > 0 else None
    close_vs_vwap_pct = ((session_close / approx_vwap) - 1.0) * 100.0 if approx_vwap and approx_vwap > 0 else None
    close_in_range_pct = (
        ((session_close - session_low) / session_range) * 100.0
        if session_range is not None and session_range > 0 and session_low is not None
        else None
    )
    fade_from_high_pct = ((session_close / session_high) - 1.0) * 100.0 if session_high and session_high > 0 else None
    opening_slice = ticker_frame.iloc[: min(len(ticker_frame), 6)].copy()
    opening_drive_pct = None
    if not opening_slice.empty and session_open > 0:
        opening_drive_pct = ((float(opening_slice["high"].max()) / session_open) - 1.0) * 100.0
    opening_squeeze_failure = bool(
        opening_drive_pct is not None
        and opening_drive_pct >= 1.5
        and fade_from_high_pct is not None
        and fade_from_high_pct <= -2.0
        and close_in_range_pct is not None
        and close_in_range_pct <= 35.0
        and high_ts.hour < 11
    )

    pm_open = None
    if not ticker_frame.loc[ticker_frame["Timestamp"].dt.strftime("%H:%M") >= "13:00"].empty:
        pm_open = _to_float(ticker_frame.loc[ticker_frame["Timestamp"].dt.strftime("%H:%M") >= "13:00"].iloc[0]["open"])
    if pm_open is None:
        pm_open = session_open if session_open > 0 else None
    pm_return_pct = ((session_close / pm_open) - 1.0) * 100.0 if pm_open and pm_open > 0 else None

    index_session_close = _to_float(index_frame.iloc[-1]["close"]) or 0.0
    index_pm_open = None
    if not index_frame.loc[index_frame["Timestamp"].dt.strftime("%H:%M") >= "13:00"].empty:
        index_pm_open = _to_float(index_frame.loc[index_frame["Timestamp"].dt.strftime("%H:%M") >= "13:00"].iloc[0]["open"])
    if index_pm_open is None:
        index_pm_open = _to_float(index_frame.iloc[0]["open"])
    index_pm_return_pct = (
        ((index_session_close / index_pm_open) - 1.0) * 100.0
        if index_pm_open is not None and index_pm_open > 0 and index_session_close > 0
        else None
    )
    pm_excess_vs_index_pct = (
        pm_return_pct - index_pm_return_pct
        if pm_return_pct is not None and index_pm_return_pct is not None
        else None
    )

    execution_bias = "neutral"
    if (
        close_in_range_pct is not None
        and close_in_range_pct <= 35.0
        and close_vs_vwap_pct is not None
        and close_vs_vwap_pct < 0.0
        and pm_excess_vs_index_pct is not None
        and pm_excess_vs_index_pct <= -0.5
    ):
        execution_bias = "distribution"
    elif (
        close_in_range_pct is not None
        and close_in_range_pct >= 65.0
        and close_vs_vwap_pct is not None
        and close_vs_vwap_pct >= 0.0
        and pm_excess_vs_index_pct is not None
        and pm_excess_vs_index_pct >= 0.5
    ):
        execution_bias = "absorption"

    burst_execution_bias = "neutral"
    trim_aggression = "moderate"
    if opening_squeeze_failure:
        burst_execution_bias = "failed_followthrough"
        trim_aggression = "high"
    elif execution_bias == "distribution":
        burst_execution_bias = "respect_t25_supply"
        trim_aggression = "high"
    elif execution_bias == "absorption":
        burst_execution_bias = "respect_day2_followthrough"
        trim_aggression = "light"
    relative_lines: List[str] = []
    for t_row, i_row in zip(ticker_checkpoints, index_checkpoints):
        excess = t_row["ret_from_open_pct"] - i_row["ret_from_open_pct"]
        relative_lines.append(f"{t_row['time']}: excess {_fmt_pct(excess)}")
    summary = (
        f"Phiên gần nhất mở `{_fmt_price(session_open)}`, cao nhất `{_fmt_price(session_high)}` lúc `{high_at}`, "
        f"thấp nhất `{_fmt_price(session_low)}` lúc `{low_at}`, đóng `{_fmt_price(session_close)}`. "
        f"Checkpoint: "
        + "; ".join(
            f"{row['time']} close `{_fmt_price(row['close'])}` / từ open `{_fmt_pct(row['ret_from_open_pct'])}` / lũy kế vol `{row['cum_volume'] / 1_000_000:.2f}m`"
            for row in ticker_checkpoints
        )
        + ". So với VNINDEX: "
        + "; ".join(relative_lines)
        + ". "
        + (
            f"Close-in-range `{_fmt_pct(close_in_range_pct, signed=False)}`, "
            f"fade từ high `{_fmt_pct(fade_from_high_pct)}`, "
            f"close vs VWAP xấp xỉ `{_fmt_pct(close_vs_vwap_pct)}`, "
            f"PM excess vs index `{_fmt_pct(pm_excess_vs_index_pct)}`. "
            f"Execution bias `{execution_bias}`"
            + ("; có dấu hiệu opening squeeze failure." if opening_squeeze_failure else ".")
        )
    )
    return {
        "summary": summary,
        "damage_hint": session_low,
        "execution_bias": execution_bias,
        "opening_squeeze_failure": opening_squeeze_failure,
        "close_in_range_pct": close_in_range_pct,
        "fade_from_high_pct": fade_from_high_pct,
        "close_vs_vwap_pct": close_vs_vwap_pct,
        "pm_return_pct": pm_return_pct,
        "pm_excess_vs_index_pct": pm_excess_vs_index_pct,
        "burst_execution_bias": burst_execution_bias,
        "trim_aggression": trim_aggression,
    }


def _refine_burst_execution_guidance(
    burst_summary: Mapping[str, Any],
    tape_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    latest_signal_age = burst_summary.get("latest_signal_age")
    execution_bias = str(tape_summary.get("execution_bias") or "neutral")
    opening_squeeze_failure = bool(tape_summary.get("opening_squeeze_failure"))

    burst_execution_bias = str(tape_summary.get("burst_execution_bias") or "neutral")
    trim_aggression = str(tape_summary.get("trim_aggression") or "moderate")
    urgent_trim_mode = "none"
    must_sell_fraction_pct = 0.0
    note = "Tape hiện tại chưa tạo ra thiên kiến execution đặc biệt."

    if latest_signal_age is None:
        return {
            "burst_execution_bias": burst_execution_bias,
            "trim_aggression": trim_aggression,
            "urgent_trim_mode": urgent_trim_mode,
            "must_sell_fraction_pct": must_sell_fraction_pct,
            "note": note,
        }

    if latest_signal_age <= 1:
        if opening_squeeze_failure:
            burst_execution_bias = "failed_day2_followthrough"
            trim_aggression = "high"
            urgent_trim_mode = "frontload_sell"
            must_sell_fraction_pct = 35.0
            note = "Burst còn rất mới nhưng tape mở kéo rồi fail; không được mặc định chờ thêm một phiên tím."
        elif execution_bias == "absorption":
            burst_execution_bias = "respect_day2_followthrough"
            trim_aggression = "light"
            note = "Burst còn mới và tape giữ được lực; tránh chốt quá sớm nếu chưa có dấu hiệu phân phối rõ."
        else:
            burst_execution_bias = "wait_for_day2_confirmation"
            trim_aggression = "moderate"
            urgent_trim_mode = "staggered_trim"
            must_sell_fraction_pct = 15.0
            note = "Burst còn mới nhưng tape chưa thật sự sạch; chỉ trim vừa phải và giữ ladder BUY sâu."
    elif latest_signal_age >= 2:
        if execution_bias == "distribution":
            burst_execution_bias = "respect_t25_supply"
            trim_aggression = "high"
            urgent_trim_mode = "frontload_sell"
            must_sell_fraction_pct = 30.0
            note = "Burst đã sang nhịp dễ gặp cung T+2.5; nếu tape phân phối thì phải trim chủ động hơn."
        elif execution_bias == "absorption":
            burst_execution_bias = "keep_core_trim_selectively"
            trim_aggression = "moderate"
            urgent_trim_mode = "staggered_trim"
            must_sell_fraction_pct = 15.0
            note = "Đã qua giai đoạn burst đầu nhưng tape vẫn hấp thụ; nên giữ core và chỉ trim chọn lọc."
        else:
            burst_execution_bias = "normal_tactical_management"
            trim_aggression = "moderate"
            urgent_trim_mode = "staggered_trim"
            must_sell_fraction_pct = 10.0
            note = "Đã qua giai đoạn burst đầu; quản trị vị thế theo tape và vùng giá thay vì kỳ vọng continuation máy móc."

    return {
        "burst_execution_bias": burst_execution_bias,
        "trim_aggression": trim_aggression,
        "urgent_trim_mode": urgent_trim_mode,
        "must_sell_fraction_pct": must_sell_fraction_pct,
        "note": note,
    }


def _base_target_band(archetype: str) -> tuple[float, float]:
    bands = {
        "quality_financial_trend": (18.0, 30.0),
        "quality_trend": (16.0, 28.0),
        "cyclical_beta": (12.0, 24.0),
        "balanced_swing": (10.0, 18.0),
        "momentum_high_beta": (8.0, 16.0),
        "special_situation": (6.0, 14.0),
    }
    return bands.get(archetype, (8.0, 18.0))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ticker_conviction_score(
    *,
    row: Mapping[str, Any],
    sector_regime: str,
    timing_summary: Mapping[str, Any],
    cycle_best: Mapping[str, Any] | None,
    playbook_row: Mapping[str, Any] | None,
) -> float:
    score = 1.0
    if sector_regime == "leader":
        score += 0.25
    elif sector_regime == "laggard":
        score -= 0.2

    best_edge = timing_summary.get("best_net_edge") or 0.0
    t10_edge = timing_summary.get("t10_net_edge") or 0.0
    rsi = _to_float(row.get("RSI14")) or 0.0
    ret20_vs_idx = _to_float(row.get("Ret20dVsIndex")) or 0.0

    if best_edge > 1.0:
        score += 0.15
    elif best_edge < 0.0:
        score -= 0.15
    if t10_edge < 0.0:
        score -= 0.2
    if ret20_vs_idx >= 8.0:
        score += 0.1
    elif ret20_vs_idx <= -5.0:
        score -= 0.1
    if rsi >= 75.0:
        score -= 0.1

    if playbook_row is not None and _bool_text(playbook_row.get("LatestSignal")):
        score += 0.1

    if cycle_best is not None:
        peak_ret = _to_float(cycle_best.get("PredPeakRetPct")) or 0.0
        drawdown = _to_float(cycle_best.get("PredDrawdownPct")) or 0.0
        if peak_ret >= 8.0 and drawdown >= -6.0:
            score += 0.1
        elif peak_ret <= 4.0 or drawdown <= -10.0:
            score -= 0.1

    return _clamp(score, 0.35, 1.8)


def _target_weight_band(
    *,
    archetype: str,
    row: Mapping[str, Any],
    sector_regime: str,
    timing_summary: Mapping[str, Any],
    human_overlay: Mapping[str, Any] | None,
) -> tuple[float, float]:
    min_weight, max_weight = _base_target_band(archetype)
    best_edge = timing_summary.get("best_net_edge") or 0.0
    t10_edge = timing_summary.get("t10_net_edge") or 0.0
    rsi = _to_float(row.get("RSI14")) or 0.0
    ret20_vs_idx = _to_float(row.get("Ret20dVsIndex")) or 0.0

    if sector_regime == "leader" and ret20_vs_idx >= 5.0 and best_edge > 0.0:
        min_weight += 2.0
        max_weight += 2.0
    elif sector_regime == "laggard":
        min_weight -= 2.0
        max_weight -= 2.0

    if best_edge > 1.0:
        max_weight += 1.0
    elif best_edge < 0.0:
        min_weight -= 1.0
        max_weight -= 1.0

    if t10_edge < 0.0:
        min_weight -= 1.0
        max_weight -= 2.0

    if rsi >= 75.0:
        min_weight -= 1.0
        max_weight -= 1.0

    target_price = _to_float(human_overlay.get("target_price")) if human_overlay is not None else None
    last_price = _to_float(row.get("Last")) or 0.0
    target_multiple = (target_price / last_price) if target_price is not None and last_price > 0.0 else None
    if target_multiple is not None:
        if target_multiple >= 1.8:
            min_weight = max(min_weight, 12.0)
            max_weight = max(max_weight, 24.0)
        elif target_multiple >= 1.4:
            min_weight = max(min_weight, 10.0)
            max_weight = max(max_weight, 20.0)

    min_weight = _clamp(min_weight, 4.0, 28.0)
    max_weight = _clamp(max_weight, min_weight + 4.0, 40.0)
    return min_weight, max_weight


def _portfolio_target_invested_pct(market_summary: Mapping[str, Any]) -> tuple[float, float]:
    regime = _market_regime(market_summary)
    atr_rank = _to_float(market_summary.get("VNINDEX_ATR14PctRank")) or 0.0
    if regime == "risk_on":
        target = 70.0
        strength_share = 0.35
        if atr_rank >= 80.0:
            target = 65.0
    elif regime == "risk_off":
        target = 35.0
        strength_share = 0.10
        if (_to_float(market_summary.get("BreadthAboveSMA20Pct")) or 0.0) <= 25.0:
            target = 30.0
    else:
        target = 55.0
        strength_share = 0.25
        if atr_rank >= 80.0:
            target = 50.0
    return target, strength_share


def _single_name_session_build_cap_pct(
    *,
    deployable_gap_pct: float,
    archetype: str,
    execution_guidance: Mapping[str, Any],
) -> float:
    if deployable_gap_pct <= 0.0:
        return 0.0
    trim_aggression = str(execution_guidance.get("trim_aggression") or "").strip().lower()
    burst_bias = str(execution_guidance.get("burst_execution_bias") or "").strip().lower()

    gap_share = 0.50
    hard_cap = 30.0
    if archetype == "special_situation":
        gap_share = 0.40
        hard_cap = 25.0

    if trim_aggression == "high" or burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
        gap_share = min(gap_share, 0.40)
        hard_cap = min(hard_cap, 22.5)

    return max(min(deployable_gap_pct * gap_share, hard_cap, deployable_gap_pct), 0.0)


def _reason_for_allocator(
    *,
    template: Mapping[str, Any],
    sector_regime: str,
    best_edge: float | None,
    t10_edge: float | None,
    allow_add_on_strength: bool,
    human_overlay: Mapping[str, Any] | None,
) -> str:
    parts = [f"Archetype `{template['label']}`"]
    if sector_regime == "leader":
        parts.append("ngành đang dẫn")
    elif sector_regime == "laggard":
        parts.append("ngành đang yếu")
    if best_edge is not None:
        parts.append(f"best timing edge {_fmt_pct(best_edge)}")
    if t10_edge is not None and t10_edge < 0.0:
        parts.append("T+10 âm nên không nên giam vốn quá mức")
    if allow_add_on_strength:
        parts.append("được phép giữ sẵn ngân sách add-on-strength")
    else:
        parts.append("không ưu tiên add-on-strength")
    target_price = _to_float(human_overlay.get("target_price")) if human_overlay is not None else None
    target_year = human_overlay.get("target_year") if human_overlay is not None else None
    if target_price is not None:
        year_text = f" trong `{target_year}`" if target_year is not None else ""
        parts.append(f"human note giữ thesis target `{_fmt_price(target_price)}`{year_text}")
    return "; ".join(parts) + "."


def _allocate_budget_with_caps(
    *,
    scores: Mapping[str, float],
    caps: Mapping[str, float],
    total_budget: float,
) -> Dict[str, float]:
    allocations = {ticker: 0.0 for ticker in scores}
    remaining_budget = max(total_budget, 0.0)
    active = {
        ticker
        for ticker, score in scores.items()
        if score > 0.0 and (caps.get(ticker) or 0.0) > 0.0
    }
    while remaining_budget > 1e-9 and active:
        total_score = sum(scores[ticker] for ticker in active)
        if total_score <= 0.0:
            break
        spent = 0.0
        exhausted: List[str] = []
        for ticker in list(active):
            share = remaining_budget * scores[ticker] / total_score
            headroom = max((caps.get(ticker) or 0.0) - allocations[ticker], 0.0)
            piece = min(share, headroom)
            if piece > 0.0:
                allocations[ticker] += piece
                spent += piece
            if headroom - piece <= 1e-9:
                exhausted.append(ticker)
        if spent <= 1e-9:
            break
        remaining_budget = max(remaining_budget - spent, 0.0)
        for ticker in exhausted:
            active.discard(ticker)
    return allocations


def _ticker_session_cap_pct(
    *,
    target: Mapping[str, Any],
    best_edge: float | None,
    strength_share: float,
) -> float:
    gap_to_min = max(_to_float(target.get("GapToMinWeightPct")) or 0.0, 0.0)
    gap_to_max = max(_to_float(target.get("GapToMaxWeightPct")) or 0.0, 0.0)
    weakness_cap = gap_to_min if _bool_text(target.get("AddOnWeaknessAllowed")) else 0.0
    strength_headroom = max(gap_to_max - gap_to_min, 0.0)
    strength_cap = 0.0
    if _bool_text(target.get("AddOnStrengthAllowed")) and strength_headroom > 0.0:
        strength_cap = min(strength_headroom, max(gap_to_max * strength_share, 2.0))
        if (best_edge or 0.0) > 1.0:
            strength_cap *= 1.1
    session_cap = min(weakness_cap + strength_cap, gap_to_max)
    if _bool_text(target.get("PersistentWeaknessBid")) and session_cap < min(gap_to_max, 2.0):
        session_cap = min(gap_to_max, max(session_cap, 2.0))
    return max(session_cap, 0.0)


def _allocate_global_buy_tranches(
    *,
    tickers: List[str],
    ticker_context: Mapping[str, Mapping[str, Any]],
    ticker_targets: Mapping[str, Dict[str, Any]],
    session_budget_pct: float,
    strength_share: float,
) -> Dict[str, Any]:
    tranche_scores: Dict[str, float] = {}
    tranche_caps: Dict[str, float] = {}
    tranche_allocations: Dict[str, float] = {}
    tranche_meta: Dict[str, Dict[str, Any]] = {}
    mandatory_floors: Dict[str, float] = {}

    for ticker in tickers:
        context = ticker_context[ticker]
        target = ticker_targets[ticker]
        blueprint = _derive_session_buy_tranche_blueprint(
            row=context["row"],
            ladder_df=context["ticker_ladder"],
            persistent_weakness_bid=bool(target.get("PersistentWeaknessBid")),
            allow_add_on_strength=bool(target.get("AddOnStrengthAllowed")),
            burst_summary=context["burst_summary"],
            execution_guidance=context["execution_guidance"],
        )
        target["TrancheBlueprintSummary"] = blueprint["summary"]
        ticker_session_cap_pct = _ticker_session_cap_pct(
            target=target,
            best_edge=context["timing_summary"].get("best_net_edge"),
            strength_share=strength_share,
        )
        target["TickerSessionCapPct"] = round(ticker_session_cap_pct, 2)
        target["SessionBuyTranches"] = []
        for tranche in blueprint["tranches"]:
            share_pct = _to_float(tranche.get("PlanSharePctOfTickerSession")) or 0.0
            max_cap_pct = ticker_session_cap_pct * share_pct / 100.0
            if max_cap_pct <= 0.0:
                continue
            key = f"{ticker}|{tranche.get('Role')}|{tranche.get('LimitPrice')}"
            base_score = max(_to_float(tranche.get("BaseTrancheScore")) or 0.0, 0.0)
            score = base_score * max(context["conviction_score"], 0.1)
            role = str(tranche.get("Role") or "")
            burst_bias = str(context["execution_guidance"].get("burst_execution_bias") or "").strip().lower()
            execution_bias = str(context["tape_summary"].get("execution_bias") or "").strip().lower()
            if role in {"bridge", "continuation_reserve"} and not _bool_text(target.get("AddOnStrengthAllowed")):
                score *= 0.88
            if role in {"bridge", "continuation_reserve"} and burst_bias in {"failed_followthrough", "failed_day2_followthrough"}:
                score *= 0.86
            if execution_bias == "distribution" and role in {"continuation_reserve", "bridge", "shallow_core"}:
                score *= 0.9
            if execution_bias == "absorption" and role in {"bridge", "continuation_reserve"}:
                score *= 1.05
            if _bool_text(target.get("PersistentWeaknessBid")) and role == "deep_core":
                score *= 1.12
            if (context["timing_summary"].get("t10_net_edge") or 0.0) < 0.0 and role in {"bridge", "continuation_reserve"}:
                score *= 0.9
            tranche_scores[key] = max(score, 0.01)
            tranche_caps[key] = round(max_cap_pct, 4)
            tranche_meta[key] = {
                "Ticker": ticker,
                "Role": role,
                "LimitPrice": _to_float(tranche.get("LimitPrice")),
                "EntryScoreRank": _to_float(tranche.get("EntryScoreRank")),
                "Reason": tranche.get("Reason"),
                "MaxCapitalPctOfPortfolio": round(max_cap_pct, 2),
            }
            floor_pct = _to_float(tranche.get("MandatoryFloorPctOfTickerCap")) or 0.0
            if floor_pct > 0.0:
                mandatory_floors[key] = min(max_cap_pct, ticker_session_cap_pct * floor_pct / 100.0)

    floor_total = sum(mandatory_floors.values())
    if floor_total > session_budget_pct and floor_total > 0.0:
        scale = session_budget_pct / floor_total
        mandatory_floors = {key: value * scale for key, value in mandatory_floors.items()}
    for key, value in mandatory_floors.items():
        tranche_allocations[key] = value

    remaining_budget_pct = max(session_budget_pct - sum(tranche_allocations.values()), 0.0)
    residual_caps = {
        key: max(tranche_caps[key] - tranche_allocations.get(key, 0.0), 0.0)
        for key in tranche_caps
    }
    residual_allocations = _allocate_budget_with_caps(
        scores=tranche_scores,
        caps=residual_caps,
        total_budget=remaining_budget_pct,
    )
    for key, value in residual_allocations.items():
        tranche_allocations[key] = tranche_allocations.get(key, 0.0) + value

    ranked_tranches: List[Dict[str, Any]] = []
    per_ticker_tranches: Dict[str, List[Dict[str, Any]]] = {ticker: [] for ticker in tickers}
    for key, allocated_pct in tranche_allocations.items():
        if allocated_pct <= 0.0:
            continue
        meta = tranche_meta[key]
        tranche = {
            **meta,
            "Score": round(tranche_scores[key], 6),
            "AllocatedCapitalPctOfPortfolio": round(allocated_pct, 2),
        }
        ranked_tranches.append(tranche)
        per_ticker_tranches[meta["Ticker"]].append(tranche)

    ranked_tranches.sort(key=lambda item: (-item["Score"], item["Ticker"], item["LimitPrice"]))
    for ticker in tickers:
        per_ticker_tranches[ticker].sort(key=lambda item: item["LimitPrice"], reverse=True)
    return {
        "RankedTranches": ranked_tranches,
        "PerTickerTranches": per_ticker_tranches,
    }


def _build_allocator(
    *,
    tickers: List[str],
    universe_df: pd.DataFrame,
    ticker_context: Mapping[str, Mapping[str, Any]],
    market_summary: Mapping[str, Any],
    total_capital_kvnd: int | None,
) -> Dict[str, Any]:
    target_invested_pct, strength_share = _portfolio_target_invested_pct(market_summary)
    single_name_mode = len(tickers) == 1
    current_invested_pct = 0.0
    current_equity_value_kvnd = 0.0
    capital_base_kvnd = total_capital_kvnd
    effective_current_weights: Dict[str, float] = {}
    for ticker in tickers:
        row = ticker_context[ticker]["row"]
        position_market_value_kvnd = max(_to_float(row.get("PositionMarketValue_kVND")) or 0.0, 0.0)
        current_equity_value_kvnd += position_market_value_kvnd
        effective_current_weights[ticker] = position_market_value_kvnd
    if capital_base_kvnd is not None and capital_base_kvnd > 0:
        current_invested_pct = (current_equity_value_kvnd / capital_base_kvnd) * 100.0
        for ticker in tickers:
            effective_current_weights[ticker] = (effective_current_weights[ticker] / capital_base_kvnd) * 100.0
    else:
        for ticker in tickers:
            row = ticker_context[ticker]["row"]
            effective_current_weights[ticker] = max(_to_float(row.get("PositionWeightPct")) or 0.0, 0.0)
            current_invested_pct += effective_current_weights[ticker]
        capital_base_kvnd = max(_to_float(universe_df["EnginePortfolioMarketValue_kVND"].dropna().iloc[0]) or 0.0, 0.0) or None
    deployable_gap_pct = max(target_invested_pct - current_invested_pct, 0.0)
    strength_budget_pct = deployable_gap_pct * strength_share
    weakness_budget_pct = max(deployable_gap_pct - strength_budget_pct, 0.0)

    weakness_scores: Dict[str, float] = {}
    strength_scores: Dict[str, float] = {}
    weakness_caps: Dict[str, float] = {}
    strength_caps: Dict[str, float] = {}
    ticker_targets: Dict[str, Dict[str, Any]] = {}

    for ticker in tickers:
        context = ticker_context[ticker]
        row = context["row"]
        template = context["template"]
        timing_summary = context["timing_summary"]
        conviction = context["conviction_score"]
        human_overlay = context["human_overlay"]
        current_weight = effective_current_weights[ticker]
        min_weight, max_weight = context["target_band"]
        gap_to_min = max(min_weight - current_weight, 0.0)
        headroom_to_max = max(max_weight - current_weight, 0.0)
        human_target_price = _to_float(human_overlay.get("target_price")) if human_overlay is not None else None
        last_price = _to_float(row.get("Last")) or 0.0
        human_target_multiple = (human_target_price / last_price) if human_target_price is not None and last_price > 0.0 else None
        if single_name_mode:
            single_name_buffer = 25.0 if human_target_multiple is not None and human_target_multiple >= 1.8 else 15.0
            min_weight = max(min_weight, target_invested_pct)
            max_weight = max(max_weight, min(target_invested_pct + single_name_buffer, 95.0))
            gap_to_min = max(min_weight - current_weight, 0.0)
            headroom_to_max = max(max_weight - current_weight, 0.0)
        persistent_weakness_capacity = 0.0
        if human_target_multiple is not None and human_target_multiple >= 1.8 and headroom_to_max > 0.0:
            persistent_weakness_capacity = headroom_to_max
        allow_strength = bool(template["allow_add_on_strength"]) and headroom_to_max > 0.0 and (timing_summary.get("best_net_edge") or 0.0) > 0.0
        allow_weakness = bool(template["allow_add_on_weakness"]) and (gap_to_min > 0.0 or persistent_weakness_capacity > 0.0)
        weakness_capacity = gap_to_min if gap_to_min > 0.0 else persistent_weakness_capacity
        weakness_scores[ticker] = weakness_capacity * conviction if allow_weakness else 0.0
        strength_scores[ticker] = headroom_to_max * max(conviction, 0.5) if allow_strength else 0.0
        weakness_caps[ticker] = weakness_capacity
        strength_caps[ticker] = headroom_to_max if allow_strength else 0.0
        ticker_targets[ticker] = {
            "CurrentWeightPct": current_weight,
            "TargetWeightMinPct": min_weight,
            "TargetWeightMaxPct": max_weight,
            "GapToMinWeightPct": gap_to_min,
            "GapToMaxWeightPct": headroom_to_max,
            "AddOnStrengthAllowed": allow_strength,
            "AddOnWeaknessAllowed": allow_weakness,
            "PersistentWeaknessBid": persistent_weakness_capacity > 0.0,
            "PreferredBuildStyle": (
                "persistent_passive_ladder"
                if persistent_weakness_capacity > 0.0 and not allow_strength
                else "mixed_strength_and_weakness"
                if allow_strength and allow_weakness
                else "weakness_only"
                if allow_weakness
                else "strength_only"
                if allow_strength
                else "hold_or_trim"
            ),
            "AllocatorReason": _reason_for_allocator(
                template=template,
                sector_regime=context["sector_regime"],
                best_edge=timing_summary.get("best_net_edge"),
                t10_edge=timing_summary.get("t10_net_edge"),
                allow_add_on_strength=allow_strength,
                human_overlay=human_overlay,
            ),
        }

    session_build_cap_pct = deployable_gap_pct
    deferred_build_pct = 0.0
    if single_name_mode and tickers:
        only_ticker = tickers[0]
        session_build_cap_pct = _single_name_session_build_cap_pct(
            deployable_gap_pct=deployable_gap_pct,
            archetype=ticker_context[only_ticker]["archetype"],
            execution_guidance=ticker_context[only_ticker]["execution_guidance"],
        )
        deferred_build_pct = max(deployable_gap_pct - session_build_cap_pct, 0.0)

    if single_name_mode:
        weakness_total = sum(weakness_scores.values())
        strength_total = sum(strength_scores.values())
        if weakness_total <= 0.0:
            strength_budget_pct = deployable_gap_pct
            weakness_budget_pct = 0.0
        if strength_total <= 0.0:
            weakness_budget_pct = deployable_gap_pct
            strength_budget_pct = 0.0
        if deployable_gap_pct > 0.0 and session_build_cap_pct < deployable_gap_pct:
            scale = session_build_cap_pct / deployable_gap_pct
            weakness_budget_pct *= scale
            strength_budget_pct *= scale

        weakness_allocations = _allocate_budget_with_caps(
            scores=weakness_scores,
            caps=weakness_caps,
            total_budget=weakness_budget_pct,
        )
        residual_strength_caps = {
            ticker: max((strength_caps.get(ticker) or 0.0) - (weakness_allocations.get(ticker) or 0.0), 0.0)
            for ticker in tickers
        }
        strength_allocations = _allocate_budget_with_caps(
            scores=strength_scores,
            caps=residual_strength_caps,
            total_budget=strength_budget_pct,
        )

        for ticker in tickers:
            weakness_alloc = weakness_allocations.get(ticker, 0.0)
            strength_alloc = strength_allocations.get(ticker, 0.0)
            target = ticker_targets[ticker]
            target["WeaknessBuildPct"] = round(weakness_alloc, 2)
            target["StrengthReservePct"] = round(strength_alloc, 2)
            target["SuggestedNewCapitalPct"] = round(weakness_alloc + strength_alloc, 2)
            target["DeferredBuildPct"] = round(max((target.get("GapToMinWeightPct") or 0.0) - (weakness_alloc + strength_alloc), 0.0), 2)
            target["SessionBuyTranches"] = []
            target["SessionBuyPlanSummary"] = None
    else:
        global_buy = _allocate_global_buy_tranches(
            tickers=tickers,
            ticker_context=ticker_context,
            ticker_targets=ticker_targets,
            session_budget_pct=session_build_cap_pct,
            strength_share=strength_share,
        )
        strength_budget_pct = 0.0
        weakness_budget_pct = session_build_cap_pct
        for ticker in tickers:
            target = ticker_targets[ticker]
            tranches = global_buy["PerTickerTranches"].get(ticker, [])
            target["SessionBuyTranches"] = tranches
            if tranches:
                target["SessionBuyPlanSummary"] = ", ".join(
                    f"{_fmt_price(_to_float(item.get('LimitPrice')))}: {_fmt_ratio(_to_float(item.get('AllocatedCapitalPctOfPortfolio')))}% portfolio ({item.get('Role')})"
                    for item in tranches
                )
            else:
                target["SessionBuyPlanSummary"] = "không có tranche đủ tốt trong phiên"
            total_alloc = sum((_to_float(item.get("AllocatedCapitalPctOfPortfolio")) or 0.0) for item in tranches)
            strength_alloc = sum(
                (_to_float(item.get("AllocatedCapitalPctOfPortfolio")) or 0.0)
                for item in tranches
                if item.get("Role") == "bridge" and _bool_text(target.get("AddOnStrengthAllowed"))
            )
            weakness_alloc = max(total_alloc - strength_alloc, 0.0)
            target["WeaknessBuildPct"] = round(weakness_alloc, 2)
            target["StrengthReservePct"] = round(strength_alloc, 2)
            target["SuggestedNewCapitalPct"] = round(total_alloc, 2)
            target["DeferredBuildPct"] = round(max((target.get("GapToMinWeightPct") or 0.0) - total_alloc, 0.0), 2)

    rationale = (
        f"Market regime `{_market_regime_vi(_market_regime(market_summary))}` => target invested `{target_invested_pct:.2f}%`; "
        f"current invested `{current_invested_pct:.2f}%`; deployable gap `{deployable_gap_pct:.2f}%`."
    )
    if capital_base_kvnd is not None and capital_base_kvnd > 0:
        rationale += (
            f" Capital base `{capital_base_kvnd:.0f} kVND`, "
            f"equity currently deployed `{current_equity_value_kvnd:.0f} kVND`."
        )
    if strength_budget_pct > 0.0:
        rationale += f" Giữ sẵn `{strength_budget_pct:.2f}%` làm strength reserve để tránh thiếu tiền khi mã chạy."
    if deferred_build_pct > 0.0:
        rationale += (
            f" Single-name mode chỉ cho phép dùng `{session_build_cap_pct:.2f}%` trong phiên kế tiếp "
            f"và giữ lại `{deferred_build_pct:.2f}%` cho các phiên sau."
        )
    elif not single_name_mode:
        rationale += (
            f" Multi-name mode dùng allocator toàn cục ở cấp tranche với session budget `{session_build_cap_pct:.2f}%`, "
            "không cố định quota vốn trước cho từng mã."
        )
    return {
        "CapitalBaseKVND": round(capital_base_kvnd, 2) if capital_base_kvnd is not None else None,
        "SingleNameMode": single_name_mode,
        "CurrentEquityValueKVND": round(current_equity_value_kvnd, 2),
        "TargetInvestedPct": round(target_invested_pct, 2),
        "CurrentInvestedPct": round(current_invested_pct, 2),
        "DeployableGapPct": round(deployable_gap_pct, 2),
        "SessionBuildCapPct": round(session_build_cap_pct, 2),
        "DeferredBuildPct": round(deferred_build_pct, 2),
        "StrengthBudgetPct": round(strength_budget_pct, 2),
        "WeaknessBudgetPct": round(weakness_budget_pct, 2),
        "Rationale": rationale,
        "GlobalBuyTranches": global_buy["RankedTranches"] if not single_name_mode else [],
        "Tickers": ticker_targets,
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative_output_path(path: Path, base_dir: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _build_profile_markdown(
    ticker: str,
    archetype: str,
    template: Mapping[str, Any],
    row: Mapping[str, Any],
    market_regime: str,
    sector_regime: str,
    cycle_best: Mapping[str, Any] | None,
) -> str:
    bullets = _build_objective_profile_bullets(row, sector_regime, cycle_best)
    return "\n".join(
        [
            f"# Hồ sơ {ticker}",
            "",
            f"Snapshot gốc: `{row.get('EngineRunAt')}`",
            f"Archetype: `{template['label']}`",
            f"Market regime nền: `{_market_regime_vi(market_regime)}`",
            "",
            "## Bản chất mã",
            "",
            *[f"- {line}" for line in template["profile_lines"]],
            "",
            "## Vì sao xếp loại này",
            "",
            *[f"- {line}" for line in bullets],
            "",
            "## Điều cần theo dõi dài hạn",
            "",
            *[f"- {line}" for line in template["focus_lines"]],
            "",
            "## Kiểu đánh mặc định",
            "",
            f"- Phong cách: `{template['style']}`.",
            f"- Khung nắm giữ ưu tiên: `{template['hold_window']}`.",
            *[f"- {line}" for line in template["order_lines"]],
        ]
    )


def _build_weekly_markdown(
    ticker: str,
    row: Mapping[str, Any],
    sector_row: Mapping[str, Any],
    market_summary: Mapping[str, Any],
    template: Mapping[str, Any],
    cycle_best: Mapping[str, Any] | None,
    timing_summary: Mapping[str, Any],
    burst_summary: Mapping[str, Any],
    human_overlay: Mapping[str, Any] | None,
    allocation_target: Mapping[str, Any],
    prior_state: Mapping[str, Any],
    week_label: str,
) -> str:
    inherited = prior_state.get("WeeklySummary") or prior_state.get("ProfileSummary") or "Không có state trước đó."
    sector_ret20 = _fmt_pct(_to_float(sector_row.get("SectorMedianRet20dVsIndex")))
    sector_breadth20 = _fmt_pct(_to_float(sector_row.get("SectorBreadthAboveSMA20Pct")), signed=False)
    cycle_line = (
        f"Cycle tốt nhất hiện tại là `{cycle_best.get('ForecastWindow')}` với peak kỳ vọng "
        f"`{_fmt_pct(_to_float(cycle_best.get('PredPeakRetPct')))}` "
        f"và drawdown kỳ vọng `{_fmt_pct(_to_float(cycle_best.get('PredDrawdownPct')))}`."
        if cycle_best is not None
        else "Không có cycle forecast."
    )
    if human_overlay is not None and _to_float(human_overlay.get("target_price")) is not None:
        year_suffix = f" trong {human_overlay.get('target_year')}" if human_overlay.get("target_year") else ""
        human_line = (
            f"Human note hiện giữ target `{_fmt_price(_to_float(human_overlay.get('target_price')))}{year_suffix}`; "
            "đây là strategic overlay, chỉ được trim tactical chứ không phủ định thesis."
        )
    else:
        human_line = "Không có human target có cấu trúc cho mã này."
    return "\n".join(
        [
            f"# Weekly research {ticker} - {week_label}",
            "",
            f"Snapshot gốc: `{row.get('EngineRunAt')}`",
            "",
            "## Kế thừa",
            "",
            f"- State tuần trước / profile trước đó: {inherited}",
            "",
            "## Trạng thái cấu trúc tuần này",
            "",
            f"- Thị trường nền: `{_market_regime_vi(_market_regime(market_summary))}` với breadth20 `{_fmt_pct(_to_float(market_summary.get('BreadthAboveSMA20Pct')), signed=False)}` và breadth5d `{_fmt_pct(_to_float(market_summary.get('BreadthPositive5dPct')), signed=False)}` ({_market_breadth_source_line(market_summary)}).",
            f"- Ngành `{row.get('Sector')}`: breadth20 `{sector_breadth20}`, ret20d vs index `{sector_ret20}`.",
            f"- Mã `{ticker}`: Ret20d `{_fmt_pct(_to_float(row.get('Ret20d')))} / Ret20dVsIndex {_fmt_pct(_to_float(row.get('Ret20dVsIndex')))} / RSI14 {_fmt_ratio(_to_float(row.get('RSI14')))} / ATR14Pct {_fmt_pct(_to_float(row.get('ATR14Pct')), signed=False)}`.",
            f"- {cycle_line}",
            f"- Timing ngắn hạn: {timing_summary['summary']}",
            f"- Burst / T+2.5 pattern: {burst_summary['summary']}",
            f"- {human_line}",
            f"- Kỷ luật phân bổ: target weight `{_fmt_ratio(allocation_target.get('TargetWeightMinPct'))}% - {_fmt_ratio(allocation_target.get('TargetWeightMaxPct'))}%`, current `{_fmt_ratio(allocation_target.get('CurrentWeightPct'))}%`, build phiên tới `{_fmt_ratio(allocation_target.get('SuggestedNewCapitalPct'))}%` và còn defer `{_fmt_ratio(allocation_target.get('DeferredBuildPct'))}%` của tổng vốn.",
            "",
            "## Điều chỉnh thesis tuần",
            "",
            f"- Archetype giữ nguyên là `{template['label']}`.",
            f"- Cách đánh tuần này vẫn ưu tiên `{template['style']}`.",
            (
                "- Vì timing dài hơn đã suy yếu, không được kéo thời gian giữ vô điều kiện."
                if (timing_summary.get("t10_net_edge") or 0.0) < 0.0
                else "- Timing chưa gãy, có thể giữ thesis tuần nếu tape không xấu đi."
            ),
            "",
            "## Hướng dẫn tuần",
            "",
            *[f"- {line}" for line in template["order_lines"]],
        ]
    )


def _build_daily_markdown(
    ticker: str,
    row: Mapping[str, Any],
    template: Mapping[str, Any],
    timing_summary: Mapping[str, Any],
    ladder_summary: Mapping[str, Any],
    burst_summary: Mapping[str, Any],
    human_overlay: Mapping[str, Any] | None,
    ohlc_row: Mapping[str, Any] | None,
    intraday_report_row: Mapping[str, Any] | None,
    playbook_row: Mapping[str, Any] | None,
    cycle_best: Mapping[str, Any] | None,
    tape_summary: Mapping[str, Any],
    execution_guidance: Mapping[str, Any],
    allocation_target: Mapping[str, Any],
    session_buy_plan: Mapping[str, Any],
    prior_state: Mapping[str, Any],
) -> str:
    inherited = prior_state.get("DailySummary") or prior_state.get("WeeklySummary") or "Không có daily state trước đó."
    ohlc_line = (
        "OHLC phiên kế tiếp: "
        f"open/high/low/close dự báo `{_fmt_price(_to_float(ohlc_row.get('ForecastOpen')))} / {_fmt_price(_to_float(ohlc_row.get('ForecastHigh')))} / "
        f"{_fmt_price(_to_float(ohlc_row.get('ForecastLow')))} / {_fmt_price(_to_float(ohlc_row.get('ForecastClose')))}"
        "` "
        f", bias `{ohlc_row.get('ForecastCandleBias')}`."
        if ohlc_row is not None
        else "Không có OHLC forecast."
    )
    playbook_line = (
        f"Playbook live: `{playbook_row.get('StrategyFamily')}` / `{playbook_row.get('StrategyLabel')}`, "
        f"LatestSignal `{playbook_row.get('LatestSignal')}`."
        if playbook_row is not None
        else "Không có playbook live."
    )
    cycle_line = (
        f"Cycle tốt nhất hiện tại: `{cycle_best.get('ForecastWindow')}` với peak "
        f"`{_fmt_pct(_to_float(cycle_best.get('PredPeakRetPct')))}` "
        f"và drawdown `{_fmt_pct(_to_float(cycle_best.get('PredDrawdownPct')))}`."
        if cycle_best is not None
        else "Không có cycle forecast."
    )
    intraday_line = _summarise_intraday_report(pd.DataFrame([intraday_report_row]) if intraday_report_row is not None else pd.DataFrame())
    if human_overlay is not None and _to_float(human_overlay.get("target_price")) is not None:
        year_suffix = f" trong {human_overlay.get('target_year')}" if human_overlay.get("target_year") else ""
        human_line = (
            f"Human note: target `{_fmt_price(_to_float(human_overlay.get('target_price')))}{year_suffix}`; "
            "vẫn phải trade theo kỹ thuật nhưng mặc định luôn giữ ladder BUY sâu nếu còn headroom."
        )
    else:
        human_line = "Không có human target có cấu trúc."
    return "\n".join(
        [
            f"# Daily research {ticker}",
            "",
            f"Snapshot gốc: `{row.get('EngineRunAt')}`",
            "",
            "## Kế thừa",
            "",
            f"- State gần nhất: {inherited}",
            "",
            "## Tín hiệu hiện tại",
            "",
            f"- Giá / xu hướng: Last `{_fmt_price(_to_float(row.get('Last')))} / Ref {_fmt_price(_to_float(row.get('Ref')))} / Change {_fmt_pct(_to_float(row.get('ChangePct')))} / RSI14 {_fmt_ratio(_to_float(row.get('RSI14')))} / DistSMA20 {_fmt_pct(_to_float(row.get('DistSMA20Pct')))}.`",
            f"- Timing: {timing_summary['summary']}",
            f"- Ladder tốt nhất: {ladder_summary['summary']}.",
            f"- {cycle_line}",
            f"- {ohlc_line}",
            f"- Intraday rest-of-session: {intraday_line}",
            f"- {playbook_line}",
            f"- Burst / T+2.5 pattern: {burst_summary['summary']}",
            f"- Execution bias: `{tape_summary.get('execution_bias')}` / burst handling `{execution_guidance.get('burst_execution_bias')}` / trim aggression `{execution_guidance.get('trim_aggression')}`.",
            f"- Execution note: {execution_guidance.get('note')}",
            f"- {human_line}",
            f"- Kỷ luật size: target band `{_fmt_ratio(allocation_target.get('TargetWeightMinPct'))}% - {_fmt_ratio(allocation_target.get('TargetWeightMaxPct'))}%`, current `{_fmt_ratio(allocation_target.get('CurrentWeightPct'))}%`, weakness build phiên tới `{_fmt_ratio(allocation_target.get('WeaknessBuildPct'))}%`, strength reserve `{_fmt_ratio(allocation_target.get('StrengthReservePct'))}%`, defer `{_fmt_ratio(allocation_target.get('DeferredBuildPct'))}%`.",
            f"- Session buy plan: {session_buy_plan.get('summary')}.",
            (
                f"- Nếu có SELL, phải front-load tối thiểu `{_fmt_ratio(execution_guidance.get('must_sell_fraction_pct'))}%` lượng muốn trim theo mode `{execution_guidance.get('urgent_trim_mode')}`."
                if (execution_guidance.get("must_sell_fraction_pct") or 0.0) > 0.0
                else "- Không có yêu cầu front-load SELL bắt buộc ở snapshot hiện tại."
            ),
            "",
            "## Tape phiên gần nhất",
            "",
            f"- {tape_summary['summary']}",
            "",
            "## Hướng dẫn ra lệnh hôm nay",
            "",
            *[f"- {line}" for line in template["order_lines"]],
            (
                f"- Ladders hiện tại cho thấy vùng mua đẹp hơn nằm quanh `{_fmt_price(ladder_summary['buy_zone_low'])} - {_fmt_price(ladder_summary['buy_zone_high'])}`."
                if ladder_summary["buy_zone_low"] is not None and ladder_summary["buy_zone_high"] is not None
                else "- Không có vùng mua ladder đủ rõ; tránh ép lệnh nếu edge không sạch."
            ),
            (
                f"- Nếu cần chia size BUY, dùng session plan này trước: {session_buy_plan.get('summary')}."
                if session_buy_plan.get("tranches")
                else "- Không có session buy plan đủ rõ; có thể tự chia ladder nhưng vẫn ưu tiên nấc sâu."
            ),
            (
                "- Vì T+10 đã âm, nếu vị thế không chạy trong 3-5 phiên thì nên giảm tham vọng giữ vốn."
                if (timing_summary.get("t10_net_edge") or 0.0) < 0.0
                else "- Timing chưa gãy rõ; có thể giữ thesis ngắn hạn nếu tape xác nhận."
            ),
            f"- Execution hiện tại nghiêng `{execution_guidance.get('burst_execution_bias')}` với mức trim `{execution_guidance.get('trim_aggression')}`.",
            (
                f"- Nếu có SELL trong ngày, ít nhất `{_fmt_ratio(execution_guidance.get('must_sell_fraction_pct'))}%` lượng trim phải đặt ở tranche thực dụng trước; các tranche stretch mới đặt sau."
                if (execution_guidance.get("must_sell_fraction_pct") or 0.0) > 0.0
                else "- Nếu có SELL thì có thể dùng ladder bình thường, chưa cần tranche cưỡng bức để ưu tiên khớp."
            ),
        ]
    )


def _build_state_payload(
    ticker: str,
    row: Mapping[str, Any],
    market_summary: Mapping[str, Any],
    template: Mapping[str, Any],
    archetype: str,
    timing_summary: Mapping[str, Any],
    ladder_summary: Mapping[str, Any],
    burst_summary: Mapping[str, Any],
    human_overlay: Mapping[str, Any] | None,
    cycle_best: Mapping[str, Any] | None,
    ohlc_row: Mapping[str, Any] | None,
    tape_summary: Mapping[str, Any],
    execution_guidance: Mapping[str, Any],
    allocation_target: Mapping[str, Any],
    portfolio_allocator: Mapping[str, Any],
    session_buy_plan: Mapping[str, Any],
    profile_path: Path,
    weekly_path: Path,
    daily_path: Path,
    output_base_dir: Path,
) -> Dict[str, Any]:
    damage_level = None
    if cycle_best is not None:
        damage_level = _to_float(cycle_best.get("PredDrawdownPrice"))
    if damage_level is None:
        damage_level = tape_summary.get("damage_hint")
    bullish_confirm = None
    if ohlc_row is not None:
        bullish_confirm = _to_float(ohlc_row.get("ForecastHigh"))
    if bullish_confirm is None:
        bullish_confirm = _to_float(row.get("GridAbove_T1"))
    profile_summary = f"{ticker} được xếp vào nhóm {template['label']}."
    weekly_summary = (
        f"Luận điểm tuần hiện ưu tiên {template['style']}; "
        f"timing tốt nhất là {timing_summary.get('best_window') or 'không rõ'} với net edge {_fmt_pct(timing_summary.get('best_net_edge'))}; "
        f"burst pattern: {burst_summary['summary']}"
    )
    daily_summary = (
        f"Daily bias: giá hiện tại {_fmt_price(_to_float(row.get('Last')))}; "
        f"vùng buy tốt hơn nằm quanh {_fmt_price(ladder_summary.get('buy_zone_low'))} - {_fmt_price(ladder_summary.get('buy_zone_high'))}; "
        f"damage level gần nhất {_fmt_price(damage_level)}; "
        f"allocator build phiên tới {_fmt_ratio(allocation_target.get('SuggestedNewCapitalPct'))}% "
        f"và còn defer {_fmt_ratio(allocation_target.get('DeferredBuildPct'))}%."
    )
    return {
        "Ticker": ticker,
        "GeneratedAt": pd.Timestamp.utcnow().isoformat(),
        "EngineRunAt": str(row.get("EngineRunAt")),
        "Archetype": archetype,
        "ArchetypeLabel": template["label"],
        "TradingStyle": template["style"],
        "PreferredHoldWindow": template["hold_window"],
        "MarketRegime": _market_regime(market_summary),
        "MarketBreadthSource": market_summary.get("BreadthSource"),
        "MarketBreadthUniverseTickerCount": market_summary.get("BreadthUniverseTickerCount"),
        "ProfileSummary": profile_summary,
        "WeeklySummary": weekly_summary,
        "DailySummary": daily_summary,
        "DefaultAddOnStrengthAllowed": bool(template["allow_add_on_strength"]),
        "DefaultAddOnWeaknessAllowed": bool(template["allow_add_on_weakness"]),
        "AddOnStrengthAllowed": bool(allocation_target.get("AddOnStrengthAllowed")),
        "AddOnWeaknessAllowed": bool(allocation_target.get("AddOnWeaknessAllowed")),
        "PortfolioTargetInvestedPct": portfolio_allocator.get("TargetInvestedPct"),
        "PortfolioCurrentInvestedPct": portfolio_allocator.get("CurrentInvestedPct"),
        "PortfolioDeployableGapPct": portfolio_allocator.get("DeployableGapPct"),
        "PortfolioSessionBuildCapPct": portfolio_allocator.get("SessionBuildCapPct"),
        "PortfolioDeferredBuildPct": portfolio_allocator.get("DeferredBuildPct"),
        "PortfolioSingleNameMode": bool(portfolio_allocator.get("SingleNameMode")),
        "BestTimingWindow": timing_summary.get("best_window"),
        "BestTimingNetEdgePct": timing_summary.get("best_net_edge"),
        "T10NetEdgePct": timing_summary.get("t10_net_edge"),
        "BurstSampleCount": burst_summary.get("sample_count"),
        "BurstNextDayPositiveRate": burst_summary.get("next_day_positive_rate"),
        "BurstNextDayStrongRate": burst_summary.get("next_day_strong_rate"),
        "BurstThirdDayNegativeRate": burst_summary.get("third_day_negative_rate"),
        "BurstAvgThreeDayDrawdownPct": burst_summary.get("avg_three_day_drawdown_pct"),
        "LatestBurstSignalAge": burst_summary.get("latest_signal_age"),
        "ExecutionBias": tape_summary.get("execution_bias"),
        "OpeningSqueezeFailure": bool(tape_summary.get("opening_squeeze_failure")),
        "CloseInRangePct": tape_summary.get("close_in_range_pct"),
        "FadeFromHighPct": tape_summary.get("fade_from_high_pct"),
        "CloseVsVWAPPct": tape_summary.get("close_vs_vwap_pct"),
        "PMReturnPct": tape_summary.get("pm_return_pct"),
        "PMExcessVsIndexPct": tape_summary.get("pm_excess_vs_index_pct"),
        "BurstExecutionBias": execution_guidance.get("burst_execution_bias"),
        "TrimAggression": execution_guidance.get("trim_aggression"),
        "UrgentTrimMode": execution_guidance.get("urgent_trim_mode"),
        "MustSellFractionPct": execution_guidance.get("must_sell_fraction_pct"),
        "ExecutionNote": execution_guidance.get("note"),
        "CurrentWeightPct": allocation_target.get("CurrentWeightPct"),
        "TargetWeightMinPct": allocation_target.get("TargetWeightMinPct"),
        "TargetWeightMaxPct": allocation_target.get("TargetWeightMaxPct"),
        "GapToMinWeightPct": allocation_target.get("GapToMinWeightPct"),
        "GapToMaxWeightPct": allocation_target.get("GapToMaxWeightPct"),
        "WeaknessBuildPct": allocation_target.get("WeaknessBuildPct"),
        "StrengthReservePct": allocation_target.get("StrengthReservePct"),
        "SuggestedNewCapitalPct": allocation_target.get("SuggestedNewCapitalPct"),
        "DeferredBuildPct": allocation_target.get("DeferredBuildPct"),
        "PreferredBuildStyle": allocation_target.get("PreferredBuildStyle"),
        "PersistentWeaknessBid": allocation_target.get("PersistentWeaknessBid"),
        "SessionBuyPlanSummary": session_buy_plan.get("summary"),
        "SessionBuyTranches": session_buy_plan.get("tranches"),
        "AllocatorReason": allocation_target.get("AllocatorReason"),
        "HumanOverlayText": human_overlay.get("raw_line") if human_overlay is not None else None,
        "HumanTargetPrice": _to_float(human_overlay.get("target_price")) if human_overlay is not None else None,
        "HumanTargetYear": human_overlay.get("target_year") if human_overlay is not None else None,
        "PreferredBuyZoneLow": ladder_summary.get("buy_zone_low"),
        "PreferredBuyZoneHigh": ladder_summary.get("buy_zone_high"),
        "BullishConfirmAbove": bullish_confirm,
        "DamageBelow": damage_level,
        "CyclePeakPrice": _to_float(cycle_best.get("PredPeakPrice")) if cycle_best is not None else None,
        "ForecastClose": _to_float(ohlc_row.get("ForecastClose")) if ohlc_row is not None else None,
        "Paths": {
            "Profile": _relative_output_path(profile_path, output_base_dir),
            "Weekly": _relative_output_path(weekly_path, output_base_dir),
            "Daily": _relative_output_path(daily_path, output_base_dir),
        },
    }


def build_research_bundle(
    *,
    universe_csv: Path,
    market_summary_json: Path,
    sector_summary_csv: Path,
    analysis_dir: Path,
    intraday_dir: Path,
    research_dir: Path,
    human_notes_path: Path | None = None,
    total_capital_kvnd: int | None = None,
    force_profile: bool = False,
    force_weekly: bool = False,
    force_daily: bool = False,
) -> Dict[str, Any]:
    universe_df = _read_csv(universe_csv, REQUIRED_UNIVERSE_COLUMNS, "Universe snapshot")
    market_summary = _read_json(market_summary_json, REQUIRED_MARKET_KEYS, "Market summary")
    sector_df = _read_csv(sector_summary_csv, REQUIRED_SECTOR_COLUMNS, "Sector summary")
    timing_df = _read_optional_csv(analysis_dir / "ml_single_name_timing.csv", REQUIRED_TIMING_COLUMNS, "Single-name timing")
    ladder_df = _read_optional_csv(analysis_dir / "ml_entry_ladder_eval.csv", REQUIRED_LADDER_COLUMNS, "Entry ladder eval")
    ohlc_df = _read_optional_csv(analysis_dir / "ml_ohlc_next_session.csv", REQUIRED_OHLC_COLUMNS, "OHLC next session")
    playbook_df = _read_optional_csv(
        analysis_dir / "ticker_playbooks_live" / "ticker_playbook_best_configs.csv",
        REQUIRED_PLAYBOOK_COLUMNS,
        "Ticker playbook best configs",
    )
    cycle_best_df = _read_optional_csv(
        analysis_dir / "ml_cycle_forecast" / "cycle_forecast_best_horizon_by_ticker.csv",
        REQUIRED_CYCLE_BEST_COLUMNS,
        "Cycle best horizon",
    )
    _read_optional_csv(analysis_dir / "ml_range_predictions_full_2y.csv", REQUIRED_RANGE_COLUMNS, "Range full_2y")
    _read_optional_csv(analysis_dir / "ml_range_predictions_recent_focus.csv", REQUIRED_RANGE_COLUMNS, "Range recent_focus")
    intraday_report_df = _read_optional_csv(
        analysis_dir / "ml_intraday_rest_of_session.csv",
        REQUIRED_INTRADAY_REPORT_COLUMNS,
        "Intraday rest-of-session",
    )
    human_overlays = _parse_human_notes(human_notes_path)
    daily_history_dir = intraday_dir.parent

    engine_run_at = pd.Timestamp(universe_df["EngineRunAt"].dropna().astype(str).iloc[0])
    snapshot_date = engine_run_at.tz_convert("Asia/Ho_Chi_Minh").date() if engine_run_at.tzinfo else engine_run_at.date()
    week_label = f"{snapshot_date.isocalendar().year}-W{snapshot_date.isocalendar().week:02d}"

    research_dir.mkdir(parents=True, exist_ok=True)
    tickers = [
        _normalise_ticker(raw)
        for raw in universe_df["Ticker"].dropna().tolist()
        if _normalise_ticker(raw) and _normalise_ticker(raw) != "VNINDEX"
    ]
    tickers = list(dict.fromkeys(tickers))
    ticker_root = research_dir / "tickers"
    ticker_root.mkdir(parents=True, exist_ok=True)
    active_tickers = set(tickers)
    for stale_dir in ticker_root.iterdir():
        if not stale_dir.is_dir():
            continue
        if stale_dir.name.upper() in active_tickers:
            continue
        shutil.rmtree(stale_dir)

    manifest: Dict[str, Any] = {
        "SchemaVersion": 1,
        "GeneratedAt": pd.Timestamp.utcnow().isoformat(),
        "EngineRunAt": str(engine_run_at),
        "SnapshotDate": snapshot_date.isoformat(),
        "UniverseTickers": tickers,
        "TotalCapitalKVND": total_capital_kvnd,
        "Tickers": {},
    }

    ticker_context: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        row_match = universe_df.loc[universe_df["Ticker"].astype(str).str.upper() == ticker]
        if row_match.empty:
            raise RuntimeError(f"Ticker {ticker} not found in universe snapshot")
        row = row_match.iloc[0].to_dict()
        sector_row = _pick_sector_row(sector_df, str(row.get("Sector", "")))
        sector_regime = _sector_regime(sector_row)
        archetype = _classify_archetype(row)
        template = _archetype_template(archetype)

        ticker_dir = research_dir / "tickers" / ticker
        profile_path = ticker_dir / "profile.md"
        weekly_path = ticker_dir / "weekly" / f"{week_label}.md"
        daily_path = ticker_dir / "daily" / f"{snapshot_date.isoformat()}.md"
        state_path = ticker_dir / "state.json"
        prior_state = _load_existing_json(state_path)

        ticker_timing = _pick_rows_by_ticker(timing_df, ticker)
        ticker_ladder = _pick_rows_by_ticker(ladder_df, ticker)
        ticker_intraday = _pick_rows_by_ticker(intraday_report_df, ticker)
        ohlc_row = _pick_first_by_ticker(ohlc_df, ticker)
        playbook_row = _pick_first_by_ticker(playbook_df, ticker)
        cycle_best = _pick_first_by_ticker(cycle_best_df, ticker)
        timing_summary = _summarise_timing(ticker_timing)
        ladder_summary = _summarise_best_ladders(ticker_ladder)
        tape_summary = _summarise_intraday_tape(intraday_dir, ticker)
        burst_summary = _summarise_burst_patterns(daily_history_dir, ticker)
        execution_guidance = _refine_burst_execution_guidance(burst_summary, tape_summary)
        human_overlay = human_overlays.get(ticker)
        conviction_score = _ticker_conviction_score(
            row=row,
            sector_regime=sector_regime,
            timing_summary=timing_summary,
            cycle_best=cycle_best,
            playbook_row=playbook_row,
        )
        target_band = _target_weight_band(
            archetype=archetype,
            row=row,
            sector_regime=sector_regime,
            timing_summary=timing_summary,
            human_overlay=human_overlay,
        )
        ticker_context[ticker] = {
            "row": row,
            "sector_row": sector_row,
            "sector_regime": sector_regime,
            "archetype": archetype,
            "template": template,
            "profile_path": profile_path,
            "weekly_path": weekly_path,
            "daily_path": daily_path,
            "state_path": state_path,
            "prior_state": prior_state,
            "timing_summary": timing_summary,
            "ticker_ladder": ticker_ladder,
            "ladder_summary": ladder_summary,
            "tape_summary": tape_summary,
            "burst_summary": burst_summary,
            "execution_guidance": execution_guidance,
            "human_overlay": human_overlay,
            "playbook_row": playbook_row,
            "cycle_best": cycle_best,
            "ohlc_row": ohlc_row,
            "ticker_intraday": ticker_intraday,
            "conviction_score": conviction_score,
            "target_band": target_band,
        }

    portfolio_allocator = _build_allocator(
        tickers=tickers,
        universe_df=universe_df,
        ticker_context=ticker_context,
        market_summary=market_summary,
        total_capital_kvnd=total_capital_kvnd,
    )
    manifest["PortfolioAllocator"] = portfolio_allocator

    for ticker in tickers:
        context = ticker_context[ticker]
        row = context["row"]
        sector_row = context["sector_row"]
        sector_regime = context["sector_regime"]
        archetype = context["archetype"]
        template = context["template"]
        profile_path = context["profile_path"]
        weekly_path = context["weekly_path"]
        daily_path = context["daily_path"]
        state_path = context["state_path"]
        prior_state = context["prior_state"]
        timing_summary = context["timing_summary"]
        ladder_summary = context["ladder_summary"]
        tape_summary = context["tape_summary"]
        burst_summary = context["burst_summary"]
        execution_guidance = context["execution_guidance"]
        human_overlay = context["human_overlay"]
        playbook_row = context["playbook_row"]
        cycle_best = context["cycle_best"]
        ohlc_row = context["ohlc_row"]
        ticker_intraday = context["ticker_intraday"]
        allocation_target = portfolio_allocator["Tickers"][ticker]
        if allocation_target.get("SessionBuyTranches"):
            session_buy_plan = {
                "summary": allocation_target.get("SessionBuyPlanSummary"),
                "tranches": allocation_target.get("SessionBuyTranches"),
            }
        else:
            session_buy_plan = _build_session_buy_tranches(
                row=row,
                ladder_df=ticker_ladder,
                allocation_target=allocation_target,
                burst_summary=burst_summary,
                execution_guidance=execution_guidance,
            )

        profile_markdown = _build_profile_markdown(
            ticker=ticker,
            archetype=archetype,
            template=template,
            row=row,
            market_regime=_market_regime(market_summary),
            sector_regime=sector_regime,
            cycle_best=cycle_best,
        )
        _write_text(profile_path, profile_markdown)

        weekly_markdown = _build_weekly_markdown(
            ticker=ticker,
            row=row,
            sector_row=sector_row,
            market_summary=market_summary,
            template=template,
            cycle_best=cycle_best,
            timing_summary=timing_summary,
            burst_summary=burst_summary,
            human_overlay=human_overlay,
            allocation_target=allocation_target,
            prior_state=prior_state,
            week_label=week_label,
        )
        _write_text(weekly_path, weekly_markdown)

        daily_markdown = _build_daily_markdown(
            ticker=ticker,
            row=row,
            template=template,
            timing_summary=timing_summary,
            ladder_summary=ladder_summary,
            burst_summary=burst_summary,
            human_overlay=human_overlay,
            ohlc_row=ohlc_row,
            intraday_report_row=_pick_first_by_ticker(ticker_intraday, ticker) if not ticker_intraday.empty else None,
            playbook_row=playbook_row,
            cycle_best=cycle_best,
            tape_summary=tape_summary,
            execution_guidance=execution_guidance,
            allocation_target=allocation_target,
            session_buy_plan=session_buy_plan,
            prior_state=prior_state,
        )
        _write_text(daily_path, daily_markdown)

        state_payload = _build_state_payload(
            ticker=ticker,
            row=row,
            market_summary=market_summary,
            template=template,
            archetype=archetype,
            timing_summary=timing_summary,
            ladder_summary=ladder_summary,
            burst_summary=burst_summary,
            human_overlay=human_overlay,
            cycle_best=cycle_best,
            ohlc_row=ohlc_row,
            tape_summary=tape_summary,
            execution_guidance=execution_guidance,
            allocation_target=allocation_target,
            portfolio_allocator=portfolio_allocator,
            session_buy_plan=session_buy_plan,
            profile_path=profile_path,
            weekly_path=weekly_path,
            daily_path=daily_path,
            output_base_dir=research_dir.parent,
        )
        _write_json(state_path, state_payload)

        manifest["Tickers"][ticker] = {
            "Archetype": archetype,
            "ProfilePath": _relative_output_path(profile_path, research_dir.parent),
            "WeeklyPath": _relative_output_path(weekly_path, research_dir.parent),
            "DailyPath": _relative_output_path(daily_path, research_dir.parent),
            "StatePath": _relative_output_path(state_path, research_dir.parent),
            "PreferredHoldWindow": template["hold_window"],
            "DefaultAllowAddOnStrength": bool(template["allow_add_on_strength"]),
            "DefaultAllowAddOnWeakness": bool(template["allow_add_on_weakness"]),
            "AllowAddOnStrength": bool(allocation_target.get("AddOnStrengthAllowed")),
            "AllowAddOnWeakness": bool(allocation_target.get("AddOnWeaknessAllowed")),
            "TargetWeightMinPct": allocation_target.get("TargetWeightMinPct"),
            "TargetWeightMaxPct": allocation_target.get("TargetWeightMaxPct"),
            "SuggestedNewCapitalPct": allocation_target.get("SuggestedNewCapitalPct"),
            "DeferredBuildPct": allocation_target.get("DeferredBuildPct"),
            "StrengthReservePct": allocation_target.get("StrengthReservePct"),
            "PersistentWeaknessBid": bool(allocation_target.get("PersistentWeaknessBid")),
            "SessionBuyPlanSummary": session_buy_plan.get("summary"),
            "SessionBuyTranches": session_buy_plan.get("tranches"),
            "PortfolioSingleNameMode": bool(portfolio_allocator.get("SingleNameMode")),
            "HumanTargetPrice": _to_float(human_overlay.get("target_price")) if human_overlay is not None else None,
            "ExecutionBias": tape_summary.get("execution_bias"),
            "BurstExecutionBias": execution_guidance.get("burst_execution_bias"),
            "TrimAggression": execution_guidance.get("trim_aggression"),
            "UrgentTrimMode": execution_guidance.get("urgent_trim_mode"),
            "MustSellFractionPct": execution_guidance.get("must_sell_fraction_pct"),
        }

    manifest_path = research_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-ticker research artifacts (profile / weekly / daily) for the live working universe."
    )
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--market-summary-json", type=Path, default=DEFAULT_MARKET_SUMMARY_JSON)
    parser.add_argument("--sector-summary-csv", type=Path, default=DEFAULT_SECTOR_SUMMARY_CSV)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--intraday-dir", type=Path, default=DEFAULT_INTRADAY_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--human-notes-path", type=Path, default=None)
    parser.add_argument("--total-capital-kvnd", type=int, default=None)
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--force-weekly", action="store_true")
    parser.add_argument("--force-daily", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_research_bundle(
        universe_csv=args.universe_csv,
        market_summary_json=args.market_summary_json,
        sector_summary_csv=args.sector_summary_csv,
        analysis_dir=args.analysis_dir,
        intraday_dir=args.intraday_dir,
        research_dir=args.research_dir,
        human_notes_path=args.human_notes_path,
        total_capital_kvnd=args.total_capital_kvnd,
        force_profile=bool(args.force_profile),
        force_weekly=bool(args.force_weekly),
        force_daily=bool(args.force_daily),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
