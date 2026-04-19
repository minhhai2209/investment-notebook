from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "codex_universe"

UNIVERSE_COLUMNS = [
    "Ticker",
    "Sector",
    "IsVN30",
    "Last",
    "Ref",
    "ChangePct",
    "Vol20Pct",
    "Vol60Pct",
    "Beta60_Index",
    "Corr60_Index",
    "SMA20",
    "SMA50",
    "SMA200",
    "EMA20",
    "RSI14",
    "ATR14",
    "ATR14Pct",
    "MACD",
    "MACDSignal",
    "Z20",
    "Ret5d",
    "Ret20d",
    "Ret60d",
    "Pos52wPct",
    "ADTV20_shares",
    "IntradayVol_shares",
    "IntradayValue_kVND",
    "IntradayPctADV20",
    "ForeignFlowDate",
    "NetBuySellForeign_shares_1d",
    "NetBuySellForeign_shares_5d",
    "NetBuySellForeign_shares_20d",
    "NetBuySellForeign_kVND_1d",
    "NetBuySellForeign_kVND_5d",
    "NetBuySellForeign_kVND_20d",
    "ProprietaryFlowDate",
    "NetBuySellProprietary_shares_1d",
    "NetBuySellProprietary_shares_5d",
    "NetBuySellProprietary_shares_20d",
    "NetBuySellProprietary_kVND_1d",
    "NetBuySellProprietary_kVND_5d",
    "NetBuySellProprietary_kVND_20d",
    "ForeignRoomRemaining_shares",
    "ForeignHoldingPct",
    "ADTV20Rank",
    "ADTV20PctRank",
    "Ceil",
    "Floor",
    "TickSize",
    "LotSize",
    "SlippageOneTickPct",
    "FloorValid",
    "CeilValid",
    "ValidBid1",
    "ValidAsk1",
    "TicksToFloor",
    "TicksToCeil",
    "PE_fwd",
    "PB",
    "ROE",
    "DistRefPct",
    "DistSMA20Pct",
    "DistSMA50Pct",
    "GridBelow_T1",
    "GridBelow_T2",
    "GridBelow_T3",
    "GridAbove_T1",
    "GridAbove_T2",
    "GridAbove_T3",
    "High52w",
    "Low52w",
    "OneLotATR_kVND",
    "VNINDEX_ATR14PctRank",
    "SectorADTVRank",
    "Corr20_Index",
    "Beta20_Index",
    "CoMoveWithIndex20Pct",
    "Ret20dVsIndex",
    "Ret60dVsIndex",
    "Ret20dVsSector",
    "Ret60dVsSector",
    "RelStrength20Rank",
    "RelStrength60Rank",
    "SectorBreadthAboveSMA20Pct",
    "SectorBreadthAboveSMA50Pct",
    "SectorBreadthPositive5dPct",
    "EngineRunAt",
    "PositionQuantity",
    "PositionPctADV20",
    "PositionAvgPrice",
    "PositionMarketValue_kVND",
    "PositionWeightPct",
    "EnginePortfolioMarketValue_kVND",
    "PositionCostBasis_kVND",
    "PositionUnrealized_kVND",
    "PositionATR_kVND",
    "PositionPNLPct",
    "SectorWeightPct",
    "BetaContribution",
]

MARKET_SUMMARY_KEYS = [
    "VNINDEX_ATR14PctRank",
    "UniverseTickerCount",
    "IndexRangePos20",
    "IndexRangePos60",
    "IndexRangePos120",
    "IndexDistToUpper20Pct",
    "IndexDistToLower20Pct",
    "IndexDistToUpper60Pct",
    "IndexDistToLower60Pct",
    "IndexDistToUpper120Pct",
    "IndexDistToLower120Pct",
    "IndexDrawdownFromHigh60Pct",
    "IndexReboundFromLow60Pct",
    "BreadthAboveSMA20Pct",
    "BreadthAboveSMA50Pct",
    "BreadthAboveSMA200Pct",
    "BreadthPositive1dPct",
    "BreadthPositive5dPct",
    "AdvanceDeclineRatio",
    "NewHigh20Pct",
    "NewLow20Pct",
    "MarketDispersion20Pct",
    "MarketCoMovement20Pct",
    "MarketMedianCorr20",
    "GeneratedAt",
]

SECTOR_SUMMARY_COLUMNS = [
    "EngineRunAt",
    "Sector",
    "TickerCount",
    "SectorBreadthAboveSMA20Pct",
    "SectorBreadthAboveSMA50Pct",
    "SectorBreadthPositive5dPct",
    "SectorMedianRet5d",
    "SectorMedianRet20d",
    "SectorMedianRet60d",
    "SectorMedianRet20dVsIndex",
    "SectorMedianRet60dVsIndex",
    "SectorMeanCoMoveWithIndex20Pct",
    "SectorMedianForeignFlow5d_kVND",
    "SectorMedianForeignFlow20d_kVND",
    "SectorMedianProprietaryFlow5d_kVND",
    "SectorMedianProprietaryFlow20d_kVND",
    "SectorADTVRank",
]

RANGE_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "Horizon",
    "ForecastWindow",
    "Base",
    "Low",
    "Mid",
    "High",
    "PredLowRetPct",
    "PredMidRetPct",
    "PredHighRetPct",
    "RecentFocusWeight",
    "Full2YWeight",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
]

CYCLE_MATRIX_COLUMNS = ["Ticker"]
for horizon in ["1M", "2M", "3M", "4M", "5M", "6M"]:
    CYCLE_MATRIX_COLUMNS.extend(
        [
            f"Variant_{horizon}",
            f"Model_{horizon}",
            f"PredPeakRetPct_{horizon}",
            f"PredPeakDays_{horizon}",
            f"PredPeakPrice_{horizon}",
            f"PredDrawdownPct_{horizon}",
            f"PredDrawdownPrice_{horizon}",
            f"PeakRetMAEPct_{horizon}",
            f"PeakDayMAE_{horizon}",
            f"DrawdownMAEPct_{horizon}",
        ]
    )

CYCLE_BEST_COLUMNS = [
    "Ticker",
    "HorizonMonths",
    "ForecastWindow",
    "Variant",
    "Model",
    "PredPeakRetPct",
    "PredPeakDays",
    "PredDrawdownPct",
    "PeakRetMAEPct",
    "PeakDayMAE",
    "DrawdownMAEPct",
    "SelectionScore",
]

PLAYBOOK_COLUMNS = [
    "Ticker",
    "StrategyFamily",
    "StrategyLabel",
    "LatestSignal",
    "LatestSignalDate",
    "TrainScore",
    "TestScore",
    "AllScore",
    "RobustScore",
    "TrainTrades",
    "TestTrades",
    "AllTrades",
    "TrainWinRatePct",
    "TestWinRatePct",
    "AllWinRatePct",
    "TrainAvgReturnPct",
    "TestAvgReturnPct",
    "AllAvgReturnPct",
    "TrainAvgHoldDays",
    "TestAvgHoldDays",
    "AllAvgHoldDays",
    "TrainWorstDrawdownPct",
    "TestWorstDrawdownPct",
    "AllWorstDrawdownPct",
]

OHLC_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "Horizon",
    "ForecastWindow",
    "Base",
    "ForecastDate",
    "Model",
    "EvalRows",
    "OpenMAEPct",
    "HighMAEPct",
    "LowMAEPct",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
    "SelectionScore",
    "ForecastOpen",
    "ForecastHigh",
    "ForecastLow",
    "ForecastClose",
    "ForecastCloseRetPct",
    "ForecastRangePct",
    "ForecastCandleBias",
]

INTRADAY_COLUMNS = [
    "SnapshotDate",
    "SnapshotTimeBucket",
    "Ticker",
    "Base",
    "Low",
    "Mid",
    "High",
    "PredLowRetPct",
    "PredMidRetPct",
    "PredHighRetPct",
    "Model",
    "EvalRows",
    "CloseMAEPct",
    "RangeMAEPct",
    "CloseDirHitPct",
    "SelectionScore",
]

TIMING_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "Horizon",
    "ForecastWindow",
    "Base",
    "ForecastDate",
    "Variant",
    "Model",
    "EvalRows",
    "PeakRetMAEPct",
    "PeakDayMAE",
    "DrawdownMAEPct",
    "CloseMAEPct",
    "TradeScoreMAEPct",
    "TradeScoreHitPct",
    "SelectionScore",
    "PredPeakRetPct",
    "PredPeakDay",
    "PredDrawdownPct",
    "PredCloseRetPct",
    "PredPeakPrice",
    "PredDrawdownPrice",
    "PredClosePrice",
    "PredRewardRisk",
    "PredTradeScore",
    "PredNetEdgePct",
    "PredCapitalEfficiencyPctPerDay",
]

ENTRY_LADDER_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "PriceRank",
    "EntryScoreRank",
    "LimitPrice",
    "EntryAnchor",
    "Base",
    "TickSize",
    "LotSize",
    "EntryVsLastPct",
    "ForecastLowT1",
    "RangeLowBlendT5",
    "RangeLowBlendT10",
    "CycleDrawdownPrice",
    "EntryVsForecastLowT1Pct",
    "EntryVsRangeLowT5Pct",
    "EntryVsRangeLowT10Pct",
    "EntryVsCycleDrawdownPct",
    "NetRetToNextClosePct",
    "NetRetToNextHighPct",
    "BestTimingWindow",
    "BestTimingNetEdgePct",
    "BestTimingCloseRetPct",
    "BestTimingRewardRisk",
    "CycleNetEdgePct",
    "CycleRewardRisk",
    "FillScoreT1",
    "FillScoreT5",
    "FillScoreT10",
    "FillScoreCycle",
    "FillScoreComposite",
    "EntryScore",
]

STRATEGY_BUCKET_COLUMNS = [
    "Ticker",
    "StrategyBucket",
    "AllowNewBuy",
    "AllowAvgDown",
    "TargetState",
]

FILE_SPECS: List[Dict[str, Any]] = [
    {
        "path": "bundle_manifest.json",
        "kind": "manifest",
        "required": True,
        "summary": "Index kỹ thuật của toàn bộ live bundle; prompt phải đọc file này đầu tiên để biết schema/presence contract.",
        "usage_notes": [
            "Dùng manifest này làm contract chính cho file presence, schema tối thiểu và ý nghĩa từng file.",
            "Không tự suy diễn thêm cột/key hay schema khác ngoài manifest.",
        ],
    },
    {
        "path": "total_capital_kVND.txt",
        "kind": "text_number",
        "required": True,
        "summary": "Một số duy nhất, đơn vị kVND; xấp xỉ tổng vốn mục tiêu của danh mục.",
    },
    {
        "path": "universe.csv",
        "kind": "csv",
        "required": True,
        "summary": "Snapshot hợp nhất của VNINDEX và từng mã trong working universe, gồm kỹ thuật, execution, flows, relative strength và vị thế.",
        "required_columns": UNIVERSE_COLUMNS,
    },
    {
        "path": "market_summary.json",
        "kind": "json",
        "required": True,
        "summary": "Snapshot cấp thị trường: breadth, co-movement, range position và volatility regime của VNINDEX.",
        "required_keys": MARKET_SUMMARY_KEYS,
    },
    {
        "path": "sector_summary.csv",
        "kind": "csv",
        "required": True,
        "summary": "Snapshot cấp ngành: breadth, median returns, relative strength và flow summary.",
        "required_columns": SECTOR_SUMMARY_COLUMNS,
    },
    {
        "path": "ml_range_predictions_full_2y.csv",
        "kind": "csv",
        "required": True,
        "summary": "Forecast range nhiều horizon từ biến thể full_2y.",
        "required_columns": RANGE_COLUMNS,
    },
    {
        "path": "ml_range_predictions_recent_focus.csv",
        "kind": "csv",
        "required": True,
        "summary": "Forecast range nhiều horizon từ biến thể recent_focus.",
        "required_columns": RANGE_COLUMNS,
    },
    {
        "path": "ml_cycle_forecast_ticker_matrix.csv",
        "kind": "csv",
        "required": True,
        "summary": "Ma trận cycle forecast 1M..6M theo từng mã.",
        "required_columns": CYCLE_MATRIX_COLUMNS,
    },
    {
        "path": "ml_cycle_forecast_best_horizon.csv",
        "kind": "csv",
        "required": True,
        "summary": "Khung cycle tốt nhất hiện tại cho từng mã.",
        "required_columns": CYCLE_BEST_COLUMNS,
    },
    {
        "path": "ticker_playbook_best_configs.csv",
        "kind": "csv",
        "required": True,
        "summary": "Playbook kỹ thuật tốt nhất hiện hành theo từng mã.",
        "required_columns": PLAYBOOK_COLUMNS,
    },
    {
        "path": "ml_ohlc_next_session.csv",
        "kind": "csv",
        "required": True,
        "summary": "Forecast OHLC cho phiên giao dịch kế tiếp.",
        "required_columns": OHLC_COLUMNS,
    },
    {
        "path": "human_notes.md",
        "kind": "markdown",
        "required": False,
        "summary": "Overlay chỉ thị/giả định của người dùng, không phải market feed khách quan.",
        "usage_notes": [
            "Nếu file hiện diện thì phải đọc trước khi phân tích vốn, tin tức và market regime.",
            "Nếu note mâu thuẫn dữ liệu khách quan thì phải nêu rõ mâu thuẫn nhưng vẫn phản ánh overlay đó trong phạm vi hard constraints.",
        ],
    },
    {
        "path": "ml_intraday_rest_of_session.csv",
        "kind": "csv",
        "required": False,
        "summary": "Forecast intraday cho phần còn lại của phiên hiện tại.",
        "required_columns": INTRADAY_COLUMNS,
        "usage_notes": [
            "Chỉ dùng khi snapshot vẫn còn cửa giao dịch có thể thực thi trong cùng ngày.",
            "Nếu file không tồn tại thì không xem là lỗi; chỉ có nghĩa là hiện không có lớp forecast intraday bổ sung.",
        ],
    },
    {
        "path": "ml_single_name_timing.csv",
        "kind": "csv",
        "required": False,
        "summary": "Forecast timing ngắn hạn theo mã: peak, days-to-peak, drawdown, close return, net edge.",
        "required_columns": TIMING_COLUMNS,
        "usage_notes": [
            "Dùng như lớp đánh giá một vòng trade tập trung trong vài phiên tới.",
            "Nếu file không tồn tại thì không xem là lỗi; chỉ có nghĩa là hiện không có lớp timing single-name bổ sung.",
        ],
    },
    {
        "path": "ml_entry_ladder_eval.csv",
        "kind": "csv",
        "required": False,
        "summary": "Bảng chấm điểm từng nấc LimitPrice BUY hợp lệ theo mã.",
        "required_columns": ENTRY_LADDER_COLUMNS,
        "usage_notes": [
            "Dùng để so trade-off giữa độ sâu giá mua, upside/downside và khả năng chạm ước lượng của từng nấc giá.",
            "Nếu dữ liệu lịch sử đủ dày thì fill-score có thể phản ánh xác suất chạm giá học từ chính mã đó; nếu không đủ thì đó là score xấp xỉ.",
            "Nếu file không tồn tại thì không xem là lỗi.",
        ],
    },
    {
        "path": "strategy_buckets.csv",
        "kind": "csv",
        "required": False,
        "summary": "Bucket chiến lược theo mã để giới hạn hành động được phép.",
        "required_columns": STRATEGY_BUCKET_COLUMNS,
        "usage_notes": [
            "Đây là chỉ thị/giả định của người dùng, không phải market feed khách quan.",
            "Nếu file hiện diện thì phải đọc trước khi phân tích market regime hay xây bộ lệnh.",
            "Bucket chỉ giới hạn hành động được phép; không tự động đồng nghĩa với conviction cao.",
        ],
    },
    {
        "path": "research/manifest.json",
        "kind": "json",
        "required": False,
        "summary": "Index của research bundle tự động theo mã; nếu có phải đọc trước per-ticker analysis.",
        "required_keys": ["SchemaVersion", "UniverseTickers", "Tickers"],
        "usage_notes": [
            "Nếu file hiện diện thì phải đọc manifest trước, rồi đọc các artifact liên quan của từng mã trong working universe.",
            "Research không phải market feed khách quan thuần túy, nhưng cũng không được hạ xuống thành tham khảo cho có.",
            "Nếu research stale hoặc mâu thuẫn với lớp dữ liệu hiện tại thì phải nêu rõ khi sử dụng.",
        ],
    },
]


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _missing_items(actual: Iterable[str], required: Iterable[str]) -> List[str]:
    actual_set = {str(item) for item in actual}
    return [item for item in required if item not in actual_set]


def _validate_csv(path: Path, required_columns: Iterable[str]) -> Dict[str, Any]:
    frame = pd.read_csv(path, nrows=0)
    missing_columns = _missing_items(frame.columns, required_columns)
    return {
        "missing_columns": missing_columns,
        "column_count": len(frame.columns),
    }


def _validate_json(path: Path, required_keys: Iterable[str]) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing_keys = _missing_items(payload.keys(), required_keys)
    return {
        "missing_keys": missing_keys,
    }


def _validate_text_number(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"{path} is empty")
    int(content)
    return {"value_preview": content}


def _validate_markdown(path: Path) -> Dict[str, Any]:
    path.read_text(encoding="utf-8")
    return {}


def _extract_working_universe(bundle_dir: Path) -> List[str]:
    universe_path = bundle_dir / "universe.csv"
    if not universe_path.exists():
        return []
    frame = pd.read_csv(universe_path, usecols=["Ticker"])
    tickers = []
    for raw in frame["Ticker"].dropna().tolist():
        ticker = str(raw).strip().upper()
        if not ticker or ticker == "VNINDEX" or ticker in tickers:
            continue
        tickers.append(ticker)
    return tickers


def build_bundle_manifest(*, bundle_dir: Path, output_path: Path | None = None) -> Dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    output_path = (output_path or (bundle_dir / "bundle_manifest.json")).resolve()

    manifest: Dict[str, Any] = {
        "SchemaVersion": 1,
        "GeneratedAt": pd.Timestamp.utcnow().isoformat(),
        "BundleDir": bundle_dir.as_posix(),
        "WorkingUniverseTickers": _extract_working_universe(bundle_dir),
        "Files": {},
        "ReadHints": [
            "Đọc bundle_manifest.json trước để biết contract của bundle live.",
            "Dùng manifest này để kiểm tra file nào bắt buộc, file nào optional, và schema tối thiểu của từng file.",
            "Summary và UsageNotes của từng file là mô tả canonical cho purpose/semantics của bundle live.",
            "Sau manifest, đọc human_notes/strategy_buckets/research theo rule của prompt live.",
        ],
    }

    errors: List[str] = []
    for spec in FILE_SPECS:
        rel_path = str(spec["path"])
        path = bundle_dir / rel_path
        entry: Dict[str, Any] = {
            "Path": rel_path,
            "Kind": spec["kind"],
            "Required": bool(spec["required"]),
            "Summary": spec["summary"],
            "Exists": path.exists(),
        }
        if spec.get("usage_notes"):
            entry["UsageNotes"] = list(spec["usage_notes"])
        try:
            if spec["kind"] == "manifest":
                entry["Exists"] = True
            elif spec["kind"] == "directory":
                if path.exists() and not path.is_dir():
                    raise ValueError(f"{rel_path} is not a directory")
                entry["FileCount"] = sum(1 for child in path.rglob("*") if child.is_file()) if path.is_dir() else 0
            elif path.exists():
                if spec["kind"] == "csv":
                    entry["Validation"] = _validate_csv(path, spec.get("required_columns", []))
                    if spec.get("required_columns"):
                        entry["RequiredColumns"] = list(spec["required_columns"])
                elif spec["kind"] == "json":
                    entry["Validation"] = _validate_json(path, spec.get("required_keys", []))
                    if spec.get("required_keys"):
                        entry["RequiredKeys"] = list(spec["required_keys"])
                elif spec["kind"] == "text_number":
                    entry["Validation"] = _validate_text_number(path)
                elif spec["kind"] == "markdown":
                    entry["Validation"] = _validate_markdown(path)
            if spec["required"] and spec["kind"] != "manifest" and not path.exists():
                raise FileNotFoundError(f"Missing required bundle file: {rel_path}")
            validation = entry.get("Validation", {})
            if validation.get("missing_columns"):
                raise ValueError(f"{rel_path} missing required columns: {', '.join(validation['missing_columns'])}")
            if validation.get("missing_keys"):
                raise ValueError(f"{rel_path} missing required keys: {', '.join(validation['missing_keys'])}")
        except Exception as exc:
            entry["ValidationError"] = str(exc)
            if spec["required"] or path.exists():
                errors.append(str(exc))
        manifest["Files"][rel_path] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if errors:
        raise RuntimeError("Bundle manifest validation failed:\n- " + "\n- ".join(errors))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a machine-readable manifest for the live Codex bundle.")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle_manifest(bundle_dir=args.bundle_dir, output_path=args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
