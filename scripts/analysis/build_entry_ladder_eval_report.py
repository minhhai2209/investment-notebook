from __future__ import annotations

import argparse
import math
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.evaluate_ohlc_models import FEATURE_COLUMNS, _load_daily_ohlcv, build_ticker_ohlc_sample


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
OUTPUT_FILE_NAME = "ml_entry_ladder_eval.csv"
FILL_MODEL_METRICS_FILE_NAME = "ml_entry_ladder_fill_model_metrics.csv"
FILL_SELECTED_MODELS_FILE_NAME = "ml_entry_ladder_fill_selected_models.csv"
ROUND_TRIP_FEE_PCT = 0.06
DEFAULT_FILL_HORIZONS = (1, 5, 10)
DEFAULT_FILL_DEPTH_ATR_GRID = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
DEFAULT_FILL_MIN_TRAIN_DATES = 120
DEFAULT_FILL_HOLDOUT_DATES = 40
FILL_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + [
    "TickerATR14Pct",
    "EntryDepthAtr",
    "EntryVsBasePct",
]
REQUIRED_OUTPUT_COLUMNS = [
    "SnapshotDate",
    "Ticker",
    "PriceRank",
    "EntryScoreRank",
    "LimitPrice",
    "EntryAnchor",
    "EntryAnchorCategory",
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

TACTICAL_ENTRY_ANCHORS = {
    "valid_bid1",
    "grid_below_t1",
    "grid_below_t2",
    "grid_below_t3",
}
HISTORICAL_ENTRY_ANCHORS = {
    "forecast_low_t1",
    "range_low_blend_t5",
    "range_low_blend_t10",
    "cycle_drawdown",
    "atr_1x_below",
    "atr_1_5x_below",
}


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _normalise_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _load_universe_frame(universe_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(universe_csv)
    _require_columns(
        frame,
        [
            "Ticker",
            "EngineRunAt",
            "Last",
            "TickSize",
            "LotSize",
            "Floor",
            "Ceil",
            "ValidBid1",
            "GridBelow_T1",
            "GridBelow_T2",
            "GridBelow_T3",
            "ATR14",
            "ATR14Pct",
        ],
        "Universe snapshot",
    )
    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    frame = frame[(frame["Ticker"] != "") & (frame["Ticker"] != "VNINDEX")].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{universe_csv} did not provide any non-index ticker")
    return frame


def _validate_ticker_coverage(
    label: str,
    frame: pd.DataFrame,
    expected_tickers: Sequence[str],
) -> None:
    if "Ticker" not in frame.columns:
        raise ValueError(f"{label} missing required column 'Ticker'")
    actual = {_normalise_ticker(value) for value in frame["Ticker"].dropna().tolist()}
    missing = sorted(set(expected_tickers) - actual)
    if missing:
        preview = ", ".join(missing[:12])
        raise RuntimeError(f"{label} does not cover required tickers: {preview}")


def _load_ohlc_frame(path: Path, expected_tickers: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        [
            "Ticker",
            "ForecastLow",
            "ForecastHigh",
            "ForecastClose",
        ],
        "Next-session OHLC report",
    )
    frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    frame = frame[(frame["Ticker"] != "") & (frame["Ticker"] != "VNINDEX")].reset_index(drop=True)
    _validate_ticker_coverage("Next-session OHLC report", frame, expected_tickers)
    return frame


def _load_cycle_frame(path: Path, expected_tickers: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        [
            "Ticker",
            "PredPeakPrice",
            "PredDrawdownPrice",
            "PredPeakRetPct",
            "PredPeakDays",
            "PredDrawdownPct",
        ],
        "Cycle forecast best-horizon report",
    )
    frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    frame = frame[(frame["Ticker"] != "") & (frame["Ticker"] != "VNINDEX")].reset_index(drop=True)
    _validate_ticker_coverage("Cycle forecast best-horizon report", frame, expected_tickers)
    return frame


def _load_single_name_frame(path: Path, expected_tickers: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        [
            "Ticker",
            "ForecastWindow",
            "PredPeakPrice",
            "PredDrawdownPrice",
            "PredClosePrice",
            "PredPeakDay",
        ],
        "Single-name timing report",
    )
    frame["Ticker"] = frame["Ticker"].map(_normalise_ticker)
    frame = frame[(frame["Ticker"] != "") & (frame["Ticker"] != "VNINDEX")].reset_index(drop=True)
    _validate_ticker_coverage("Single-name timing report", frame, expected_tickers)
    return frame


def _blend_range_variant_values(
    full_row: pd.Series,
    recent_row: pd.Series,
    value_column: str,
) -> float:
    full_weight = float(recent_row.get("Full2YWeight", full_row.get("Full2YWeight", 0.5)))
    recent_weight = float(recent_row.get("RecentFocusWeight", full_row.get("RecentFocusWeight", 0.5)))
    total_weight = full_weight + recent_weight
    if total_weight <= 0:
        full_weight = recent_weight = 0.5
        total_weight = 1.0
    return float(
        ((float(full_row[value_column]) * full_weight) + (float(recent_row[value_column]) * recent_weight))
        / total_weight
    )


def _load_range_blend_frame(
    full_path: Path,
    recent_path: Path,
    expected_tickers: Sequence[str],
) -> pd.DataFrame:
    full_frame = pd.read_csv(full_path)
    recent_frame = pd.read_csv(recent_path)
    required = [
        "Ticker",
        "ForecastWindow",
        "Low",
        "Mid",
        "High",
    ]
    _require_columns(full_frame, required, "Full-2Y range report")
    _require_columns(recent_frame, required + ["RecentFocusWeight", "Full2YWeight"], "Recent-focus range report")
    full_frame["Ticker"] = full_frame["Ticker"].map(_normalise_ticker)
    recent_frame["Ticker"] = recent_frame["Ticker"].map(_normalise_ticker)

    clean_rows: List[Dict[str, object]] = []
    keyed_full = {
        (_normalise_ticker(row["Ticker"]), str(row["ForecastWindow"]).strip()): row
        for _, row in full_frame.iterrows()
    }
    keyed_recent = {
        (_normalise_ticker(row["Ticker"]), str(row["ForecastWindow"]).strip()): row
        for _, row in recent_frame.iterrows()
    }
    shared_keys = sorted(set(keyed_full) & set(keyed_recent))
    if not shared_keys:
        raise RuntimeError("Range blend merge is empty; cannot build ladder evaluation report")
    for key in shared_keys:
        full_row = keyed_full[key]
        recent_row = keyed_recent[key]
        clean_rows.append(
            {
                "Ticker": key[0],
                "ForecastWindow": key[1],
                "BlendLow": _blend_range_variant_values(full_row, recent_row, "Low"),
                "BlendMid": _blend_range_variant_values(full_row, recent_row, "Mid"),
                "BlendHigh": _blend_range_variant_values(full_row, recent_row, "High"),
            }
        )
    blended = pd.DataFrame(clean_rows)
    _validate_ticker_coverage("Blended range report", blended, expected_tickers)
    return blended


def _compute_atr14(price_frame: pd.DataFrame) -> pd.Series:
    prev_close = price_frame["Close"].shift(1)
    true_range = pd.concat(
        [
            price_frame["High"] - price_frame["Low"],
            (price_frame["High"] - prev_close).abs(),
            (price_frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(14).mean()


def _build_fill_probability_base_frame(ticker: str, history_dir: Path) -> pd.DataFrame:
    normalized = _normalise_ticker(ticker)
    base_frame = build_ticker_ohlc_sample(normalized, history_dir, max_horizon=1)
    base_frame = (
        base_frame[["Date", "Ticker", "BaseClose"] + list(FEATURE_COLUMNS)]
        .drop_duplicates(subset=["Date"])
        .copy()
    )
    base_frame["Date"] = pd.to_datetime(base_frame["Date"], errors="coerce")
    base_frame = base_frame.dropna(subset=["Date"]).set_index("Date", drop=False).sort_index()

    price_frame = _load_daily_ohlcv(normalized, history_dir)
    atr14 = _compute_atr14(price_frame)
    base_frame["ATR14"] = atr14.reindex(base_frame.index)
    base_frame["TickerATR14Pct"] = ((base_frame["ATR14"] / base_frame["BaseClose"]) * 100.0).replace([np.inf, -np.inf], np.nan)
    return base_frame


def build_ticker_fill_probability_sample(
    ticker: str,
    history_dir: Path,
    *,
    horizons: Sequence[int] = DEFAULT_FILL_HORIZONS,
    depth_atr_grid: Sequence[float] = DEFAULT_FILL_DEPTH_ATR_GRID,
) -> pd.DataFrame:
    base_frame = _build_fill_probability_base_frame(ticker, history_dir)
    price_frame = _load_daily_ohlcv(_normalise_ticker(ticker), history_dir)

    frames: List[pd.DataFrame] = []
    for horizon in horizons:
        future_lows = pd.concat(
            [price_frame["Low"].shift(-day) for day in range(1, int(horizon) + 1)],
            axis=1,
        ).min(axis=1)
        for depth_atr in depth_atr_grid:
            frame = base_frame.copy()
            frame["Horizon"] = int(horizon)
            frame["EntryDepthAtr"] = float(depth_atr)
            entry_price = frame["BaseClose"] - (frame["ATR14"] * float(depth_atr))
            frame["EntryVsBasePct"] = ((entry_price / frame["BaseClose"]) - 1.0) * 100.0
            frame["TargetTouch"] = (
                future_lows.reindex(frame.index) <= entry_price
            ).astype(float)
            frames.append(frame.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.replace([np.inf, -np.inf], np.nan)
    sample = sample.dropna(subset=["Date", "BaseClose", "ATR14", "TickerATR14Pct", "EntryDepthAtr", "EntryVsBasePct", "TargetTouch"])
    return sample.reset_index(drop=True)


def build_fill_classifier_factories() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_balanced": lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gbm": lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=180,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _score_fill_eval_frame(actual: pd.Series, predicted_probability: np.ndarray) -> Dict[str, float]:
    actual_int = actual.astype(int)
    predicted = np.clip(np.asarray(predicted_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    accuracy = float(accuracy_score(actual_int, (predicted >= 0.5).astype(int)))
    brier = float(brier_score_loss(actual_int, predicted))
    auc = float("nan")
    if actual_int.nunique() >= 2:
        auc = float(roc_auc_score(actual_int, predicted))
    return {
        "EvalRows": int(actual.shape[0]),
        "PositiveRate": float(actual_int.mean()),
        "BrierLoss": brier,
        "Accuracy": accuracy,
        "AUC": auc,
        "SelectionScore": float(brier + (1.0 - accuracy)),
    }


def select_best_fill_models(
    fill_sample: pd.DataFrame,
    *,
    min_train_dates: int = DEFAULT_FILL_MIN_TRAIN_DATES,
    holdout_dates: int = DEFAULT_FILL_HOLDOUT_DATES,
) -> tuple[Dict[tuple[str, int], Pipeline], pd.DataFrame, pd.DataFrame]:
    model_factories = build_fill_classifier_factories()
    fitted_models: Dict[tuple[str, int], Pipeline] = {}
    metrics_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for (ticker, horizon), frame in fill_sample.groupby(["Ticker", "Horizon"], sort=True):
        frame = frame.sort_values("Date").reset_index(drop=True)
        unique_dates = pd.Index(sorted(pd.to_datetime(frame["Date"], errors="coerce").dropna().unique()))
        chosen_model: Pipeline | None = None
        chosen_summary: Dict[str, object] | None = None

        if unique_dates.shape[0] >= (min_train_dates + 15):
            eval_date_count = min(int(holdout_dates), max(15, unique_dates.shape[0] // 4))
            train_dates = unique_dates[:-eval_date_count]
            eval_dates = unique_dates[-eval_date_count:]
            train_df = frame[frame["Date"].isin(train_dates)].copy()
            eval_df = frame[frame["Date"].isin(eval_dates)].copy()
            if (
                train_df["Date"].nunique() >= min_train_dates
                and eval_df["Date"].nunique() >= 10
                and train_df["TargetTouch"].nunique() >= 2
                and eval_df["TargetTouch"].nunique() >= 2
            ):
                for model_name, model_factory in model_factories.items():
                    model = model_factory()
                    model.fit(train_df[FILL_FEATURE_COLUMNS], train_df["TargetTouch"].astype(int))
                    predicted = model.predict_proba(eval_df[FILL_FEATURE_COLUMNS])[:, 1]
                    summary = _score_fill_eval_frame(eval_df["TargetTouch"], predicted)
                    row = {
                        "Ticker": ticker,
                        "Horizon": int(horizon),
                        "Model": model_name,
                        **summary,
                    }
                    metrics_rows.append(row)
                    if chosen_summary is None or float(row["SelectionScore"]) < float(chosen_summary["SelectionScore"]):
                        chosen_model = model_factory()
                        chosen_model.fit(frame[FILL_FEATURE_COLUMNS], frame["TargetTouch"].astype(int))
                        chosen_summary = row

        if chosen_summary is None:
            chosen_summary = {
                "Ticker": ticker,
                "Horizon": int(horizon),
                "Model": "heuristic_fallback",
                "EvalRows": 0,
                "PositiveRate": float(frame["TargetTouch"].mean()) if not frame.empty else float("nan"),
                "BrierLoss": float("nan"),
                "Accuracy": float("nan"),
                "AUC": float("nan"),
                "SelectionScore": float("nan"),
            }
        else:
            fitted_models[(str(ticker), int(horizon))] = chosen_model  # type: ignore[assignment]
        selected_rows.append(chosen_summary)

    metrics_df = pd.DataFrame(metrics_rows)
    selected_df = pd.DataFrame(selected_rows)
    return fitted_models, metrics_df, selected_df


def _snap_buy_price(price: float, tick_size: float, floor_valid: float, ceil_valid: float) -> float:
    raw = Decimal(str(price))
    tick = Decimal(str(tick_size))
    floor = Decimal(str(floor_valid))
    ceil = Decimal(str(ceil_valid))
    if tick <= 0:
        raise ValueError("tick_size must be > 0")
    snapped = (raw / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    if snapped < floor:
        snapped = floor
    if snapped > ceil:
        snapped = ceil
    precision = max(0, -tick.as_tuple().exponent)
    return float(snapped.quantize(Decimal("1").scaleb(-precision)))


def _coerce_price_or_nan(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    try:
        price = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return price if price > 1.0 else float("nan")


def _resolve_price_band(universe_row: pd.Series) -> tuple[float, float]:
    floor_price = _coerce_price_or_nan(universe_row.get("FloorValid"))
    ceil_price = _coerce_price_or_nan(universe_row.get("CeilValid"))
    if not math.isfinite(floor_price):
        floor_price = float(universe_row["Floor"])
    if not math.isfinite(ceil_price):
        ceil_price = float(universe_row["Ceil"])
    if floor_price <= 0 or ceil_price <= 0 or floor_price > ceil_price:
        raise RuntimeError("Invalid price band in universe row")
    return floor_price, ceil_price


def _clamp_score(value: float) -> float:
    return float(min(99.0, max(1.0, value)))


def _entry_fill_multiplier(fill_score_composite: float) -> float:
    """Convert fill-score style 0-100 input into a direct EV weight.

    Deep entries already improve upside mechanically because all projected
    returns are recomputed from the lower entry price. Using fill as a direct
    weight keeps low-probability bids from dominating the ranking purely due to
    deeper price, which is especially important for live ladders where missed
    fills carry real opportunity cost.
    """

    if not math.isfinite(fill_score_composite):
        return 0.0
    return float(min(max(fill_score_composite, 0.0), 100.0) / 100.0)


def _fill_score_from_reference(entry_price: float, reference_price: float, scale: float) -> float:
    if not math.isfinite(reference_price):
        return float("nan")
    safe_scale = max(scale, 1e-6)
    logistic = 1.0 / (1.0 + math.exp(-(entry_price - reference_price) / safe_scale))
    return _clamp_score(logistic * 100.0)


def _predict_fill_score(
    model_lookup: Dict[tuple[str, int], Pipeline],
    *,
    ticker: str,
    horizon: int,
    live_feature_row: pd.Series | None,
    entry_price: float,
    base_price: float,
    atr_abs: float,
    heuristic_reference: float,
    heuristic_scale: float,
) -> float:
    model = model_lookup.get((str(ticker), int(horizon)))
    if model is None or live_feature_row is None or not math.isfinite(atr_abs) or atr_abs <= 1e-9:
        return _fill_score_from_reference(entry_price, heuristic_reference, heuristic_scale)

    feature_payload = {column: float(live_feature_row[column]) for column in FEATURE_COLUMNS}
    feature_payload["TickerATR14Pct"] = float(live_feature_row["TickerATR14Pct"])
    feature_payload["EntryDepthAtr"] = max((float(base_price) - float(entry_price)) / float(atr_abs), 0.0)
    feature_payload["EntryVsBasePct"] = ((float(entry_price) / float(base_price)) - 1.0) * 100.0
    feature_frame = pd.DataFrame([feature_payload], columns=FILL_FEATURE_COLUMNS)
    probability = float(model.predict_proba(feature_frame)[:, 1][0])
    return _clamp_score(probability * 100.0)


def _reward_risk_ratio(upside_pct: float, downside_pct: float) -> float:
    downside = abs(float(downside_pct))
    if downside <= 1e-9:
        return float("nan")
    return float(upside_pct) / downside


def _classify_entry_anchor(anchor_name: object) -> str:
    anchors = {
        str(part).strip()
        for part in str(anchor_name or "").split("|")
        if str(part).strip()
    }
    has_historical = any(anchor in HISTORICAL_ENTRY_ANCHORS for anchor in anchors)
    has_tactical = any(anchor in TACTICAL_ENTRY_ANCHORS for anchor in anchors)
    if has_historical and has_tactical:
        return "mixed"
    if has_historical:
        return "historical"
    if has_tactical:
        return "tactical"
    return "unknown"


def _best_timing_metrics_for_entry(
    timing_frame: pd.DataFrame,
    entry_price: float,
) -> Dict[str, object]:
    best_row: Dict[str, object] | None = None
    best_key: tuple[float, float] | None = None
    for row in timing_frame.to_dict(orient="records"):
        peak_ret_pct = ((float(row["PredPeakPrice"]) / entry_price) - 1.0) * 100.0
        drawdown_pct = ((float(row["PredDrawdownPrice"]) / entry_price) - 1.0) * 100.0
        close_ret_pct = ((float(row["PredClosePrice"]) / entry_price) - 1.0) * 100.0
        net_edge_pct = peak_ret_pct - abs(min(drawdown_pct, 0.0)) - ROUND_TRIP_FEE_PCT
        capital_efficiency = net_edge_pct / max(float(row["PredPeakDay"]), 1.0)
        candidate = {
            "BestTimingWindow": row["ForecastWindow"],
            "BestTimingNetEdgePct": float(net_edge_pct),
            "BestTimingCloseRetPct": float(close_ret_pct - ROUND_TRIP_FEE_PCT),
            "BestTimingRewardRisk": _reward_risk_ratio(peak_ret_pct, drawdown_pct),
            "BestTimingCapitalEfficiencyPctPerDay": float(capital_efficiency),
        }
        key = (
            float(candidate["BestTimingCapitalEfficiencyPctPerDay"]),
            float(candidate["BestTimingNetEdgePct"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_row = candidate
    if best_row is None:
        raise RuntimeError("No timing rows available to score entry ladder")
    return best_row


def _build_candidate_price_map(
    universe_row: pd.Series,
    *,
    forecast_low_t1: float,
    range_low_t5: float,
    range_low_t10: float,
    cycle_drawdown_price: float,
) -> List[tuple[float, str]]:
    tick_size = float(universe_row["TickSize"])
    floor_valid, ceil_valid = _resolve_price_band(universe_row)
    base = float(universe_row["Last"])
    atr = float(universe_row["ATR14"])
    raw_candidates = [
        ("valid_bid1", universe_row["ValidBid1"]),
        ("grid_below_t1", universe_row["GridBelow_T1"]),
        ("grid_below_t2", universe_row["GridBelow_T2"]),
        ("grid_below_t3", universe_row["GridBelow_T3"]),
        ("forecast_low_t1", forecast_low_t1),
        ("range_low_blend_t5", range_low_t5),
        ("range_low_blend_t10", range_low_t10),
        ("cycle_drawdown", cycle_drawdown_price),
        ("atr_1x_below", base - atr),
        ("atr_1_5x_below", base - (1.5 * atr)),
    ]
    dedup: Dict[float, List[str]] = {}
    for anchor, raw_price in raw_candidates:
        if not pd.notna(raw_price):
            continue
        raw_float = float(raw_price)
        if raw_float <= 0:
            continue
        snapped = _snap_buy_price(raw_float, tick_size, floor_valid, ceil_valid)
        if snapped > base + 1e-9:
            continue
        dedup.setdefault(snapped, []).append(anchor)
    candidates = sorted(dedup.items(), key=lambda item: item[0], reverse=True)
    return [(price, "|".join(anchors)) for price, anchors in candidates]


def run_report(
    *,
    history_dir: Path,
    universe_csv: Path,
    ohlc_csv: Path,
    range_full_csv: Path,
    range_recent_csv: Path,
    cycle_csv: Path,
    single_name_csv: Path,
    output_dir: Path,
) -> Dict[str, object]:
    universe = _load_universe_frame(universe_csv)
    tickers = universe["Ticker"].astype(str).tolist()
    ohlc = _load_ohlc_frame(ohlc_csv, tickers).set_index("Ticker", drop=False)
    cycle = _load_cycle_frame(cycle_csv, tickers).set_index("Ticker", drop=False)
    single_name = _load_single_name_frame(single_name_csv, tickers)
    range_blend = _load_range_blend_frame(range_full_csv, range_recent_csv, tickers)
    fill_samples = [
        build_ticker_fill_probability_sample(
            ticker,
            history_dir,
            horizons=DEFAULT_FILL_HORIZONS,
            depth_atr_grid=DEFAULT_FILL_DEPTH_ATR_GRID,
        )
        for ticker in tickers
    ]
    fill_samples = [frame for frame in fill_samples if not frame.empty]
    fill_sample = pd.concat(fill_samples, ignore_index=True) if fill_samples else pd.DataFrame()
    fill_model_lookup, fill_metrics_df, fill_selected_df = select_best_fill_models(fill_sample)
    live_fill_feature_lookup: Dict[str, pd.Series] = {}
    for ticker in tickers:
        live_base_frame = _build_fill_probability_base_frame(ticker, history_dir)
        if live_base_frame.empty:
            continue
        live_fill_feature_lookup[ticker] = live_base_frame.reset_index(drop=True).sort_values("Date").iloc[-1]

    range_lookup = {
        (_normalise_ticker(row["Ticker"]), str(row["ForecastWindow"]).strip()): row
        for _, row in range_blend.iterrows()
    }
    timing_lookup = {
        ticker_name: frame.copy().reset_index(drop=True)
        for ticker_name, frame in single_name.groupby("Ticker", sort=False)
    }

    output_rows: List[Dict[str, object]] = []
    for _, universe_row in universe.iterrows():
        ticker = str(universe_row["Ticker"])
        ohlc_row = ohlc.loc[ticker]
        cycle_row = cycle.loc[ticker]
        timing_rows = timing_lookup.get(ticker)
        if timing_rows is None or timing_rows.empty:
            raise RuntimeError(f"Single-name timing report missing rows for {ticker}")

        blend_t5 = range_lookup.get((ticker, "T+5"))
        blend_t10 = range_lookup.get((ticker, "T+10"))
        if blend_t5 is None or blend_t10 is None:
            raise RuntimeError(f"Blended range report missing T+5/T+10 rows for {ticker}")

        atr_abs = float(universe_row["ATR14"])
        tick_size = float(universe_row["TickSize"])
        scale = max(atr_abs * 0.50, tick_size * 2.0)
        forecast_low_t1 = float(ohlc_row["ForecastLow"])
        range_low_t5 = float(blend_t5["BlendLow"])
        range_low_t10 = float(blend_t10["BlendLow"])
        cycle_drawdown_price = float(cycle_row["PredDrawdownPrice"])
        candidate_entries = _build_candidate_price_map(
            universe_row,
            forecast_low_t1=forecast_low_t1,
            range_low_t5=range_low_t5,
            range_low_t10=range_low_t10,
            cycle_drawdown_price=cycle_drawdown_price,
        )
        if not candidate_entries:
            raise RuntimeError(f"No valid ladder entries generated for {ticker}")

        base = float(universe_row["Last"])
        snapshot_date = str(universe_row["EngineRunAt"]).split("T", 1)[0]
        for price_rank, (entry_price, anchor_name) in enumerate(candidate_entries, start=1):
            best_timing = _best_timing_metrics_for_entry(timing_rows, entry_price)
            next_close_ret_pct = ((float(ohlc_row["ForecastClose"]) / entry_price) - 1.0) * 100.0 - ROUND_TRIP_FEE_PCT
            next_high_ret_pct = ((float(ohlc_row["ForecastHigh"]) / entry_price) - 1.0) * 100.0 - ROUND_TRIP_FEE_PCT
            cycle_peak_ret_pct = ((float(cycle_row["PredPeakPrice"]) / entry_price) - 1.0) * 100.0
            cycle_drawdown_pct = ((float(cycle_row["PredDrawdownPrice"]) / entry_price) - 1.0) * 100.0
            cycle_net_edge_pct = cycle_peak_ret_pct - abs(min(cycle_drawdown_pct, 0.0)) - ROUND_TRIP_FEE_PCT

            live_fill_feature_row = live_fill_feature_lookup.get(ticker)
            fill_t1 = _predict_fill_score(
                fill_model_lookup,
                ticker=ticker,
                horizon=1,
                live_feature_row=live_fill_feature_row,
                entry_price=entry_price,
                base_price=base,
                atr_abs=atr_abs,
                heuristic_reference=forecast_low_t1,
                heuristic_scale=scale,
            )
            fill_t5 = _predict_fill_score(
                fill_model_lookup,
                ticker=ticker,
                horizon=5,
                live_feature_row=live_fill_feature_row,
                entry_price=entry_price,
                base_price=base,
                atr_abs=atr_abs,
                heuristic_reference=range_low_t5,
                heuristic_scale=scale,
            )
            fill_t10 = _predict_fill_score(
                fill_model_lookup,
                ticker=ticker,
                horizon=10,
                live_feature_row=live_fill_feature_row,
                entry_price=entry_price,
                base_price=base,
                atr_abs=atr_abs,
                heuristic_reference=range_low_t10,
                heuristic_scale=scale,
            )
            fill_cycle = _fill_score_from_reference(entry_price, cycle_drawdown_price, scale)
            fill_scores = [value for value in [fill_t1, fill_t5, fill_t10, fill_cycle] if math.isfinite(value)]
            fill_composite = float(np.mean(fill_scores)) if fill_scores else float("nan")

            pre_fill_score = (
                (0.50 * float(best_timing["BestTimingNetEdgePct"]))
                + (0.30 * float(cycle_net_edge_pct))
                + (0.20 * float(next_close_ret_pct))
            )
            fill_multiplier = _entry_fill_multiplier(fill_composite)
            entry_score = pre_fill_score * fill_multiplier

            output_rows.append(
                {
                    "SnapshotDate": snapshot_date,
                    "Ticker": ticker,
                    "PriceRank": int(price_rank),
                    "EntryScoreRank": 0,
                    "LimitPrice": float(entry_price),
                    "EntryAnchor": anchor_name,
                    "EntryAnchorCategory": _classify_entry_anchor(anchor_name),
                    "Base": base,
                    "TickSize": tick_size,
                    "LotSize": int(float(universe_row["LotSize"])),
                    "EntryVsLastPct": ((entry_price / base) - 1.0) * 100.0,
                    "ForecastLowT1": forecast_low_t1,
                    "RangeLowBlendT5": range_low_t5,
                    "RangeLowBlendT10": range_low_t10,
                    "CycleDrawdownPrice": cycle_drawdown_price,
                    "EntryVsForecastLowT1Pct": ((entry_price / forecast_low_t1) - 1.0) * 100.0,
                    "EntryVsRangeLowT5Pct": ((entry_price / range_low_t5) - 1.0) * 100.0,
                    "EntryVsRangeLowT10Pct": ((entry_price / range_low_t10) - 1.0) * 100.0,
                    "EntryVsCycleDrawdownPct": ((entry_price / cycle_drawdown_price) - 1.0) * 100.0,
                    "NetRetToNextClosePct": float(next_close_ret_pct),
                    "NetRetToNextHighPct": float(next_high_ret_pct),
                    "BestTimingWindow": str(best_timing["BestTimingWindow"]),
                    "BestTimingNetEdgePct": float(best_timing["BestTimingNetEdgePct"]),
                    "BestTimingCloseRetPct": float(best_timing["BestTimingCloseRetPct"]),
                    "BestTimingRewardRisk": float(best_timing["BestTimingRewardRisk"]),
                    "CycleNetEdgePct": float(cycle_net_edge_pct),
                    "CycleRewardRisk": _reward_risk_ratio(cycle_peak_ret_pct, cycle_drawdown_pct),
                    "FillScoreT1": float(fill_t1),
                    "FillScoreT5": float(fill_t5),
                    "FillScoreT10": float(fill_t10),
                    "FillScoreCycle": float(fill_cycle),
                    "FillScoreComposite": float(fill_composite),
                    "EntryScore": float(entry_score),
                }
            )

    report_df = pd.DataFrame(output_rows)
    if report_df.empty:
        raise RuntimeError("Entry ladder evaluation report is empty")

    report_df["EntryScoreRank"] = (
        report_df.groupby("Ticker")["EntryScore"].rank(method="first", ascending=False).astype(int)
    )
    report_df = report_df.sort_values(["Ticker", "EntryScoreRank", "PriceRank", "LimitPrice"], ascending=[True, True, True, False]).reset_index(drop=True)
    _require_columns(report_df, REQUIRED_OUTPUT_COLUMNS, "Entry ladder evaluation report")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE_NAME
    fill_metrics_path = output_dir / FILL_MODEL_METRICS_FILE_NAME
    fill_selected_path = output_dir / FILL_SELECTED_MODELS_FILE_NAME
    report_df.to_csv(output_path, index=False)
    fill_metrics_df.to_csv(fill_metrics_path, index=False)
    fill_selected_df.to_csv(fill_selected_path, index=False)
    return {
        "output_path": output_path,
        "fill_metrics_path": fill_metrics_path,
        "fill_selected_path": fill_selected_path,
        "row_count": int(report_df.shape[0]),
        "ticker_count": int(report_df["Ticker"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-ticker entry-ladder evaluation report for each live ticker and valid LO buy price.",
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Directory containing per-ticker daily OHLCV caches.")
    parser.add_argument("--universe-csv", type=Path, required=True, help="Path to out/universe.csv.")
    parser.add_argument("--ohlc-csv", type=Path, required=True, help="Path to out/analysis/ml_ohlc_next_session.csv.")
    parser.add_argument("--range-full-csv", type=Path, required=True, help="Path to out/analysis/ml_range_predictions_full_2y.csv.")
    parser.add_argument("--range-recent-csv", type=Path, required=True, help="Path to out/analysis/ml_range_predictions_recent_focus.csv.")
    parser.add_argument("--cycle-csv", type=Path, required=True, help="Path to out/analysis/ml_cycle_forecast/cycle_forecast_best_horizon_by_ticker.csv.")
    parser.add_argument("--single-name-csv", type=Path, required=True, help="Path to out/analysis/ml_single_name_timing.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where ml_entry_ladder_eval.csv will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_report(
        history_dir=args.history_dir,
        universe_csv=args.universe_csv,
        ohlc_csv=args.ohlc_csv,
        range_full_csv=args.range_full_csv,
        range_recent_csv=args.range_recent_csv,
        cycle_csv=args.cycle_csv,
        single_name_csv=args.single_name_csv,
        output_dir=args.output_dir,
    )
    print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
