from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKER_COLOR_DIR = REPO_ROOT / "out" / "analysis" / "ticker_color_models_5_10_logistic"
MIN_USABLE_AUC = 0.55
MIN_USABLE_HIT_PCT = 55.0


def _normalise_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


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


def load_optional_overlay(color_dir: Path | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if color_dir is None:
        return None, None
    comparison_path = color_dir / "ticker_color_comparison.csv"
    current_path = color_dir / "ticker_color_current_forecast.csv"
    if not comparison_path.exists() or not current_path.exists():
        return None, None

    comparison_df = pd.read_csv(comparison_path)
    current_df = pd.read_csv(current_path)
    _require_columns(
        comparison_df,
        [
            "Ticker",
            "TargetName",
            "Horizon",
            "BestModel",
            "BestAUC",
            "BestAccuracy",
            "BestStrongSignalHitPct",
            "BestStrongSignalAvgSignedEdgePct",
        ],
        "Ticker color comparison",
    )
    _require_columns(
        current_df,
        ["Ticker", "TargetName", "Horizon", "Model", "ProbabilityPositive", "CurrentBias", "BaseClose"],
        "Ticker color current forecast",
    )
    comparison_df = comparison_df.copy()
    current_df = current_df.copy()
    comparison_df["Ticker"] = comparison_df["Ticker"].map(_normalise_ticker)
    current_df["Ticker"] = current_df["Ticker"].map(_normalise_ticker)
    return comparison_df, current_df


def extract_ticker_overlay(
    ticker: str,
    comparison_df: pd.DataFrame | None,
    current_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if comparison_df is None or current_df is None:
        return []
    ticker = _normalise_ticker(ticker)
    scoped_comparison = comparison_df.loc[comparison_df["Ticker"].eq(ticker)].copy()
    scoped_current = current_df.loc[current_df["Ticker"].eq(ticker)].copy()
    if scoped_comparison.empty or scoped_current.empty:
        return []

    merged = scoped_comparison.merge(
        scoped_current,
        left_on=["Ticker", "TargetName", "Horizon", "BestModel"],
        right_on=["Ticker", "TargetName", "Horizon", "Model"],
        how="left",
    )

    rows: List[Dict[str, Any]] = []
    for _, row in merged.sort_values(["TargetName", "Horizon"]).iterrows():
        auc = _safe_float(row.get("BestAUC"))
        hit_pct = _safe_float(row.get("BestStrongSignalHitPct"))
        signed_edge = _safe_float(row.get("BestStrongSignalAvgSignedEdgePct"))
        rows.append(
            {
                "Ticker": ticker,
                "TargetName": str(row.get("TargetName")),
                "Horizon": int(row.get("Horizon")),
                "BestModel": str(row.get("BestModel")),
                "BestAUC": _round_or_none(auc, 3),
                "BestAccuracy": _round_or_none(row.get("BestAccuracy"), 3),
                "BestStrongSignalHitPct": _round_or_none(hit_pct, 2),
                "BestStrongSignalAvgSignedEdgePct": _round_or_none(signed_edge, 3),
                "ProbabilityPositive": _round_or_none(row.get("ProbabilityPositive"), 3),
                "CurrentBias": row.get("CurrentBias"),
                "BaseClose": _round_or_none(row.get("BaseClose"), 2),
                "IsUsable": bool(
                    auc is not None
                    and hit_pct is not None
                    and signed_edge is not None
                    and auc >= MIN_USABLE_AUC
                    and hit_pct >= MIN_USABLE_HIT_PCT
                    and signed_edge > 0.0
                ),
            }
        )
    return rows


def summarise_ticker_overlay(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"Rows": [], "UsableRows": [], "OverlayScore": 0, "Summary": None}

    usable_rows = [dict(row) for row in rows if bool(row.get("IsUsable"))]
    overlay_score = 0
    summary_parts: List[str] = []
    for row in usable_rows:
        target_name = str(row.get("TargetName"))
        horizon = int(row.get("Horizon") or 0)
        current_bias = str(row.get("CurrentBias") or "NA")
        auc = _round_or_none(row.get("BestAUC"), 2)
        hit_pct = _round_or_none(row.get("BestStrongSignalHitPct"), 1)
        signed_edge = _round_or_none(row.get("BestStrongSignalAvgSignedEdgePct"), 2)

        if target_name == "GREEN":
            if current_bias == "YES":
                overlay_score += 4 if horizon >= 10 else 3
            elif current_bias == "NO":
                overlay_score -= 2 if horizon >= 10 else 1
        elif target_name == "RED":
            if current_bias == "YES":
                overlay_score -= 4 if horizon >= 10 else 3
            elif current_bias == "NO":
                overlay_score += 2 if horizon >= 10 else 1

        summary_parts.append(
            f"{target_name.lower()} T+{horizon} {current_bias} (AUC {auc:.2f}, hit {hit_pct:.1f}%, edge {signed_edge:.2f}%)"
        )

    return {
        "Rows": [dict(row) for row in rows],
        "UsableRows": usable_rows,
        "OverlayScore": int(overlay_score),
        "Summary": "; ".join(summary_parts[:4]) if summary_parts else None,
    }
