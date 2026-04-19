from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.evaluate_deterministic_strategies import DEFAULT_CASE_TICKERS, build_feature_pack
from scripts.data_fetching.macro_factor_cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG_PATH,
    load_macro_factor_matrix,
    load_macro_specs,
    refresh_macro_factor_cache,
)


DEFAULT_MACRO_CASE_TICKERS = ["HPG", "FPT", "SSI", "VCB", "GAS", "PLX", "GVR", "MWG", "MBB", "TCB"]


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _rolling_beta(asset_returns: pd.Series, factor_returns: pd.Series, window: int) -> pd.Series:
    variance = factor_returns.rolling(window).var()
    return asset_returns.rolling(window).cov(factor_returns) / variance


def summarise_factor_shocks(asset_returns_pct: pd.Series, asset_fwd5_pct: pd.Series, factor_returns_pct: pd.Series) -> Dict[str, float]:
    aligned = pd.concat(
        [
            asset_returns_pct.rename("AssetRet1Pct"),
            asset_fwd5_pct.rename("AssetFwd5Pct"),
            factor_returns_pct.rename("FactorRet1Pct"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {
            "UpShockCount": 0.0,
            "DownShockCount": 0.0,
            "AvgRetOnUpShockPct": float("nan"),
            "AvgRetOnDownShockPct": float("nan"),
            "AvgFwd5OnUpShockPct": float("nan"),
            "AvgFwd5OnDownShockPct": float("nan"),
        }

    positive = aligned.loc[aligned["FactorRet1Pct"] > 0, "FactorRet1Pct"]
    negative = aligned.loc[aligned["FactorRet1Pct"] < 0, "FactorRet1Pct"]
    up_threshold = positive.quantile(0.9) if len(positive) >= 10 else float("nan")
    down_threshold = negative.quantile(0.1) if len(negative) >= 10 else float("nan")
    up_mask = aligned["FactorRet1Pct"] >= up_threshold if np.isfinite(up_threshold) else pd.Series(False, index=aligned.index)
    down_mask = aligned["FactorRet1Pct"] <= down_threshold if np.isfinite(down_threshold) else pd.Series(False, index=aligned.index)

    def _avg(mask: pd.Series, column: str) -> float:
        scoped = aligned.loc[mask, column]
        return float(scoped.mean()) if not scoped.empty else float("nan")

    return {
        "UpShockCount": float(up_mask.sum()),
        "DownShockCount": float(down_mask.sum()),
        "AvgRetOnUpShockPct": _avg(up_mask, "AssetRet1Pct"),
        "AvgRetOnDownShockPct": _avg(down_mask, "AssetRet1Pct"),
        "AvgFwd5OnUpShockPct": _avg(up_mask, "AssetFwd5Pct"),
        "AvgFwd5OnDownShockPct": _avg(down_mask, "AssetFwd5Pct"),
    }


def build_case_studies(sensitivity_df: pd.DataFrame, case_tickers: Sequence[str], top_factors: int) -> pd.DataFrame:
    if sensitivity_df.empty:
        return pd.DataFrame()
    case_set = {_normalise_ticker(ticker) for ticker in case_tickers}
    scoped = sensitivity_df[sensitivity_df["Ticker"].isin(case_set)].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["AbsCorr60"] = scoped["CurrentCorr60"].abs()
    scoped = scoped.sort_values(
        ["Ticker", "AbsCorr60", "CurrentAlignment20Pct"],
        ascending=[True, False, False],
    )
    scoped["FactorRank"] = scoped.groupby("Ticker").cumcount() + 1
    scoped = scoped.groupby("Ticker", as_index=False, group_keys=False).head(top_factors).reset_index(drop=True)
    return scoped.drop(columns=["AbsCorr60"])


def run_analysis(
    history_dir: Path,
    sector_map_path: Path,
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    case_tickers: Sequence[str],
    top_factors: int,
    max_age_hours: int,
    refresh_factors: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_macro_specs(config_path)
    factor_names = list(specs.keys())
    if refresh_factors:
        refresh_macro_factor_cache(
            config_path=config_path,
            cache_dir=cache_dir,
            factor_names=factor_names,
            max_age_hours=max_age_hours,
        )
    factor_values = load_macro_factor_matrix(cache_dir, factor_names=factor_names)
    if factor_values.empty:
        raise RuntimeError(f"No cached macro factors found in {cache_dir}")
    factor_names = [factor_name for factor_name in factor_names if factor_name in factor_values.columns]
    if not factor_names:
        raise RuntimeError(f"No configured macro factors available in {cache_dir}")

    features = build_feature_pack(history_dir, sector_map_path)
    aligned_values = factor_values.reindex(features.close.index).ffill()
    factor_returns = aligned_values.pct_change().replace([np.inf, -np.inf], np.nan)
    factor_ret5 = aligned_values.pct_change(5) * 100.0
    factor_ret20 = aligned_values.pct_change(20) * 100.0
    factor_vol20 = factor_returns.rolling(20).std() * 100.0

    latest = features.close.index.max()
    current_regime_rows: List[Dict[str, object]] = []
    for factor_name in factor_names:
        latest_value = aligned_values.loc[latest, factor_name] if latest in aligned_values.index else float("nan")
        current_regime_rows.append(
            {
                "Date": latest.date().isoformat(),
                "Factor": factor_name,
                "Label": specs[factor_name].label,
                "Source": specs[factor_name].source,
                "LatestValue": float(latest_value) if pd.notna(latest_value) else float("nan"),
                "Ret5Pct": float(factor_ret5.loc[latest, factor_name]) if latest in factor_ret5.index else float("nan"),
                "Ret20Pct": float(factor_ret20.loc[latest, factor_name]) if latest in factor_ret20.index else float("nan"),
                "Vol20Pct": float(factor_vol20.loc[latest, factor_name]) if latest in factor_vol20.index else float("nan"),
            }
        )
    current_regime = pd.DataFrame(current_regime_rows).sort_values("Factor").reset_index(drop=True)
    factor_ret20_map = current_regime.set_index("Factor")["Ret20Pct"].to_dict()

    sensitivity_rows: List[Dict[str, object]] = []
    asset_returns_pct = features.ret1 * 100.0
    asset_fwd5_pct = features.fwd5 * 100.0
    for ticker in features.close.columns:
        for factor_name in factor_names:
            factor_ret = factor_returns[factor_name]
            corr20 = features.ret1[ticker].rolling(20).corr(factor_ret)
            corr60 = features.ret1[ticker].rolling(60).corr(factor_ret)
            beta20 = _rolling_beta(features.ret1[ticker], factor_ret, 20)
            beta60 = _rolling_beta(features.ret1[ticker], factor_ret, 60)
            shock = summarise_factor_shocks(asset_returns_pct[ticker], asset_fwd5_pct[ticker], factor_ret * 100.0)
            current_corr60 = float(corr60.loc[latest]) if latest in corr60.index else float("nan")
            current_alignment = current_corr60 * float(factor_ret20_map.get(factor_name, float("nan")))
            sensitivity_rows.append(
                {
                    "Date": latest.date().isoformat(),
                    "Ticker": ticker,
                    "Sector": features.sectors.get(ticker, "Unknown"),
                    "Factor": factor_name,
                    "FactorLabel": specs[factor_name].label,
                    "CurrentCorr20": float(corr20.loc[latest]) if latest in corr20.index else float("nan"),
                    "CurrentCorr60": current_corr60,
                    "CurrentBeta20": float(beta20.loc[latest]) if latest in beta20.index else float("nan"),
                    "CurrentBeta60": float(beta60.loc[latest]) if latest in beta60.index else float("nan"),
                    "FactorRet20Pct": float(factor_ret20_map.get(factor_name, float("nan"))),
                    "CurrentAlignment20Pct": current_alignment,
                    **shock,
                }
            )

    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
        ["Ticker", "CurrentAlignment20Pct"],
        ascending=[True, False],
    ).reset_index(drop=True)
    case_studies = build_case_studies(sensitivity_df, case_tickers=case_tickers, top_factors=top_factors)
    current_ticker_summary = (
        sensitivity_df.assign(AbsCorr60=lambda frame: frame["CurrentCorr60"].abs())
        .sort_values(["Ticker", "AbsCorr60", "CurrentAlignment20Pct"], ascending=[True, False, False])
        .groupby("Ticker", as_index=False, group_keys=False)
        .head(1)
        .drop(columns=["AbsCorr60"])
        .reset_index(drop=True)
    )

    current_regime.to_csv(output_dir / "macro_factor_current_regime.csv", index=False)
    sensitivity_df.to_csv(output_dir / "macro_factor_sensitivity.csv", index=False)
    case_studies.to_csv(output_dir / "macro_factor_case_studies.csv", index=False)
    current_ticker_summary.to_csv(output_dir / "macro_factor_ticker_summary.csv", index=False)

    summary_payload = {
        "factor_count": len(factor_names),
        "ticker_count": int(features.close.shape[1]),
        "case_tickers": list(case_tickers),
        "top_factors_per_ticker": int(top_factors),
        "latest_date": latest.date().isoformat(),
    }
    (output_dir / "macro_factor_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MacroFactorSensitivity")
    print(current_regime[["Factor", "Ret5Pct", "Ret20Pct", "Vol20Pct"]].to_string(index=False))
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ticker sensitivity to cached macro factors.")
    parser.add_argument("--history-dir", default="out/data", help="Directory containing *_daily.csv cache files.")
    parser.add_argument("--sector-map", default="data/industry_map.csv", help="Ticker -> sector CSV.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Macro factor YAML config.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Macro factor cache directory.")
    parser.add_argument("--output-dir", default="out/analysis", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--case-tickers", nargs="*", default=DEFAULT_MACRO_CASE_TICKERS, help="Tickers to highlight in case studies.")
    parser.add_argument("--top-factors", type=int, default=3, help="How many factors to keep per case ticker.")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Refresh factor caches older than this many hours.")
    parser.add_argument("--no-refresh-factors", action="store_true", help="Use cached factor CSVs only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_analysis(
        history_dir=Path(args.history_dir),
        sector_map_path=Path(args.sector_map),
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        case_tickers=[_normalise_ticker(ticker) for ticker in args.case_tickers or DEFAULT_CASE_TICKERS],
        top_factors=args.top_factors,
        max_age_hours=args.max_age_hours,
        refresh_factors=not args.no_refresh_factors,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
