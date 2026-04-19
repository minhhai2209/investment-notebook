"""Standalone data engine that collects market data and pre-computes metrics."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from scripts.data_fetching.collect_intraday import ensure_intraday_latest_df
from scripts.data_fetching.cafef_flows import build_flow_feature_frame
from scripts.data_fetching.vietstock_overview_api import build_fundamental_frame
from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
from scripts.indicators import atr_wilder, ma, rsi_wilder, ema

LOGGER = logging.getLogger(__name__)
VN_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
FALLBACK_VN30_MEMBERS: set[str] = {
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
}


def fetch_vn30_members(timeout: int = 15) -> set[str]:
    """Scrape danh sách thành phần VN30 từ bảng giá Vietstock.

    Trang này render sẵn các hàng ``<tr data-symbol=...>`` nên không cần JS. Nếu
    không thể tải/parse dữ liệu thì fallback về danh sách static để engine vẫn chạy.
    """

    url = "https://banggia.vietstock.vn/bang-gia/vn30"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure path
        LOGGER.warning(
            "Không lấy được danh sách VN30 từ %s (%s); fallback sang danh sách static.",
            url,
            exc,
        )
        return set(FALLBACK_VN30_MEMBERS)

    tickers = set(re.findall(r'data-symbol="([A-Z0-9]{3,})"', r.text or ""))
    if not tickers:
        LOGGER.warning(
            "Trang VN30 không có data-symbol (%s); fallback sang danh sách static.",
            url,
        )
        return set(FALLBACK_VN30_MEMBERS)
    return {t.upper() for t in tickers}


class ConfigurationError(RuntimeError):
    """Raised when the engine configuration file is invalid."""


@dataclass
class PresetConfig:
    name: str
    buy_tiers: List[float]
    sell_tiers: List[float]
    description: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, object]) -> "PresetConfig":
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Preset '{name}' must be a mapping, got {type(raw).__name__}")
        buy = raw.get("buy_tiers", [])
        sell = raw.get("sell_tiers", [])
        if not isinstance(buy, list) or not all(isinstance(x, (int, float)) for x in buy):
            raise ConfigurationError(f"Preset '{name}' requires buy_tiers as a list of numbers")
        if not isinstance(sell, list) or not all(isinstance(x, (int, float)) for x in sell):
            raise ConfigurationError(f"Preset '{name}' requires sell_tiers as a list of numbers")
        desc = raw.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ConfigurationError(f"Preset '{name}' description must be a string")
        return cls(name=name, buy_tiers=[float(x) for x in buy], sell_tiers=[float(x) for x in sell], description=desc)


@dataclass
class ShortlistFilterConfig:
    """Configuration for conservative preset shortlisting.

    The intention is to remove only the *very weak* tickers from preset outputs,
    while keeping everything else for consideration. Users can also force-keep or
    force-exclude specific tickers.
    """

    enabled: bool = False
    # Technical weakness thresholds (all must be met to exclude unless logic says otherwise)
    rsi14_max: Optional[float] = 25.0  # RSI_14 <= rsi14_max
    max_pct_to_lo_252: Optional[float] = 2.0  # PctToLo_252 <= X (near 52w low)
    return20_max: Optional[float] = -15.0  # Return_20 <= X
    return60_max: Optional[float] = -25.0  # Return_60 <= X
    require_below_sma50_and_200: bool = True  # LastPrice < SMA_50 and < SMA_200
    min_adv_20: Optional[float] = None  # ADV_20 <= X means illiquid (optional)
    # Compose: if True, require all active conditions to be true to drop; if False, any.
    drop_logic_all: bool = True
    # Manual overrides
    keep: List[str] | None = None
    exclude: List[str] | None = None

    def normalized_keep(self) -> List[str]:
        return sorted({t.strip().upper() for t in (self.keep or []) if isinstance(t, str) and t.strip()})

    def normalized_exclude(self) -> List[str]:
        return sorted({t.strip().upper() for t in (self.exclude or []) if isinstance(t, str) and t.strip()})

@dataclass
class EngineConfig:
    universe_csv: Path
    include_indices: bool
    industry_ticker_filter: Optional[List[str]]
    industry_ticker_filter_source: Optional[str]
    moving_averages: List[int]
    rsi_periods: List[int]
    atr_periods: List[int]
    ema_periods: List[int]
    returns_periods: List[int]
    bollinger_windows: List[int]
    bollinger_k: float
    bollinger_include_bands: bool
    range_lookback_days: int
    adv_periods: List[int]
    macd_fast: int
    macd_slow: int
    macd_signal: int
    presets: Dict[str, PresetConfig]
    portfolio_dir: Path
    output_base_dir: Path
    market_snapshot_path: Path
    presets_dir: Path
    portfolios_dir: Path
    diagnostics_dir: Path
    market_cache_dir: Path
    history_min_days: int
    intraday_window_minutes: int
    cafef_flow_enabled: bool
    cafef_flow_cache_dir: Path
    cafef_flow_max_age_hours: int
    vietstock_overview_enabled: bool
    vietstock_overview_cache_dir: Path
    vietstock_overview_max_age_hours: int
    aggressiveness: str
    max_order_pct_adv: float
    slice_adv_ratio: float
    min_lot: int
    max_qty_per_order: int
    shortlist_filter: Optional[ShortlistFilterConfig]
    # Optional CSV to override exchange reference prices used for bands
    reference_overrides_csv: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "EngineConfig":
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        config_dir = path.parent.resolve()
        repo_root = _find_repo_root(config_dir)
        load_dotenv(dotenv_path=repo_root / ".env", override=False)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigurationError("Engine config must be a mapping")
        uni = data.get("universe", {})
        if not isinstance(uni, dict):
            raise ConfigurationError("universe section must be a mapping")
        csv_path = uni.get("csv")
        if not isinstance(csv_path, str):
            raise ConfigurationError("universe.csv must be a string path")
        universe_csv = _resolve_path(csv_path, config_dir, repo_root)
        include_indices = bool(uni.get("include_indices", False))
        core_universe_tickers = _parse_optional_ticker_list(
            uni.get("core_tickers"),
            "universe.core_tickers",
        )
        preferred_universe_tickers = _parse_optional_ticker_list(
            uni.get("preferred_tickers"),
            "universe.preferred_tickers",
        )
        configured_universe_tickers = sorted(set(core_universe_tickers or [])) or _parse_optional_ticker_list(
            uni.get("tickers"),
            "universe.tickers",
        )
        technical = data.get("technical_indicators", {}) or {}
        if not isinstance(technical, dict):
            raise ConfigurationError("technical_indicators must be a mapping")
        moving_averages = [int(x) for x in technical.get("moving_averages", [])]
        rsi_periods = [int(x) for x in technical.get("rsi_periods", [])]
        atr_periods = [int(x) for x in technical.get("atr_periods", [])]
        ema_periods = [int(x) for x in technical.get("ema_periods", [])]
        returns_periods = [int(x) for x in technical.get("returns_periods", [])]
        bb_cfg = technical.get("bollinger", {}) or {}
        if not isinstance(bb_cfg, dict):
            raise ConfigurationError("technical_indicators.bollinger must be a mapping if provided")
        bollinger_windows = [int(x) for x in bb_cfg.get("windows", [])]
        bollinger_k = float(bb_cfg.get("k", 2.0))
        bollinger_include_bands = bool(bb_cfg.get("include_bands", False))
        range_lookback_days = int(technical.get("range_lookback_days", 252))
        adv_periods = [int(x) for x in technical.get("adv_periods", [])]
        macd_cfg = technical.get("macd", {}) or {}
        if not isinstance(macd_cfg, dict):
            raise ConfigurationError("technical_indicators.macd must be a mapping")
        macd_fast = int(macd_cfg.get("fast", 12))
        macd_slow = int(macd_cfg.get("slow", 26))
        macd_signal = int(macd_cfg.get("signal", 9))

        raw_presets = data.get("presets", {}) or {}
        if not isinstance(raw_presets, dict):
            raise ConfigurationError("presets section must be a mapping if provided")
        presets = {name: PresetConfig.from_dict(name, cfg) for name, cfg in raw_presets.items()}

        portfolio_cfg = data.get("portfolio", {}) or {}
        if not isinstance(portfolio_cfg, dict):
            raise ConfigurationError("portfolio section must be a mapping")
        portfolio_dir = _resolve_path(
            portfolio_cfg.get("directory", "data/portfolios"), config_dir, repo_root
        )
        raw_industry_filter = os.environ.get("INDUSTRY_TICKER_FILTER", "").strip()
        industry_ticker_filter: Optional[List[str]] = None
        industry_ticker_filter_source: Optional[str] = None
        if raw_industry_filter:
            parsed_filter = _parse_ticker_list(raw_industry_filter)
            if not parsed_filter:
                raise ConfigurationError(
                    "INDUSTRY_TICKER_FILTER is set but does not contain any valid tickers"
                )
            industry_ticker_filter = parsed_filter
            industry_ticker_filter_source = "INDUSTRY_TICKER_FILTER"
        else:
            if configured_universe_tickers:
                industry_ticker_filter = configured_universe_tickers
                if core_universe_tickers:
                    industry_ticker_filter_source = "config universe.core_tickers"
                else:
                    industry_ticker_filter_source = "config universe.tickers"

        output_cfg = data.get("output", {}) or {}
        if not isinstance(output_cfg, dict):
            raise ConfigurationError("output section must be a mapping")
        output_base_dir = _resolve_path(output_cfg.get("base_dir", "out"), config_dir, repo_root)
        market_snapshot_rel = output_cfg.get("market_snapshot", "technical_snapshot.csv")
        presets_rel = output_cfg.get("presets_dir", ".")
        portfolios_rel = output_cfg.get("portfolios_dir", ".")
        diagnostics_rel = output_cfg.get("diagnostics_dir", ".")
        market_snapshot_path = (output_base_dir / market_snapshot_rel).resolve()
        presets_dir = (output_base_dir / presets_rel).resolve()
        portfolios_dir = (output_base_dir / portfolios_rel).resolve()
        diagnostics_dir = (output_base_dir / diagnostics_rel).resolve()

        data_cfg = data.get("data", {}) or {}
        if not isinstance(data_cfg, dict):
            raise ConfigurationError("data section must be a mapping")
        market_cache_dir = _resolve_path(data_cfg.get("history_cache", "out/data"), config_dir, repo_root)
        history_min_days = int(data_cfg.get("history_min_days", 400))
        intraday_window_minutes = int(data_cfg.get("intraday_window_minutes", 12 * 60))
        cafef_flow_enabled = bool(data_cfg.get("cafef_flow_enabled", True))
        cafef_flow_cache_dir = _resolve_path(
            data_cfg.get("cafef_flow_cache", "out/cafef_flows"), config_dir, repo_root
        )
        cafef_flow_max_age_hours = int(data_cfg.get("cafef_flow_max_age_hours", 4))
        vietstock_overview_enabled = bool(data_cfg.get("vietstock_overview_enabled", True))
        vietstock_overview_cache_dir = _resolve_path(
            data_cfg.get("vietstock_overview_cache", "out/vietstock_overview"),
            config_dir,
            repo_root,
        )
        vietstock_overview_max_age_hours = int(data_cfg.get("vietstock_overview_max_age_hours", 24))
        ref_override_raw = data_cfg.get("reference_overrides")
        reference_overrides_csv: Optional[Path] = None
        if ref_override_raw:
            reference_overrides_csv = _resolve_path(str(ref_override_raw), config_dir, repo_root)

        execution_cfg = data.get("execution", {}) or {}
        if not isinstance(execution_cfg, dict):
            raise ConfigurationError("execution section must be a mapping")
        aggressiveness = str(execution_cfg.get("aggressiveness", "med"))
        max_order_pct_adv = float(execution_cfg.get("max_order_pct_adv", 0.1))
        slice_adv_ratio = float(execution_cfg.get("slice_adv_ratio", 0.25))
        min_lot = int(execution_cfg.get("min_lot", 100))
        max_qty_per_order = int(execution_cfg.get("max_qty_per_order", 500_000))
        if min_lot <= 0:
            raise ConfigurationError("execution.min_lot must be positive")
        if max_qty_per_order <= 0 or max_qty_per_order > 500_000:
            raise ConfigurationError("execution.max_qty_per_order must be in (0, 500000]")
        if not 0 < max_order_pct_adv <= 1:
            raise ConfigurationError("execution.max_order_pct_adv must be in (0, 1]")
        if not 0 < slice_adv_ratio <= 1:
            raise ConfigurationError("execution.slice_adv_ratio must be in (0, 1]")

        # Optional shortlist filter
        filters_cfg = data.get("filters", {}) or {}
        if not isinstance(filters_cfg, dict):
            raise ConfigurationError("filters section must be a mapping if provided")
        shortlist_cfg_raw = filters_cfg.get("shortlist", {}) or {}
        if shortlist_cfg_raw is not None and not isinstance(shortlist_cfg_raw, dict):
            raise ConfigurationError("filters.shortlist must be a mapping if provided")
        shortlist_filter: Optional[ShortlistFilterConfig]
        if shortlist_cfg_raw:
            # Map YAML keys to dataclass fields with type coercion
            shortlist_filter = ShortlistFilterConfig(
                enabled=bool(shortlist_cfg_raw.get("enabled", False)),
                rsi14_max=(float(shortlist_cfg_raw["rsi14_max"]) if "rsi14_max" in shortlist_cfg_raw and shortlist_cfg_raw["rsi14_max"] is not None else ShortlistFilterConfig.rsi14_max),
                max_pct_to_lo_252=(float(shortlist_cfg_raw["max_pct_to_lo_252"]) if "max_pct_to_lo_252" in shortlist_cfg_raw and shortlist_cfg_raw["max_pct_to_lo_252"] is not None else ShortlistFilterConfig.max_pct_to_lo_252),
                return20_max=(float(shortlist_cfg_raw["return20_max"]) if "return20_max" in shortlist_cfg_raw and shortlist_cfg_raw["return20_max"] is not None else ShortlistFilterConfig.return20_max),
                return60_max=(float(shortlist_cfg_raw["return60_max"]) if "return60_max" in shortlist_cfg_raw and shortlist_cfg_raw["return60_max"] is not None else ShortlistFilterConfig.return60_max),
                require_below_sma50_and_200=bool(shortlist_cfg_raw.get("require_below_sma50_and_200", True)),
                min_adv_20=(float(shortlist_cfg_raw["min_adv_20"]) if "min_adv_20" in shortlist_cfg_raw and shortlist_cfg_raw["min_adv_20"] is not None else None),
                drop_logic_all=bool(shortlist_cfg_raw.get("drop_logic_all", True)),
                keep=[str(x).upper() for x in shortlist_cfg_raw.get("keep", [])],
                exclude=[str(x).upper() for x in shortlist_cfg_raw.get("exclude", [])],
            )
        else:
            shortlist_filter = None

        return cls(
            universe_csv=universe_csv,
            include_indices=include_indices,
            moving_averages=moving_averages,
            rsi_periods=rsi_periods,
            atr_periods=atr_periods,
            ema_periods=ema_periods,
            returns_periods=returns_periods,
            bollinger_windows=bollinger_windows,
            bollinger_k=bollinger_k,
            bollinger_include_bands=bollinger_include_bands,
            range_lookback_days=range_lookback_days,
            adv_periods=adv_periods,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            presets=presets,
            portfolio_dir=portfolio_dir,
            output_base_dir=output_base_dir,
            market_snapshot_path=market_snapshot_path,
            presets_dir=presets_dir,
            portfolios_dir=portfolios_dir,
            diagnostics_dir=diagnostics_dir,
            market_cache_dir=market_cache_dir,
            history_min_days=history_min_days,
            intraday_window_minutes=intraday_window_minutes,
            cafef_flow_enabled=cafef_flow_enabled,
            cafef_flow_cache_dir=cafef_flow_cache_dir,
            cafef_flow_max_age_hours=cafef_flow_max_age_hours,
            vietstock_overview_enabled=vietstock_overview_enabled,
            vietstock_overview_cache_dir=vietstock_overview_cache_dir,
            vietstock_overview_max_age_hours=vietstock_overview_max_age_hours,
            aggressiveness=aggressiveness,
            max_order_pct_adv=max_order_pct_adv,
            slice_adv_ratio=slice_adv_ratio,
            min_lot=min_lot,
            max_qty_per_order=max_qty_per_order,
            shortlist_filter=shortlist_filter,
            industry_ticker_filter=industry_ticker_filter,
            industry_ticker_filter_source=industry_ticker_filter_source,
            reference_overrides_csv=reference_overrides_csv,
        )


def _find_repo_root(start: Path) -> Path:
    current = start
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _resolve_path(candidate: str, config_dir: Path, repo_root: Path) -> Path:
    if not isinstance(candidate, str):
        raise ConfigurationError(f"Expected string path, got {type(candidate).__name__}")
    path = Path(candidate)
    if path.is_absolute():
        return path.resolve()

    config_candidate = (config_dir / path).resolve()
    try:
        config_candidate.relative_to(repo_root)
    except ValueError:
        raise ConfigurationError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    if config_candidate.exists():
        return config_candidate

    root_candidate = (repo_root / path).resolve()
    try:
        root_candidate.relative_to(repo_root)
    except ValueError:
        raise ConfigurationError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    return root_candidate


def _parse_ticker_list(raw: str) -> List[str]:
    tokens = [t.strip().upper() for t in re.split(r"[,\s]+", raw) if t and t.strip()]
    return sorted({t for t in tokens if t})


def _parse_optional_ticker_list(value: object, field_name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        tickers = _parse_ticker_list(value)
    elif isinstance(value, list):
        tickers = _parse_ticker_list(",".join(str(item) for item in value if item is not None))
    else:
        raise ConfigurationError(f"{field_name} must be a string or list of tickers")
    return tickers or None


def _tick_size(price: float) -> float:
    if price is None or pd.isna(price):
        return float("nan")
    value = float(price)
    if value < 10.0:
        return 0.01
    if value < 50.0:
        return 0.05
    return 0.10


def _as_decimal(value: float | int | str) -> Optional[Decimal]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _quantize_to_tick(value: float, rounding) -> float:
    dec_value = _as_decimal(value)
    if dec_value is None:
        return float("nan")
    tick = _as_decimal(_tick_size(float(dec_value)))
    if tick is None or tick == 0:
        return float("nan")
    steps = (dec_value / tick).to_integral_value(rounding=rounding)
    result = steps * tick
    return float(result)


def round_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_HALF_UP)


def floor_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_FLOOR)


def ceil_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_CEILING)


def clamp_price(value: float, floor_value: float, ceil_value: float) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if floor_value is not None and not pd.isna(floor_value):
        value = max(value, float(floor_value))
    if ceil_value is not None and not pd.isna(ceil_value):
        value = min(value, float(ceil_value))
    return value


class MarketDataService(Protocol):
    """Interface for loading market data."""

    def load_history(self, tickers: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - protocol definition
        ...

    def load_intraday(self, tickers: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - protocol definition
        ...


class VndirectMarketDataService:
    """Production data source backed by VNDIRECT APIs."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def load_history(self, tickers: Sequence[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "t"])
        return ensure_and_load_history_df(
            list(tickers),
            outdir=str(self._config.market_cache_dir),
            min_days=self._config.history_min_days,
            resolution="D",
        )

    def load_intraday(self, tickers: Sequence[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["Ticker", "Ts", "Price", "RSI14", "TimeVN"])
        return ensure_intraday_latest_df(list(tickers), window_minutes=self._config.intraday_window_minutes)


class TechnicalSnapshotBuilder:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def build(self, history_df: pd.DataFrame, intraday_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            raise RuntimeError("No historical prices available to build technical snapshot")
        if "Ticker" not in history_df.columns:
            raise ValueError("History dataframe missing 'Ticker' column")
        if "Close" not in history_df.columns:
            raise ValueError("History dataframe missing 'Close' column")
        date_field = "t" if "t" in history_df.columns else "Date"
        history_df = history_df.copy()
        history_df["Ticker"] = history_df["Ticker"].astype(str).str.upper()
        intraday_lookup = {
            str(row.Ticker).upper(): row for row in intraday_df.itertuples(index=False)
        }
        industry_lookup = {}
        if "Ticker" in industry_df.columns:
            sector_col = "Sector" if "Sector" in industry_df.columns else None
            for row in industry_df.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                sector = getattr(row, sector_col) if sector_col else None
                industry_lookup[ticker] = sector
        # Optional: load reference overrides (Ticker,Ref)
        ref_override: Dict[str, float] = {}
        if self._config.reference_overrides_csv:
            ov_path = self._config.reference_overrides_csv
            if not ov_path.exists():
                raise RuntimeError(f"reference_overrides path not found: {ov_path}")
            try:
                df_ov = pd.read_csv(ov_path)
            except Exception as exc:  # pragma: no cover - defensive read validation
                raise RuntimeError(f"Failed to read reference_overrides CSV: {ov_path}: {exc}")
            for col in ("Ticker", "Ref"):
                if col not in df_ov.columns:
                    raise RuntimeError("reference_overrides CSV must have columns: Ticker,Ref")
            for r in df_ov.itertuples(index=False):
                t = str(getattr(r, "Ticker")).strip().upper()
                try:
                    ref_val = float(getattr(r, "Ref"))
                except Exception:
                    continue
                if t and not pd.isna(ref_val):
                    ref_override[t] = ref_val

        index_returns_lookup: Dict[str, float] = {}
        benchmark = "VNINDEX"
        index_mask = history_df["Ticker"] == benchmark
        if index_mask.any():
            index_series = history_df[index_mask].sort_values(date_field)
            index_close = pd.to_numeric(index_series.get("Close"), errors="coerce")
            index_returns = index_close.pct_change()
            index_dates = index_series.get(date_field)
            if index_dates is not None:
                index_keys = index_dates.astype(str)
                index_returns_lookup = dict(zip(index_keys, index_returns))

        rows: List[Dict[str, object]] = []
        for ticker, ticker_df in history_df.groupby("Ticker"):
            ticker = str(ticker).upper()
            series = ticker_df.sort_values("t" if "t" in ticker_df.columns else "Date").reset_index(drop=True)
            if series.empty:
                continue
            last = series.iloc[-1]
            last_close = float(last.get("Close", float("nan")))
            # Apply reference override if provided (wins over any further logic)
            if ticker in ref_override:
                last_close = float(ref_override[ticker])
            last_volume = float(last.get("Volume", 0.0)) if not pd.isna(last.get("Volume")) else 0.0
            prev_close = float("nan")
            if len(series) >= 2:
                prev = series.iloc[-2]
                prev_close = float(prev.get("Close", float("nan")))
            # If the latest daily bar is for today while the VN session may still be open,
            # treat it as an intraday partial bar: exclude it from indicator windows and
            # use the previous day close as reference ('giá tham chiếu').
            try:
                if "Date" in series.columns and ticker not in ref_override and len(series) >= 2:
                    now_vn = datetime.now(VN_TIMEZONE)
                    today_vn = now_vn.strftime("%Y-%m-%d")
                    last_date_str = str(last.get("Date", ""))
                    session_open = now_vn.hour < 15
                    if last_date_str == today_vn and session_open:
                        series = series.iloc[:-1].reset_index(drop=True)
                        last = series.iloc[-1]
                        last_close = float(last.get("Close", float("nan")))
                        last_volume = float(last.get("Volume", 0.0)) if not pd.isna(last.get("Volume")) else 0.0
                        prev_close = float("nan")
                        if len(series) >= 2:
                            prev = series.iloc[-2]
                            prev_close = float(prev.get("Close", float("nan")))
            except Exception:
                pass
            close_series = pd.to_numeric(series["Close"], errors="coerce")
            high_series = pd.to_numeric(series.get("High"), errors="coerce") if "High" in series else close_series
            low_series = pd.to_numeric(series.get("Low"), errors="coerce") if "Low" in series else close_series
            vol_series = pd.to_numeric(series.get("Volume"), errors="coerce") if "Volume" in series else pd.Series([float("nan")]*len(close_series))
            date_keys = series.get(date_field)
            if date_keys is None:
                date_keys = pd.Series(["" for _ in range(len(series))])
            else:
                date_keys = date_keys.astype(str)

            returns_series = close_series.pct_change()
            returns_series.index = date_keys
            valid_returns = returns_series.dropna()
            vol20 = valid_returns.tail(20).std()
            vol60 = valid_returns.tail(60).std()
            if benchmark and index_returns_lookup:
                index_series = pd.Series(
                    [index_returns_lookup.get(key, float("nan")) for key in date_keys],
                    index=date_keys,
                    dtype=float,
                )
                combined = pd.concat(
                    [returns_series, index_series], axis=1, keys=["asset", "index"]
                ).dropna()
                if len(combined) >= 2 and combined["index"].var() > 0:
                    beta60 = combined["asset"].cov(combined["index"]) / combined["index"].var()
                    corr60 = combined["asset"].corr(combined["index"])
                else:
                    beta60 = float("nan")
                    corr60 = float("nan")
            else:
                beta60 = float("nan")
                corr60 = float("nan")

            data: Dict[str, object] = {
                "Ticker": ticker,
                "Sector": industry_lookup.get(ticker, ""),
                "LastClose": last_close,
                "PreviousClose": prev_close,
                "Volume": last_volume,
            }
            intraday_volume = float("nan")
            intraday_value = float("nan")
            if ticker in intraday_lookup:
                snap = intraday_lookup[ticker]
                price = float(getattr(snap, "Price", last_close) or last_close)
                updated = getattr(snap, "TimeVN", "")
                source = "intraday"
                intraday_volume = float(getattr(snap, "IntradayVol_shares", float("nan")))
                intraday_value = float(getattr(snap, "IntradayValue_kVND", float("nan")))
            else:
                price = last_close
                updated = str(last.get("Date", ""))
                source = "close"
            data["IntradayVol_shares"] = intraday_volume
            data["IntradayValue_kVND"] = intraday_value
            # Best-effort: normalize price to HOSE tick grid to avoid float noise from intraday sources.
            if not pd.isna(price):
                price = round_to_tick(price)
            data["LastPrice"] = price
            if not pd.isna(last_close) and last_close:
                data["ChangePct"] = (price / last_close - 1.0) * 100.0
            else:
                data["ChangePct"] = float("nan")
            data["LastUpdated"] = updated
            data["PriceSource"] = source
            data["Vol20Pct"] = float(vol20 * 100.0) if pd.notna(vol20) else float("nan")
            data["Vol60Pct"] = float(vol60 * 100.0) if pd.notna(vol60) else float("nan")
            data["Beta60_Index"] = float(beta60) if pd.notna(beta60) else float("nan")
            data["Corr60_Index"] = float(corr60) if pd.notna(corr60) else float("nan")

            for window in self._config.moving_averages:
                if window <= 0:
                    continue
                col = f"SMA_{window}"
                if len(close_series) >= window:
                    data[col] = float(ma(close_series, window).iloc[-1])
                else:
                    data[col] = float("nan")
            # EMA
            for period in self._config.ema_periods:
                if period <= 0:
                    continue
                col = f"EMA_{period}"
                if len(close_series) >= period:
                    data[col] = float(ema(close_series, period).iloc[-1])
                else:
                    data[col] = float("nan")
            for period in self._config.rsi_periods:
                if period <= 0:
                    continue
                col = f"RSI_{period}"
                if len(close_series) > period:
                    data[col] = float(rsi_wilder(close_series, period).iloc[-1])
                else:
                    data[col] = float("nan")
            for period in self._config.atr_periods:
                if period <= 0:
                    continue
                col = f"ATR_{period}"
                if len(close_series) > period:
                    atr_val = float(atr_wilder(high_series, low_series, close_series, period).iloc[-1])
                    data[col] = atr_val
                    # ATR percentage vs last price
                    if last_close and not pd.isna(last_close) and last_close != 0:
                        data[f"ATRPct_{period}"] = atr_val / last_close * 100.0
                    else:
                        data[f"ATRPct_{period}"] = float("nan")
                else:
                    data[col] = float("nan")
                    data[f"ATRPct_{period}"] = float("nan")
            if len(close_series) >= max(self._config.macd_fast, self._config.macd_slow):
                close_numeric = pd.to_numeric(close_series, errors="coerce")
                macd_line = ema(close_numeric, self._config.macd_fast) - ema(
                    close_numeric, self._config.macd_slow
                )
                macd_signal = macd_line.ewm(span=self._config.macd_signal, adjust=False).mean()
                macd_value = float(macd_line.iloc[-1])
                macd_signal_value = float(macd_signal.iloc[-1])
                data["MACD"] = macd_value
                data["MACDSignal"] = macd_signal_value
                data["MACD_Hist"] = macd_value - macd_signal_value
            else:
                data["MACD"] = float("nan")
                data["MACDSignal"] = float("nan")
                data["MACD_Hist"] = float("nan")
            # Bollinger and z-score
            z_base_price = price if not pd.isna(price) else last_close
            for w in self._config.bollinger_windows:
                if w <= 1:
                    continue
                if len(close_series) >= w:
                    sma_w = ma(close_series, w).iloc[-1]
                    std_w = pd.to_numeric(close_series, errors="coerce").rolling(w).std().iloc[-1]
                    if pd.isna(sma_w) or pd.isna(std_w) or std_w == 0:
                        data[f"Z_{w}"] = float("nan")
                    else:
                        data[f"Z_{w}"] = float((z_base_price - float(sma_w)) / float(std_w))
                    if self._config.bollinger_include_bands:
                        k = self._config.bollinger_k
                        data[f"BBU_{w}_{int(k)}"] = float(sma_w + k * std_w) if not pd.isna(sma_w) and not pd.isna(std_w) else float("nan")
                        data[f"BBL_{w}_{int(k)}"] = float(sma_w - k * std_w) if not pd.isna(sma_w) and not pd.isna(std_w) else float("nan")
                else:
                    data[f"Z_{w}"] = float("nan")
                    if self._config.bollinger_include_bands:
                        k = self._config.bollinger_k
                        data[f"BBU_{w}_{int(k)}"] = float("nan")
                        data[f"BBL_{w}_{int(k)}"] = float("nan")
            # Rolling returns
            for p in self._config.returns_periods:
                if p <= 0:
                    continue
                col = f"Return_{p}"
                if len(close_series) > p and close_series.iloc[-p-1] and not pd.isna(close_series.iloc[-p-1]):
                    base = float(close_series.iloc[-p-1])
                    data[col] = (last_close / base - 1.0) * 100.0 if base else float("nan")
                else:
                    data[col] = float("nan")
            # ADV periods
            for p in self._config.adv_periods:
                if p <= 0:
                    continue
                col = f"ADV_{p}"
                if len(vol_series) >= p:
                    data[col] = float(pd.to_numeric(vol_series, errors="coerce").rolling(p).mean().iloc[-1])
                else:
                    data[col] = float("nan")
            intraday_vol_value = data.get("IntradayVol_shares")
            adv20_value = data.get("ADV_20")
            if (
                intraday_vol_value is not None
                and not pd.isna(intraday_vol_value)
                and adv20_value is not None
                and not pd.isna(adv20_value)
                and adv20_value > 0
            ):
                data["IntradayPctADV20"] = float(intraday_vol_value) / float(adv20_value)
            else:
                data["IntradayPctADV20"] = float("nan")
            # 52-week range context (close-based hi/lo)
            L = self._config.range_lookback_days
            if L > 1 and len(high_series) >= 1:
                # Use High/Low if available; fallback to Close
                window = min(L, len(high_series))
                max_hi = float(pd.concat([high_series.tail(window)], axis=1).max().iloc[-1]) if "High" in series else float(close_series.tail(window).max())
                min_lo = float(pd.concat([low_series.tail(window)], axis=1).min().iloc[-1]) if "Low" in series else float(close_series.tail(window).min())
                data["Hi_252"] = max_hi if L == 252 else max_hi
                data["Lo_252"] = min_lo if L == 252 else min_lo
                if max_hi and not pd.isna(max_hi):
                    data["PctFromHi_252"] = (last_close / max_hi - 1.0) * 100.0
                else:
                    data["PctFromHi_252"] = float("nan")
                if min_lo and not pd.isna(min_lo):
                    data["PctToLo_252"] = (last_close / min_lo - 1.0) * 100.0
                else:
                    data["PctToLo_252"] = float("nan")
            rows.append(data)
        snapshot = pd.DataFrame(rows)
        snapshot = snapshot.sort_values("Ticker").reset_index(drop=True)
        return snapshot


@dataclass
class PortfolioReport:
    positions: pd.DataFrame
    sector: pd.DataFrame


@dataclass
class PortfolioRefreshResult:
    reports: Dict[str, PortfolioReport]
    aggregate_positions: pd.DataFrame
    aggregate_sector: pd.DataFrame
    holdings: Dict[str, float]


def _load_portfolio_ticker_set(portfolio_dir: Path) -> set[str]:
    p_csv = portfolio_dir / "portfolio.csv"
    if not p_csv.exists():
        return set()
    try:
        df = pd.read_csv(p_csv, usecols=["Ticker"])
    except ValueError as exc:
        raise RuntimeError(f"Portfolio {p_csv} missing required column 'Ticker'") from exc
    return {
        str(t).strip().upper()
        for t in df["Ticker"].dropna().astype(str).tolist()
        if str(t).strip()
    }


class PortfolioReporter:
    def __init__(self, config: EngineConfig, industry_df: pd.DataFrame) -> None:
        self._config = config
        self._industry = industry_df

    @staticmethod
    def _empty_result() -> PortfolioRefreshResult:
        empty_positions = pd.DataFrame(
            columns=[
                "Ticker",
                "Quantity",
                "AvgPrice",
                "Last",
                "MarketValue_kVND",
                "CostBasis_kVND",
                "Unrealized_kVND",
                "PNLPct",
            ]
        )
        empty_sector = pd.DataFrame(columns=["Sector", "MarketValue_kVND", "WeightPct", "PNLPct"])
        return PortfolioRefreshResult({}, empty_positions, empty_sector, {})

    def refresh(self, snapshot: pd.DataFrame) -> PortfolioRefreshResult:
        portfolios_dir = self._config.portfolio_dir
        if not portfolios_dir.exists():
            LOGGER.info("Portfolio directory does not exist: %s; positions.csv will be empty", portfolios_dir)
            return self._empty_result()
        sector_lookup = {}
        if "Ticker" in self._industry.columns:
            for row in self._industry.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                sector = getattr(row, "Sector", "") if hasattr(row, "Sector") else ""
                sector_lookup[ticker] = sector

        file = portfolios_dir / "portfolio.csv"
        if not file.exists():
            LOGGER.info("Portfolio file not found: %s; positions.csv will be empty", file)
            return self._empty_result()
        portfolio_df = pd.read_csv(file)
        if portfolio_df.empty:
            return self._empty_result()
        if "Ticker" not in portfolio_df.columns or "Quantity" not in portfolio_df.columns or "AvgPrice" not in portfolio_df.columns:
            raise RuntimeError(f"Portfolio {file} missing required columns 'Ticker,Quantity,AvgPrice'")

        portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.upper()
        allowed = set(self._config.industry_ticker_filter or [])
        if allowed:
            before = len(portfolio_df)
            portfolio_df = portfolio_df[portfolio_df["Ticker"].isin(allowed)].reset_index(drop=True)
            if before and portfolio_df.empty:
                LOGGER.info(
                    "Ticker filter (%s) loại bỏ toàn bộ danh mục; positions.csv sẽ rỗng",
                    self._config.industry_ticker_filter_source or "unknown",
                )
            if portfolio_df.empty:
                return self._empty_result()

        report = self._build_report(portfolio_df, snapshot, sector_lookup)
        reports: Dict[str, PortfolioReport] = {"default": report} if report else {}
        if not report:
            return self._empty_result()

        combined = report.positions.copy()
        combined["Sector"] = combined["Sector"].fillna("Không rõ")
        agg_rows: List[Dict[str, object]] = []
        holdings: Dict[str, float] = {}
        for ticker, group in combined.groupby("Ticker", dropna=False):
            qty = float(group["Quantity"].sum())
            holdings[str(ticker)] = qty
            if qty == 0:
                avg_price = float("nan")
            else:
                avg_price = float((group["AvgPrice"] * group["Quantity"]).sum() / qty)
            last_vals = group["Last"].dropna()
            last_price = float(last_vals.iloc[0]) if not last_vals.empty else float("nan")
            market_value = float(group["MarketValue_kVND"].sum())
            cost_basis = float(group["CostBasis_kVND"].sum())
            unrealized = float(group["Unrealized_kVND"].sum())
            pnl_pct = float(unrealized / cost_basis) if cost_basis else float("nan")
            agg_rows.append(
                {
                    "Ticker": str(ticker),
                    "Quantity": qty,
                    "AvgPrice": avg_price,
                    "Last": last_price,
                    "MarketValue_kVND": market_value,
                    "CostBasis_kVND": cost_basis,
                    "Unrealized_kVND": unrealized,
                    "PNLPct": pnl_pct,
                }
            )
        aggregate_positions = pd.DataFrame(agg_rows).sort_values("Ticker").reset_index(drop=True)

        sector_summary = combined.groupby("Sector", dropna=False).agg(
            MarketValue_kVND=("MarketValue_kVND", "sum"),
            CostBasis_kVND=("CostBasis_kVND", "sum"),
        )
        total_market = float(sector_summary["MarketValue_kVND"].sum()) if not sector_summary.empty else 0.0
        if total_market:
            sector_summary["WeightPct"] = sector_summary["MarketValue_kVND"] / total_market
        else:
            sector_summary["WeightPct"] = 0.0
        sector_summary["PNLPct"] = sector_summary.apply(
            lambda row: ((row["MarketValue_kVND"] - row["CostBasis_kVND"]) / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sector_summary = (
            sector_summary.drop(columns=["CostBasis_kVND"]).reset_index().rename(columns={"index": "Sector"})
        )
        sector_summary = sector_summary.sort_values("Sector").reset_index(drop=True)

        return PortfolioRefreshResult(reports, aggregate_positions, sector_summary, holdings)

    def _build_report(
        self,
        portfolio_df: pd.DataFrame,
        snapshot: pd.DataFrame,
        sector_lookup: Dict[str, str],
    ) -> Optional[PortfolioReport]:
        df = portfolio_df.copy()
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)
        df = df[df["Quantity"] != 0]
        if df.empty:
            return None
        merged = df.merge(snapshot, on="Ticker", how="left")
        if "Sector" not in merged.columns:
            merged["Sector"] = pd.Series([float("nan")] * len(merged), index=merged.index)
        merged["Sector"] = merged["Sector"].fillna(merged["Ticker"].map(sector_lookup))
        merged["LastPrice"] = pd.to_numeric(merged.get("LastPrice"), errors="coerce")
        merged["LastClose"] = pd.to_numeric(merged.get("LastClose"), errors="coerce")
        merged["Last"] = merged["LastPrice"].fillna(merged["LastClose"])
        merged["Last"] = pd.to_numeric(merged["Last"], errors="coerce")
        merged["MarketValue_kVND"] = merged["Quantity"] * merged["Last"].fillna(0.0)
        merged["CostBasis_kVND"] = merged["Quantity"] * merged["AvgPrice"].fillna(0.0)
        merged["Unrealized_kVND"] = merged["MarketValue_kVND"] - merged["CostBasis_kVND"]
        merged["PNLPct"] = merged.apply(
            lambda row: (row["Unrealized_kVND"] / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sectorized = merged[[
            "Ticker",
            "Sector",
            "Quantity",
            "AvgPrice",
            "Last",
            "MarketValue_kVND",
            "CostBasis_kVND",
            "Unrealized_kVND",
            "PNLPct",
        ]].copy()
        sectorized["Sector"] = sectorized["Sector"].fillna("Không rõ")

        positions_output = sectorized.sort_values("Ticker").reset_index(drop=True)

        total_market = float(sectorized["MarketValue_kVND"].sum())
        sector_summary = sectorized.groupby("Sector", dropna=False).agg(
            MarketValue_kVND=("MarketValue_kVND", "sum"),
            CostBasis_kVND=("CostBasis_kVND", "sum"),
        )
        if total_market:
            sector_summary["WeightPct"] = sector_summary["MarketValue_kVND"] / total_market
        else:
            sector_summary["WeightPct"] = 0.0
        sector_summary["PNLPct"] = sector_summary.apply(
            lambda row: ((row["MarketValue_kVND"] - row["CostBasis_kVND"]) / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sector_summary = (
            sector_summary.drop(columns=["CostBasis_kVND"]).reset_index().rename(columns={"index": "Sector"})
        )
        sector_summary = sector_summary.sort_values("Sector").reset_index(drop=True)

        return PortfolioReport(positions=positions_output, sector=sector_summary)


def _build_technical_output(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Sector",
        "Last",
        "Ref",
        "ChangePct",
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "RSI14",
        "ATR14",
        "Vol20Pct",
        "Vol60Pct",
        "MACD",
        "MACDSignal",
        "Beta60_Index",
        "Corr60_Index",
        "Z20",
        "Ret5d",
        "Ret20d",
        "Ret60d",
        "ADV20",
        "IntradayVol_shares",
        "IntradayValue_kVND",
        "IntradayPctADV20",
        "PE_fwd",
        "PB",
        "ROE",
        "High52w",
        "Low52w",
    ]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)

    tickers = snapshot.get("Ticker", pd.Series([], dtype=object)).astype(str).str.upper()

    def numeric(name: str) -> pd.Series:
        if name in snapshot.columns:
            return pd.to_numeric(snapshot[name], errors="coerce")
        return pd.Series([float("nan")] * len(snapshot), index=snapshot.index)

    sector = snapshot.get("Sector")
    if sector is None:
        sector = pd.Series(["" for _ in range(len(snapshot))], index=snapshot.index, dtype=object)
    else:
        sector = sector.astype("string").fillna("").astype(object)

    last = numeric("LastPrice")
    ref = numeric("LastClose")
    last = last.fillna(ref)
    ref = ref.fillna(last)
    change = pd.Series([float("nan")] * len(snapshot), index=snapshot.index)
    valid_mask = (~last.isna()) & (~ref.isna()) & (ref != 0)
    change.loc[valid_mask] = ((last.loc[valid_mask] / ref.loc[valid_mask]) - 1.0) * 100.0

    sma20 = numeric("SMA_20")
    sma50 = numeric("SMA_50")
    sma200 = numeric("SMA_200")
    ema20 = numeric("EMA_20")
    rsi14 = numeric("RSI_14")
    atr14 = numeric("ATR_14")
    vol20 = numeric("Vol20Pct")
    vol60 = numeric("Vol60Pct")
    macd = numeric("MACD")
    macd_signal = numeric("MACDSignal")
    beta60 = numeric("Beta60_Index")
    corr60 = numeric("Corr60_Index")

    ret5 = numeric("Return_5")
    ret20 = numeric("Return_20")
    ret60 = numeric("Return_60")
    adv20 = numeric("ADV_20")
    intraday_vol = numeric("IntradayVol_shares")
    intraday_value = numeric("IntradayValue_kVND")
    intraday_pct_adv = numeric("IntradayPctADV20")
    pe_fwd = numeric("PE_fwd")
    pb = numeric("PB")
    roe_pct = numeric("ROE")
    hi52 = numeric("Hi_252")
    lo52 = numeric("Lo_252")

    # Prefer true 20-session z-score from snapshot if present (Z_20).
    z20 = numeric("Z_20")
    if z20.isna().all():
        z20 = pd.Series([float("nan")] * len(snapshot), index=snapshot.index)
        with_atr = (~atr14.isna()) & (atr14 != 0)
        z20.loc[with_atr] = ((last - sma20) / atr14).loc[with_atr]
        z20.loc[atr14 == 0] = 0.0

    technical = pd.DataFrame(
        {
            "Ticker": tickers,
            "Sector": sector,
            "Last": last,
            "Ref": ref,
            "ChangePct": change,
            "SMA20": sma20,
            "SMA50": sma50,
            "SMA200": sma200,
            "EMA20": ema20,
            "RSI14": rsi14,
            "ATR14": atr14,
            "Vol20Pct": vol20,
            "Vol60Pct": vol60,
            "MACD": macd,
            "MACDSignal": macd_signal,
            "Beta60_Index": beta60,
            "Corr60_Index": corr60,
            "Z20": z20,
            "Ret5d": ret5,
            "Ret20d": ret20,
            "Ret60d": ret60,
            "ADV20": adv20,
            "IntradayVol_shares": intraday_vol,
            "IntradayValue_kVND": intraday_value,
            "IntradayPctADV20": intraday_pct_adv,
            "PE_fwd": pe_fwd,
            "PB": pb,
            "ROE": roe_pct,
            "High52w": hi52,
            "Low52w": lo52,
        }
    )
    technical = technical.sort_values("Ticker").reset_index(drop=True)
    return technical


def _build_bands(technical: pd.DataFrame) -> pd.DataFrame:
    columns = ["Ticker", "Ref", "Ceil", "Floor", "TickSize"]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for row in technical.itertuples(index=False):
        ref = float(getattr(row, "Ref", float("nan")))
        if math.isnan(ref):
            ceil_val = float("nan")
            floor_val = float("nan")
            tick_size = float("nan")
        else:
            tick_size = _tick_size(ref)
            ceil_val = floor_to_tick(ref * 1.07)
            floor_val = ceil_to_tick(ref * 0.93)
            if not math.isnan(ceil_val) and not math.isnan(floor_val) and ceil_val < floor_val:
                ceil_val, floor_val = floor_val, ceil_val
        rows.append(
            {
                "Ticker": getattr(row, "Ticker"),
                "Ref": ref,
                "Ceil": ceil_val,
                "Floor": floor_val,
                "TickSize": tick_size,
            }
        )
    df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return df


def _compute_vnindex_metrics(history_df: pd.DataFrame) -> Dict[str, object]:
    result = {"IndexRegimeTag": "", "VNINDEX_ATR14PctRank": float("nan")}
    if history_df.empty or "Ticker" not in history_df.columns:
        return result
    index_df = history_df[history_df["Ticker"].astype(str).str.upper() == "VNINDEX"].copy()
    if index_df.empty:
        return result
    index_df = index_df.sort_values("t" if "t" in index_df.columns else "Date")
    close = pd.to_numeric(index_df.get("Close"), errors="coerce")
    high = pd.to_numeric(index_df.get("High"), errors="coerce")
    low = pd.to_numeric(index_df.get("Low"), errors="coerce")
    valid = close.notna() & high.notna() & low.notna()
    close = close[valid]
    high = high[valid]
    low = low[valid]
    if len(close) < 20:
        return result
    atr_series = atr_wilder(high, low, close, 14)
    atr_pct = (atr_series / close) * 100.0
    atr_pct = atr_pct.dropna()
    if not atr_pct.empty:
        last_val = float(atr_pct.iloc[-1])
        rank = float(((atr_pct <= last_val).sum() / len(atr_pct)) * 100.0)
        result["VNINDEX_ATR14PctRank"] = rank
    last_close = float(close.iloc[-1])
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else float("nan")
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")
    regime = "Sideway"
    if not pd.isna(sma50) and not pd.isna(sma200):
        if last_close > sma50 and sma50 > sma200:
            regime = "Bull"
        elif last_close < sma50 and sma50 < sma200:
            regime = "Bear"
    result["IndexRegimeTag"] = regime
    return result


def _compute_sector_metrics(market_df: pd.DataFrame) -> tuple[Dict[str, str], Dict[str, float]]:
    if market_df.empty or "Sector" not in market_df.columns:
        return {}, {}
    sector_regime: Dict[str, str] = {}
    sector_rank: Dict[str, float] = {}
    grouped = market_df.groupby("Sector")
    pct_threshold = 1.0
    for sector, group in grouped:
        ret20 = pd.to_numeric(group.get("Ret20d"), errors="coerce")
        median_ret = ret20.dropna().median()
        if pd.isna(median_ret):
            tag = "Sideway"
        elif median_ret >= pct_threshold:
            tag = "Bull"
        elif median_ret <= -pct_threshold:
            tag = "Bear"
        else:
            tag = "Sideway"
        sector_regime[sector] = tag
    adtv = grouped["ADTV20_shares"].sum().replace({0: float("nan")})
    ranks = adtv.rank(method="min", ascending=False, pct=True) * 100.0
    for sector, value in ranks.items():
        sector_rank[sector] = float(value)
    return sector_regime, sector_rank


def _clamp_score(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(max(lower, min(upper, value)))


def _safe_numeric_mean(values: object) -> float:
    if values is None:
        return float("nan")
    series = pd.to_numeric(values, errors="coerce")
    if isinstance(series, pd.Series):
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.mean())
    return float(series) if not pd.isna(series) else float("nan")


def _safe_numeric_median(values: object) -> float:
    if values is None:
        return float("nan")
    series = pd.to_numeric(values, errors="coerce")
    if isinstance(series, pd.Series):
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.median())
    return float(series) if not pd.isna(series) else float("nan")


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):  # pragma: no cover - defensive
        return value
    return value


def _build_close_matrix(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty or "Ticker" not in history_df.columns or "Close" not in history_df.columns:
        return pd.DataFrame()

    df = history_df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    if "Date" in df.columns:
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "t" in df.columns:
        df["_date"] = pd.to_numeric(df["t"], errors="coerce")
    else:
        return pd.DataFrame()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Close", "_date"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["_date", "Ticker"])
    daily = df.groupby(["_date", "Ticker"], as_index=False)["Close"].last()
    close_matrix = daily.pivot(index="_date", columns="Ticker", values="Close").sort_index()
    close_matrix = close_matrix.loc[:, ~close_matrix.columns.duplicated()]
    return close_matrix


def _latest_range_window_metrics(series: pd.Series, window: int) -> Dict[str, float]:
    result = {
        "pos": float("nan"),
        "dist_upper_pct": float("nan"),
        "dist_lower_pct": float("nan"),
        "drawdown_pct": float("nan"),
        "rebound_pct": float("nan"),
    }
    if series is None:
        return result
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < window or window <= 0:
        return result
    window_series = clean.tail(window)
    last = float(window_series.iloc[-1])
    hi = float(window_series.max())
    lo = float(window_series.min())
    span = hi - lo
    if span == 0:
        pos = 0.5
    else:
        pos = (last - lo) / span
    if last:
        result["dist_upper_pct"] = ((hi - last) / last) * 100.0
        result["dist_lower_pct"] = ((last - lo) / last) * 100.0
    if hi:
        result["drawdown_pct"] = ((last / hi) - 1.0) * 100.0
    if lo:
        result["rebound_pct"] = ((last / lo) - 1.0) * 100.0
    result["pos"] = _clamp_score(pos, 0.0, 1.0)
    return result


def _latest_breadth_above_ma(close_matrix: pd.DataFrame, window: int) -> float:
    if close_matrix.empty or window <= 0:
        return float("nan")
    ma = close_matrix.rolling(window).mean()
    latest_close = close_matrix.iloc[-1]
    latest_ma = ma.iloc[-1]
    valid = latest_close.notna() & latest_ma.notna()
    if not bool(valid.any()):
        return float("nan")
    above = (latest_close[valid] > latest_ma[valid]).mean()
    return float(above * 100.0)


def _latest_breadth_positive(close_matrix: pd.DataFrame, periods: int) -> float:
    if close_matrix.empty or periods <= 0:
        return float("nan")
    latest_ret = close_matrix.pct_change(periods).iloc[-1]
    valid = latest_ret.notna()
    if not bool(valid.any()):
        return float("nan")
    positive = (latest_ret[valid] > 0).mean()
    return float(positive * 100.0)


def _latest_new_extreme_pct(close_matrix: pd.DataFrame, window: int, mode: str) -> float:
    if close_matrix.empty or window <= 1:
        return float("nan")
    latest = close_matrix.iloc[-1]
    if mode == "high":
        rolling = close_matrix.rolling(window).max().iloc[-1]
        comparison = latest >= rolling
    else:
        rolling = close_matrix.rolling(window).min().iloc[-1]
        comparison = latest <= rolling
    valid = latest.notna() & rolling.notna()
    if not bool(valid.any()):
        return float("nan")
    return float(comparison[valid].mean() * 100.0)


def _build_market_breadth_summary(
    history_df: pd.DataFrame,
    index_metrics: Dict[str, object],
    *,
    working_tickers: Iterable[str] | None = None,
    breadth_tickers: Iterable[str] | None = None,
) -> tuple[Dict[str, object], pd.DataFrame]:
    summary: Dict[str, object] = {
        "VNINDEX_ATR14PctRank": index_metrics.get("VNINDEX_ATR14PctRank", float("nan")),
        "UniverseTickerCount": 0,
        "BreadthUniverseTickerCount": 0,
        "BreadthSource": "working_universe",
        "IndexRangePos20": float("nan"),
        "IndexRangePos60": float("nan"),
        "IndexRangePos120": float("nan"),
        "IndexDistToUpper20Pct": float("nan"),
        "IndexDistToLower20Pct": float("nan"),
        "IndexDistToUpper60Pct": float("nan"),
        "IndexDistToLower60Pct": float("nan"),
        "IndexDistToUpper120Pct": float("nan"),
        "IndexDistToLower120Pct": float("nan"),
        "IndexDrawdownFromHigh60Pct": float("nan"),
        "IndexReboundFromLow60Pct": float("nan"),
        "BreadthAboveSMA20Pct": float("nan"),
        "BreadthAboveSMA50Pct": float("nan"),
        "BreadthAboveSMA200Pct": float("nan"),
        "BreadthPositive1dPct": float("nan"),
        "BreadthPositive5dPct": float("nan"),
        "AdvanceDeclineRatio": float("nan"),
        "NewHigh20Pct": float("nan"),
        "NewLow20Pct": float("nan"),
        "MarketDispersion20Pct": float("nan"),
        "MarketCoMovement20Pct": float("nan"),
        "MarketMedianCorr20": float("nan"),
    }
    ticker_context = pd.DataFrame(
        columns=["Ticker", "CoMoveWithIndex20Pct", "Corr20_Index", "Beta20_Index"]
    )

    close_matrix = _build_close_matrix(history_df)
    if close_matrix.empty:
        return summary, ticker_context

    asset_columns = [
        c
        for c in close_matrix.columns
        if c not in {"VNINDEX", "VN30", "VN100"}
        and not re.fullmatch(r"VN(INDEX|30|100)", str(c), re.IGNORECASE)
    ]
    asset_close = close_matrix[asset_columns] if asset_columns else pd.DataFrame(index=close_matrix.index)

    normalized_working = {
        str(ticker).strip().upper() for ticker in (working_tickers or asset_columns) if str(ticker).strip()
    }
    summary["UniverseTickerCount"] = int(
        len([ticker for ticker in asset_columns if str(ticker).strip().upper() in normalized_working])
    )

    normalized_breadth = {
        str(ticker).strip().upper() for ticker in (breadth_tickers or asset_columns) if str(ticker).strip()
    }
    breadth_columns = [ticker for ticker in asset_columns if str(ticker).strip().upper() in normalized_breadth]
    if not breadth_columns:
        breadth_columns = asset_columns
    breadth_close = close_matrix[breadth_columns] if breadth_columns else pd.DataFrame(index=close_matrix.index)
    summary["BreadthUniverseTickerCount"] = len(breadth_columns)
    if breadth_columns != asset_columns:
        summary["BreadthSource"] = "benchmark_basket"

    if not breadth_close.empty:
        summary["BreadthAboveSMA20Pct"] = _latest_breadth_above_ma(breadth_close, 20)
        summary["BreadthAboveSMA50Pct"] = _latest_breadth_above_ma(breadth_close, 50)
        summary["BreadthAboveSMA200Pct"] = _latest_breadth_above_ma(breadth_close, 200)
        summary["BreadthPositive1dPct"] = _latest_breadth_positive(breadth_close, 1)
        summary["BreadthPositive5dPct"] = _latest_breadth_positive(breadth_close, 5)
        summary["NewHigh20Pct"] = _latest_new_extreme_pct(breadth_close, 20, "high")
        summary["NewLow20Pct"] = _latest_new_extreme_pct(breadth_close, 20, "low")
        latest_returns_1d = breadth_close.pct_change().iloc[-1]
        valid_returns_1d = latest_returns_1d.dropna()
        if not valid_returns_1d.empty:
            advances = int((valid_returns_1d > 0).sum())
            declines = int((valid_returns_1d < 0).sum())
            if declines == 0:
                summary["AdvanceDeclineRatio"] = float(advances) if advances else float("nan")
            else:
                summary["AdvanceDeclineRatio"] = float(advances / declines)
        latest_returns_20d = breadth_close.pct_change(20).iloc[-1].dropna()
        if not latest_returns_20d.empty:
            summary["MarketDispersion20Pct"] = float(latest_returns_20d.std() * 100.0)

    if "VNINDEX" in close_matrix.columns:
        index_close = pd.to_numeric(close_matrix["VNINDEX"], errors="coerce").dropna()
        for window in (20, 60, 120):
            metrics = _latest_range_window_metrics(index_close, window)
            summary[f"IndexRangePos{window}"] = metrics["pos"]
            summary[f"IndexDistToUpper{window}Pct"] = metrics["dist_upper_pct"]
            summary[f"IndexDistToLower{window}Pct"] = metrics["dist_lower_pct"]
            if window == 60:
                summary["IndexDrawdownFromHigh60Pct"] = metrics["drawdown_pct"]
                summary["IndexReboundFromLow60Pct"] = metrics["rebound_pct"]

        returns = close_matrix.pct_change()
        index_returns = pd.to_numeric(returns["VNINDEX"], errors="coerce")
        rows: List[Dict[str, object]] = []
        for ticker in asset_columns:
            asset_returns = pd.to_numeric(returns[ticker], errors="coerce")
            pair = pd.concat([asset_returns, index_returns], axis=1, keys=["asset", "index"]).dropna().tail(20)
            if pair.empty:
                corr20 = float("nan")
                beta20 = float("nan")
                comove20 = float("nan")
            else:
                comove20 = float((((pair["asset"] > 0) == (pair["index"] > 0)).mean()) * 100.0)
                if len(pair) >= 5 and pair["index"].var() > 0:
                    corr20 = float(pair["asset"].corr(pair["index"]))
                    beta20 = float(pair["asset"].cov(pair["index"]) / pair["index"].var())
                else:
                    corr20 = float("nan")
                    beta20 = float("nan")
            rows.append(
                {
                    "Ticker": ticker,
                    "CoMoveWithIndex20Pct": comove20,
                    "Corr20_Index": corr20,
                    "Beta20_Index": beta20,
                }
            )
        ticker_context = pd.DataFrame(rows)
        if not ticker_context.empty:
            summary["MarketCoMovement20Pct"] = float(
                pd.to_numeric(ticker_context["CoMoveWithIndex20Pct"], errors="coerce").dropna().mean()
            )
            summary["MarketMedianCorr20"] = float(
                pd.to_numeric(ticker_context["Corr20_Index"], errors="coerce").dropna().median()
            )

    return summary, ticker_context


def _add_relative_strength_columns(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return market_df

    df = market_df.copy()
    ticker_series = df["Ticker"].astype(str).str.upper()
    is_index = ticker_series == "VNINDEX"
    asset_mask = ~ticker_series.str.contains(r"VN(?:INDEX|30|100)", na=False)

    index_ret20 = float("nan")
    index_ret60 = float("nan")
    if is_index.any():
        index_row = df.loc[is_index].iloc[0]
        index_ret20 = float(pd.to_numeric(index_row.get("Ret20d"), errors="coerce"))
        index_ret60 = float(pd.to_numeric(index_row.get("Ret60d"), errors="coerce"))

    ret20 = pd.to_numeric(df.get("Ret20d"), errors="coerce")
    ret60 = pd.to_numeric(df.get("Ret60d"), errors="coerce")
    df["Ret20dVsIndex"] = ret20 - index_ret20 if not pd.isna(index_ret20) else float("nan")
    df["Ret60dVsIndex"] = ret60 - index_ret60 if not pd.isna(index_ret60) else float("nan")

    rel_rank_20 = pd.Series([float("nan")] * len(df), index=df.index)
    valid_20 = asset_mask & df["Ret20dVsIndex"].notna()
    if bool(valid_20.any()):
        rel_rank_20.loc[valid_20] = df.loc[valid_20, "Ret20dVsIndex"].rank(method="average", pct=True) * 100.0
    df["RelStrength20Rank"] = rel_rank_20

    rel_rank_60 = pd.Series([float("nan")] * len(df), index=df.index)
    valid_60 = asset_mask & df["Ret60dVsIndex"].notna()
    if bool(valid_60.any()):
        rel_rank_60.loc[valid_60] = df.loc[valid_60, "Ret60dVsIndex"].rank(method="average", pct=True) * 100.0
    df["RelStrength60Rank"] = rel_rank_60

    return df


def _build_sector_summary(market_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    if market_df.empty or "Sector" not in market_df.columns:
        return pd.DataFrame(columns=columns)

    ticker_series = market_df["Ticker"].astype(str).str.upper()
    asset_df = market_df[~ticker_series.str.contains(r"VN(?:INDEX|30|100)", na=False)].copy()
    if asset_df.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    for sector, group in asset_df.groupby("Sector", dropna=False):
        ticker_count = int(len(group))
        breadth20 = float((pd.to_numeric(group.get("DistSMA20Pct"), errors="coerce") > 0).mean() * 100.0)
        breadth50 = float((pd.to_numeric(group.get("DistSMA50Pct"), errors="coerce") > 0).mean() * 100.0)
        breadth_positive5 = float((pd.to_numeric(group.get("Ret5d"), errors="coerce") > 0).mean() * 100.0)
        median_ret5 = _safe_numeric_median(group.get("Ret5d"))
        median_ret20 = _safe_numeric_median(group.get("Ret20d"))
        median_ret60 = _safe_numeric_median(group.get("Ret60d"))
        median_rel20 = _safe_numeric_median(group.get("Ret20dVsIndex"))
        median_rel60 = _safe_numeric_median(group.get("Ret60dVsIndex"))
        mean_comove = _safe_numeric_mean(group.get("CoMoveWithIndex20Pct"))
        foreign_flow_5d = _safe_numeric_median(group.get("NetBuySellForeign_kVND_5d"))
        foreign_flow_20d = _safe_numeric_median(group.get("NetBuySellForeign_kVND_20d"))
        proprietary_flow_5d = _safe_numeric_median(group.get("NetBuySellProprietary_kVND_5d"))
        proprietary_flow_20d = _safe_numeric_median(group.get("NetBuySellProprietary_kVND_20d"))

        rows.append(
            {
                "Sector": sector,
                "TickerCount": ticker_count,
                "SectorBreadthAboveSMA20Pct": breadth20,
                "SectorBreadthAboveSMA50Pct": breadth50,
                "SectorBreadthPositive5dPct": breadth_positive5,
                "SectorMedianRet5d": median_ret5,
                "SectorMedianRet20d": median_ret20,
                "SectorMedianRet60d": median_ret60,
                "SectorMedianRet20dVsIndex": median_rel20,
                "SectorMedianRet60dVsIndex": median_rel60,
                "SectorMeanCoMoveWithIndex20Pct": mean_comove,
                "SectorMedianForeignFlow5d_kVND": foreign_flow_5d,
                "SectorMedianForeignFlow20d_kVND": foreign_flow_20d,
                "SectorMedianProprietaryFlow5d_kVND": proprietary_flow_5d,
                "SectorMedianProprietaryFlow20d_kVND": proprietary_flow_20d,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=columns)

    adtv = asset_df.groupby("Sector", dropna=False)["ADTV20_shares"].sum().replace({0: float("nan")})
    sector_rank = adtv.rank(method="min", ascending=False, pct=True) * 100.0
    summary["SectorADTVRank"] = summary["Sector"].map(sector_rank.to_dict())
    summary = summary[columns].sort_values(["Sector"]).reset_index(drop=True)
    return summary


def _round_and_clamp(value: float, floor_value: float, ceil_value: float) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    rounded = round_to_tick(value)
    if pd.isna(rounded):
        return float("nan")
    return clamp_price(rounded, floor_value, ceil_value)


def _build_levels(technical: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Preset",
        "SideDeclared",
        "NearTouchBuy",
        "NearTouchSell",
        "Opp1Buy",
        "Opp1Sell",
        "Opp2Buy",
        "Opp2Sell",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    merged = technical.merge(bands, on="Ticker", how="left", suffixes=("", "_band"))
    rows: List[Dict[str, object]] = []
    presets = [
        ("momentum", "BUY"),
        ("mean_reversion", "BOTH"),
        ("balanced", "BOTH"),
        ("risk_off", "SELL"),
    ]
    for row in merged.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        last = float(getattr(row, "Last", float("nan")))
        sma20 = float(getattr(row, "SMA20", float("nan")))
        atr14 = float(getattr(row, "ATR14", float("nan")))
        floor_value = float(getattr(row, "Floor", float("nan")))
        ceil_value = float(getattr(row, "Ceil", float("nan")))
        base_tick = _tick_size(last if not math.isnan(last) else getattr(row, "Ref", float("nan")))

        def near_values(preset: str) -> Tuple[float, float]:
            if preset == "momentum":
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return last + base_tick, last + 2 * base_tick
            if preset == "mean_reversion":
                if not math.isnan(sma20) and not math.isnan(atr14):
                    return sma20 - 0.5 * atr14, sma20 + 0.5 * atr14
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return last - base_tick, last + base_tick
            if preset == "balanced":
                anchor = last
                if not math.isnan(sma20):
                    if math.isnan(anchor):
                        anchor = sma20
                    else:
                        anchor = (anchor + sma20) / 2.0
                if math.isnan(base_tick):
                    return anchor, anchor
                return anchor - base_tick, anchor + base_tick
            if preset == "risk_off":
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return float("nan"), last - base_tick
            return float("nan"), float("nan")

        for preset, side in presets:
            near_buy_raw, near_sell_raw = near_values(preset)
            near_buy = _round_and_clamp(near_buy_raw, floor_value, ceil_value)
            near_sell = _round_and_clamp(near_sell_raw, floor_value, ceil_value)
            opp1_buy = float("nan")
            opp2_buy = float("nan")
            opp1_sell = float("nan")
            opp2_sell = float("nan")
            if not math.isnan(near_buy) and not math.isnan(sma20) and not math.isnan(atr14):
                opp1_buy = _round_and_clamp(min(near_buy, sma20 - 0.5 * atr14), floor_value, ceil_value)
                opp2_buy = _round_and_clamp(min(near_buy, sma20 - 1.0 * atr14), floor_value, ceil_value)
            if not math.isnan(near_sell) and not math.isnan(sma20) and not math.isnan(atr14):
                opp1_sell = _round_and_clamp(max(near_sell, sma20 + 0.5 * atr14), floor_value, ceil_value)
                opp2_sell = _round_and_clamp(max(near_sell, sma20 + 1.0 * atr14), floor_value, ceil_value)
            rows.append(
                {
                    "Ticker": ticker,
                    "Preset": preset,
                    "SideDeclared": side,
                    "NearTouchBuy": near_buy,
                    "NearTouchSell": near_sell,
                    "Opp1Buy": opp1_buy,
                    "Opp1Sell": opp1_sell,
                    "Opp2Buy": opp2_buy,
                    "Opp2Sell": opp2_sell,
                }
            )
    levels = pd.DataFrame(rows)
    levels = levels.sort_values(["Ticker", "Preset"]).reset_index(drop=True)
    return levels


def _round_to_lot(value: float, lot: int, mode: str = "round") -> int:
    if value is None or pd.isna(value) or lot <= 0:
        return 0
    units = float(value) / lot
    if mode == "floor":
        qty = math.floor(units)
    elif mode == "ceil":
        qty = math.ceil(units)
    else:
        qty = round(units)
    return int(qty * lot)


def _build_sizing(
    technical: pd.DataFrame,
    holdings: Dict[str, float],
    config: EngineConfig,
) -> pd.DataFrame:
    columns = [
        "Ticker",
        "TargetQty",
        "CurrentQty",
        "DeltaQty",
        "MaxOrderQty",
        "SliceCount",
        "SliceQty",
        "LiquidityScore_kVND",
        "VolatilityScore",
        "TodayFilledQty",
        "TodayWAP",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for row in technical.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        last = float(getattr(row, "Last", float("nan")))
        atr = float(getattr(row, "ATR14", float("nan")))
        adv20 = float(getattr(row, "ADV20", float("nan")))
        current_qty = float(holdings.get(ticker, 0.0))
        target_qty = current_qty
        delta_qty = target_qty - current_qty
        max_by_adv = config.max_order_pct_adv * adv20 if not math.isnan(adv20) else 0.0
        raw_max_order = min(max_by_adv, float(config.max_qty_per_order)) if adv20 and adv20 > 0 else 0.0
        max_order_qty = _round_to_lot(raw_max_order, config.min_lot, mode="floor")
        abs_delta = abs(delta_qty)
        if abs_delta == 0:
            slice_count = 0
        else:
            adv_slice = config.slice_adv_ratio * adv20 if not math.isnan(adv20) else 0.0
            if adv_slice <= 0:
                slice_count = 1
            else:
                slice_count = max(1, math.ceil(abs_delta / adv_slice))
        if slice_count > 0:
            slice_qty = _round_to_lot(abs_delta / slice_count, config.min_lot)
        else:
            slice_qty = 0
        liquidity = float(adv20 * last) if not math.isnan(adv20) and not math.isnan(last) else float("nan")
        if not math.isnan(last) and last > 0 and not math.isnan(atr):
            volatility = atr / last
        else:
            volatility = 0.0
        rows.append(
            {
                "Ticker": ticker,
                "TargetQty": int(round(target_qty)),
                "CurrentQty": int(round(current_qty)),
                "DeltaQty": int(round(delta_qty)),
                "MaxOrderQty": int(max_order_qty),
                "SliceCount": int(slice_count),
                "SliceQty": int(slice_qty),
                "LiquidityScore_kVND": liquidity,
                "VolatilityScore": volatility,
                "TodayFilledQty": 0,
                "TodayWAP": float("nan"),
            }
        )
    sizing = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return sizing


def _build_signals(technical: pd.DataFrame, bands: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "PresetFitMomentum",
        "PresetFitMeanRev",
        "PresetFitBalanced",
        "BandDistance",
        "SectorBias",
        "RiskGuards",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    snapshot_meta = snapshot[["Ticker", "PriceSource"]] if "PriceSource" in snapshot.columns else pd.DataFrame(columns=["Ticker", "PriceSource"])
    merged = technical.merge(bands, on="Ticker", how="left").merge(snapshot_meta, on="Ticker", how="left")
    rows: List[Dict[str, object]] = []
    for row in merged.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        rsi = float(getattr(row, "RSI14", float("nan")))
        ret20 = float(getattr(row, "Ret20d", float("nan")))
        macd = float(getattr(row, "MACD", float("nan")))
        macd_signal = float(getattr(row, "MACDSignal", float("nan")))
        z20 = float(getattr(row, "Z20", float("nan")))
        atr = float(getattr(row, "ATR14", float("nan")))
        last = float(getattr(row, "Last", float("nan")))
        floor_value = float(getattr(row, "Floor", float("nan")))
        ceil_value = float(getattr(row, "Ceil", float("nan")))
        adv20 = float(getattr(row, "ADV20", float("nan")))
        price_source = getattr(row, "PriceSource", "")

        momentum_score = 0.0
        if not math.isnan(rsi) and rsi >= 55:
            momentum_score += 0.4
        if not math.isnan(ret20) and ret20 >= 0:
            momentum_score += 0.3
        if not math.isnan(macd) and not math.isnan(macd_signal) and macd > macd_signal:
            momentum_score += 0.3
        momentum_score = min(momentum_score, 1.0)

        mean_rev_score = 0.0
        if not math.isnan(z20) and z20 <= -0.5:
            mean_rev_score += 0.5
        if not math.isnan(atr) and atr > 0 and not math.isnan(last) and not math.isnan(floor_value):
            if (last - floor_value) <= atr:
                mean_rev_score += 0.5
        mean_rev_score = min(mean_rev_score, 1.0)

        balanced = (momentum_score + mean_rev_score) / 2.0

        if math.isnan(atr):
            band_distance = float("nan")
        elif atr == 0:
            band_distance = 0.0
        else:
            candidates = []
            if not math.isnan(ceil_value) and not math.isnan(last):
                candidates.append((ceil_value - last) / atr)
            if not math.isnan(last) and not math.isnan(floor_value):
                candidates.append((last - floor_value) / atr)
            band_distance = min(candidates) if candidates else float("nan")

        risk_flags: List[str] = []
        if math.isnan(atr) or atr == 0:
            risk_flags.append("ZERO_ATR")
        if math.isnan(adv20) or adv20 < 10_000:
            risk_flags.append("LOW_LIQ")
        tick = _tick_size(last)
        if not math.isnan(tick) and not math.isnan(last):
            if (not math.isnan(ceil_value) and (ceil_value - last) <= tick) or (not math.isnan(floor_value) and (last - floor_value) <= tick):
                risk_flags.append("NEAR_LIMIT")
        if str(price_source).lower() != "intraday":
            risk_flags.append("STALE_SNAPSHOT")

        rows.append(
            {
                "Ticker": ticker,
                "PresetFitMomentum": momentum_score,
                "PresetFitMeanRev": mean_rev_score,
                "PresetFitBalanced": balanced,
                "BandDistance": band_distance,
                "SectorBias": 0,
                "RiskGuards": "|".join(sorted(set(risk_flags))),
            }
        )
    signals = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return signals


def _ticks_between(upper: float, lower: float, tick: float) -> float:
    upper_dec = _as_decimal(upper)
    lower_dec = _as_decimal(lower)
    tick_dec = _as_decimal(tick)
    if upper_dec is None or lower_dec is None or tick_dec is None or tick_dec == 0:
        return float("nan")
    diff = upper_dec - lower_dec
    if diff < 0:
        diff = Decimal(0)
    steps = diff / tick_dec
    return float(steps)


def _compute_grid_levels(start: float, direction: int, floor_value: float, ceil_value: float) -> List[float]:
    levels: List[float] = []
    price = start
    for _ in range(3):
        if price is None or math.isnan(price):
            levels.append(float("nan"))
            price = float("nan")
            continue
        tick = _tick_size(price)
        if tick is None or math.isnan(tick) or tick <= 0:
            levels.append(float("nan"))
            price = float("nan")
            continue
        candidate = price + direction * tick
        candidate = round_to_tick(candidate)
        candidate = clamp_price(candidate, floor_value, ceil_value)
        levels.append(candidate)
        price = candidate
    return levels


def _build_market_dataset(
    technical: pd.DataFrame,
    bands: pd.DataFrame,
    sizing: pd.DataFrame,
    signals: pd.DataFrame,
    config: EngineConfig,
) -> pd.DataFrame:
    """Combine core prompt datasets into a single per-ticker view."""

    technical_cols = [c for c in technical.columns if c != "Ticker"]
    band_cols = [c for c in ("Ceil", "Floor", "TickSize") if c in bands.columns]
    signal_cols = [c for c in ("BandDistance",) if c in signals.columns]

    if technical.empty:
        columns = [
            "Ticker",
            *technical_cols,
            *band_cols,
            "LotSize",
            "FloorValid",
            "CeilValid",
            "ValidBid1",
            "ValidAsk1",
            "TicksToFloor",
            "TicksToCeil",
            "ATR14Pct",
            "ADTV20_shares",
            "ADTV20_kVND",
            "DistRefPct",
            "DistSMA20Pct",
            "DistSMA50Pct",
            "GridBelow_T1",
            "GridBelow_T2",
            "GridBelow_T3",
            "GridAbove_T1",
            "GridAbove_T2",
            "GridAbove_T3",
            *signal_cols,
        ]
        return pd.DataFrame(columns=columns)

    merged = technical.copy()

    if not bands.empty:
        band_payload = bands.drop(columns=[c for c in ("Ref",) if c in bands.columns])
        merged = merged.merge(band_payload, on="Ticker", how="left")

    if not signals.empty:
        merged = merged.merge(signals, on="Ticker", how="left")

    merged = merged.sort_values("Ticker").reset_index(drop=True)

    lot_size = int(config.min_lot)
    merged["LotSize"] = lot_size

    last = merged.get("Last", pd.Series(dtype=float))
    ref = merged.get("Ref", pd.Series(dtype=float))
    sma20 = merged.get("SMA20", pd.Series(dtype=float))
    sma50 = merged.get("SMA50", pd.Series(dtype=float))
    atr14 = merged.get("ATR14", pd.Series(dtype=float))
    adv20 = merged.get("ADV20", pd.Series(dtype=float))
    ceil_series = merged.get("Ceil", pd.Series(dtype=float))
    floor_series = merged.get("Floor", pd.Series(dtype=float))
    tick_series = merged.get("TickSize", pd.Series(dtype=float))

    atr_pct = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_atr = (~atr14.isna()) & (~last.isna()) & (last != 0)
    atr_pct.loc[valid_atr] = (atr14.loc[valid_atr] / last.loc[valid_atr]) * 100.0
    merged["ATR14Pct"] = atr_pct

    merged["ADTV20_shares"] = adv20
    liquidity = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_liq = (~adv20.isna()) & (~last.isna())
    liquidity.loc[valid_liq] = adv20.loc[valid_liq] * last.loc[valid_liq]
    merged["ADTV20_kVND"] = liquidity

    adtv_rank = pd.Series([float("nan")] * len(merged), index=merged.index)
    adtv_pct_rank = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_adtv = ~liquidity.isna()
    if valid_adtv.any():
        adtv_rank.loc[valid_adtv] = liquidity.loc[valid_adtv].rank(method="min", ascending=False)
        adtv_pct_rank.loc[valid_adtv] = liquidity.loc[valid_adtv].rank(
            method="min", ascending=False, pct=True
        )
    merged["ADTV20Rank"] = adtv_rank
    merged["ADTV20PctRank"] = adtv_pct_rank * 100.0

    lot_series = pd.to_numeric(merged.get("LotSize"), errors="coerce") if "LotSize" in merged else pd.Series([float("nan")] * len(merged), index=merged.index)
    merged["OneLotNotional_kVND"] = lot_series * last
    merged["OneLotATR_kVND"] = lot_series * atr14

    slippage = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_slippage = (~tick_series.isna()) & (~last.isna()) & (last != 0)
    slippage.loc[valid_slippage] = (
        tick_series.loc[valid_slippage] / last.loc[valid_slippage]
    ) * 100.0
    merged["SlippageOneTickPct"] = slippage

    hi52 = merged.get("High52w", pd.Series([float("nan")] * len(merged), index=merged.index))
    lo52 = merged.get("Low52w", pd.Series([float("nan")] * len(merged), index=merged.index))
    range_span = hi52 - lo52
    pos52 = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_range = (~hi52.isna()) & (~lo52.isna()) & (~last.isna()) & (range_span != 0)
    pos52.loc[valid_range] = (last.loc[valid_range] - lo52.loc[valid_range]) / range_span.loc[valid_range]
    merged["Pos52wPct"] = pos52.clip(lower=0.0, upper=1.0)

    def _above_indicator(sma: pd.Series) -> pd.Series:
        indicator = pd.Series([float("nan")] * len(merged), index=merged.index)
        valid = (~last.isna()) & (~sma.isna())
        indicator.loc[valid] = (last.loc[valid] > sma.loc[valid]).astype(int)
        return indicator

    merged["AboveSMA20"] = _above_indicator(sma20)
    merged["AboveSMA50"] = _above_indicator(sma50)
    sma200 = merged.get("SMA200", pd.Series([float("nan")] * len(merged), index=merged.index))
    merged["AboveSMA200"] = _above_indicator(sma200)

    dist_ref = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_ref = (~last.isna()) & (~ref.isna()) & (ref != 0)
    dist_ref.loc[valid_ref] = (last.loc[valid_ref] / ref.loc[valid_ref] - 1.0) * 100.0
    merged["DistRefPct"] = dist_ref

    dist_sma20 = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_sma20 = (~last.isna()) & (~sma20.isna()) & (sma20 != 0)
    dist_sma20.loc[valid_sma20] = (last.loc[valid_sma20] / sma20.loc[valid_sma20] - 1.0) * 100.0
    merged["DistSMA20Pct"] = dist_sma20

    dist_sma50 = pd.Series([float("nan")] * len(merged), index=merged.index)
    valid_sma50 = (~last.isna()) & (~sma50.isna()) & (sma50 != 0)
    dist_sma50.loc[valid_sma50] = (last.loc[valid_sma50] / sma50.loc[valid_sma50] - 1.0) * 100.0
    merged["DistSMA50Pct"] = dist_sma50

    floor_valid = (~floor_series.isna()) & (~last.isna()) & (last > floor_series)
    ceil_valid = (~ceil_series.isna()) & (~last.isna()) & (last < ceil_series)
    merged["FloorValid"] = floor_valid
    merged["CeilValid"] = ceil_valid

    def _feature_row(row: pd.Series) -> pd.Series:
        last_val = float(row.get("Last", float("nan")))
        floor_val = float(row.get("Floor", float("nan")))
        ceil_val = float(row.get("Ceil", float("nan")))
        tick_val = float(row.get("TickSize", float("nan")))
        ref_val = float(row.get("Ref", float("nan")))
        if math.isnan(tick_val):
            base = last_val if not math.isnan(last_val) else ref_val
            if not math.isnan(base):
                tick_val = _tick_size(base)

        if math.isnan(last_val) or math.isnan(tick_val) or tick_val <= 0:
            bid1 = ask1 = float("nan")
            ticks_floor = ticks_ceil = float("nan")
            below_levels = [float("nan")] * 3
            above_levels = [float("nan")] * 3
        else:
            bid1_candidate = round_to_tick(last_val - tick_val)
            bid1 = clamp_price(bid1_candidate, floor_val, ceil_val)
            ask1_candidate = round_to_tick(last_val + tick_val)
            ask1 = clamp_price(ask1_candidate, floor_val, ceil_val)

            tick_for_calc = tick_val if not math.isnan(tick_val) and tick_val > 0 else float("nan")
            ticks_floor = _ticks_between(last_val, floor_val, tick_for_calc)
            ticks_ceil = _ticks_between(ceil_val, last_val, tick_for_calc)

            below_levels = _compute_grid_levels(last_val, -1, floor_val, ceil_val)
            above_levels = _compute_grid_levels(last_val, 1, floor_val, ceil_val)

        return pd.Series(
            {
                "ValidBid1": bid1,
                "ValidAsk1": ask1,
                "TicksToFloor": ticks_floor,
                "TicksToCeil": ticks_ceil,
                "GridBelow_T1": below_levels[0],
                "GridBelow_T2": below_levels[1],
                "GridBelow_T3": below_levels[2],
                "GridAbove_T1": above_levels[0],
                "GridAbove_T2": above_levels[1],
                "GridAbove_T3": above_levels[2],
            }
        )

    extras = merged.apply(_feature_row, axis=1)
    merged = pd.concat([merged, extras], axis=1)

    if "Sector" not in merged.columns:
        merged["Sector"] = ""

    ordered_columns = [
        "Ticker",
        "Sector",
        "Last",
        "Ref",
        "ChangePct",
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "RSI14",
        "ATR14",
        "ATR14Pct",
        "Vol20Pct",
        "Vol60Pct",
        "MACD",
        "MACDSignal",
        "Beta60_Index",
        "Corr60_Index",
        "Z20",
        "Ret5d",
        "Ret20d",
        "Ret60d",
        "IntradayVol_shares",
        "IntradayValue_kVND",
        "IntradayPctADV20",
        "ADTV20_shares",
        "ADTV20Rank",
        "ADTV20PctRank",
        "ForeignFlowDate",
        "ForeignRoomRemaining_shares",
        "ForeignHoldingPct",
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
        "High52w",
        "Low52w",
        "Ceil",
        "Floor",
        "TickSize",
        "LotSize",
        "OneLotATR_kVND",
        "FloorValid",
        "CeilValid",
        "ValidBid1",
        "ValidAsk1",
        "TicksToFloor",
        "TicksToCeil",
        "SlippageOneTickPct",
        "DistRefPct",
        "DistSMA20Pct",
        "DistSMA50Pct",
        "Pos52wPct",
        "GridBelow_T1",
        "GridBelow_T2",
        "GridBelow_T3",
        "GridAbove_T1",
        "GridAbove_T2",
        "GridAbove_T3",
        *signal_cols,
    ]
    ordered_existing = [c for c in ordered_columns if c in merged.columns]
    ordered_remaining = [c for c in merged.columns if c not in ordered_existing]
    merged = merged[ordered_existing + ordered_remaining]
    merged = merged.sort_values("Ticker").reset_index(drop=True)
    return merged


class DataEngine:
    """Coordinates data collection, indicator computation, and report generation."""

    def __init__(
        self,
        config: EngineConfig,
        data_service: MarketDataService,
        vn30_fetcher: Optional[Callable[[], set[str]]] = None,
    ) -> None:
        self._config = config
        self._data_service = data_service
        self._vn30_fetcher = vn30_fetcher or fetch_vn30_members

    def run(self) -> Dict[str, object]:
        self._wipe_output_dir()
        self._prepare_directories()
        universe_df = self._load_universe()
        tickers = self._resolve_tickers(universe_df)
        vn30_members = self._vn30_fetcher()
        LOGGER.info("Processing %d tickers", len(tickers))
        benchmark_tickers = ["VNINDEX"]
        benchmark_set = set(benchmark_tickers)
        breadth_benchmark_tickers = sorted(
            {
                str(ticker).strip().upper()
                for ticker in vn30_members
                if str(ticker).strip() and str(ticker).strip().upper() not in benchmark_set
            }
        )
        use_benchmark_breadth = len(tickers) <= 3 and bool(breadth_benchmark_tickers)
        history_tickers = sorted(
            {
                *tickers,
                *benchmark_tickers,
                *(breadth_benchmark_tickers if use_benchmark_breadth else []),
            }
        )
        intraday_tickers = sorted({*tickers, *benchmark_tickers})
        history_df = self._data_service.load_history(history_tickers)
        intraday_df = self._data_service.load_intraday(intraday_tickers)
        snapshot_builder = TechnicalSnapshotBuilder(self._config)
        snapshot = snapshot_builder.build(history_df, intraday_df, universe_df)
        snapshot["Ticker"] = snapshot["Ticker"].astype(str).str.upper()
        index_metrics = _compute_vnindex_metrics(history_df)
        market_summary, ticker_context_df = _build_market_breadth_summary(
            history_df,
            index_metrics,
            working_tickers=tickers,
            breadth_tickers=(breadth_benchmark_tickers if use_benchmark_breadth else tickers),
        )
        benchmark_snapshot = snapshot[snapshot["Ticker"].isin(benchmark_set)].reset_index(drop=True)
        snapshot = snapshot[snapshot["Ticker"].isin(tickers)].reset_index(drop=True)
        technical_df = _build_technical_output(snapshot)
        fundamental_feature_df: Optional[pd.DataFrame] = None
        if self._config.cafef_flow_enabled:
            flow_metrics_df = build_flow_feature_frame(
                tickers,
                self._config.cafef_flow_cache_dir,
                self._config.cafef_flow_max_age_hours,
            )
            if not flow_metrics_df.empty:
                technical_df = technical_df.merge(flow_metrics_df, on="Ticker", how="left")
        if self._config.vietstock_overview_enabled:
            fundamentals = build_fundamental_frame(
                tickers,
                self._config.vietstock_overview_cache_dir,
                self._config.vietstock_overview_max_age_hours,
            )
            if fundamentals:
                fundamental_feature_df = pd.DataFrame([
                    {"Ticker": key, **values} for key, values in fundamentals.items()
                ])
        bands_df = _build_bands(technical_df)
        # levels.csv removed from outputs; keep computation disabled
        portfolio_result = PortfolioReporter(self._config, universe_df).refresh(snapshot)
        sizing_df = _build_sizing(technical_df, portfolio_result.holdings, self._config)
        signals_df = _build_signals(technical_df, bands_df, snapshot)
        positions_df = portfolio_result.aggregate_positions
        _sector_df = portfolio_result.aggregate_sector  # kept for diagnostics/stats if needed

        run_started = datetime.now(VN_TIMEZONE)

        out_dir = self._config.output_base_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        universe_path = out_dir / "universe.csv"
        positions_path = out_dir / "positions.csv"
        market_summary_path = out_dir / "market_summary.json"
        sector_summary_path = out_dir / "sector_summary.csv"
        try:
            positions_df.to_csv(positions_path, index=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to write positions CSV to {positions_path}: {exc}")

        # Apply output schema trims per latest requirements
        def _drop_by_names(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            cols = [c for c in df.columns if c not in set(names)]
            return df[cols]

        # levels.csv removed entirely – no trimming or writing

        signal_trim_cols = [
            "PresetFitMomentum",
            "PresetFitMeanRev",
            "PresetFitBalanced",
            "RiskGuards",
            "SectorBias",
        ]
        sizing_trim_cols = [
            "TargetQty",
            "DeltaQty",
            "SliceCount",
            "SliceQty",
        ]

        # Only keep objective/raw outputs in universe.csv.
        if not signals_df.empty:
            signals_df = _drop_by_names(signals_df, signal_trim_cols)

        # sizing.csv: drop TargetQty, DeltaQty, SliceCount, SliceQty
        if not sizing_df.empty:
            sizing_df = _drop_by_names(sizing_df, sizing_trim_cols)

        market_df = _build_market_dataset(technical_df, bands_df, sizing_df, signals_df, self._config)
        columns_to_prune = [
            "ADV20",
            "ADTV20_kVND",
            "OneLotNotional_kVND",
            "AboveSMA20",
            "AboveSMA50",
            "AboveSMA200",
            "BandDistance",
        ]

        def _prune_columns(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            drop_cols = [col for col in columns_to_prune if col in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            return df

        market_df = _prune_columns(market_df)
        if fundamental_feature_df is not None and not fundamental_feature_df.empty:
            market_df = market_df.merge(
                fundamental_feature_df, on="Ticker", how="left", suffixes=("", "_fund")
            )
            for col in ("PE_fwd", "PB", "ROE"):
                fund_col = f"{col}_fund"
                if fund_col in market_df.columns:
                    if col in market_df.columns:
                        market_df[col] = market_df[fund_col].combine_first(market_df[col])
                    else:
                        market_df[col] = market_df[fund_col]
                    market_df = market_df.drop(columns=[fund_col])

        if not benchmark_snapshot.empty:
            benchmark_technical_df = _build_technical_output(benchmark_snapshot)
            benchmark_bands_df = _build_bands(benchmark_technical_df)
            benchmark_sizing_df = _build_sizing(
                benchmark_technical_df, portfolio_result.holdings, self._config
            )
            benchmark_signals_df = _build_signals(
                benchmark_technical_df, benchmark_bands_df, benchmark_snapshot
            )
            if not benchmark_signals_df.empty:
                benchmark_signals_df = _drop_by_names(benchmark_signals_df, signal_trim_cols)
            if not benchmark_sizing_df.empty:
                benchmark_sizing_df = _drop_by_names(benchmark_sizing_df, sizing_trim_cols)
            benchmark_market_df = _build_market_dataset(
                benchmark_technical_df,
                benchmark_bands_df,
                benchmark_sizing_df,
                benchmark_signals_df,
                self._config,
            )
            benchmark_market_df = _prune_columns(benchmark_market_df)
            market_df = pd.concat([benchmark_market_df, market_df], ignore_index=True)
        market_df = _prune_columns(market_df)
        if not ticker_context_df.empty:
            market_df = market_df.merge(ticker_context_df, on="Ticker", how="left")
        for col in ("CoMoveWithIndex20Pct", "Corr20_Index", "Beta20_Index"):
            if col not in market_df.columns:
                market_df[col] = float("nan")
        market_df = _add_relative_strength_columns(market_df)

        sector_summary_df = _build_sector_summary(market_df)
        if not sector_summary_df.empty:
            sector_index = sector_summary_df.set_index("Sector")
            for col in (
                "SectorBreadthAboveSMA20Pct",
                "SectorBreadthAboveSMA50Pct",
                "SectorBreadthPositive5dPct",
                "SectorADTVRank",
            ):
                market_df[col] = market_df["Sector"].map(sector_index[col].to_dict())
            market_df["Ret20dVsSector"] = (
                pd.to_numeric(market_df.get("Ret20d"), errors="coerce")
                - market_df["Sector"].map(sector_index["SectorMedianRet20d"].to_dict())
            )
            market_df["Ret60dVsSector"] = (
                pd.to_numeric(market_df.get("Ret60d"), errors="coerce")
                - market_df["Sector"].map(sector_index["SectorMedianRet60d"].to_dict())
            )
        else:
            for col in (
                "SectorBreadthAboveSMA20Pct",
                "SectorBreadthAboveSMA50Pct",
                "SectorBreadthPositive5dPct",
                "SectorADTVRank",
                "Ret20dVsSector",
                "Ret60dVsSector",
            ):
                market_df[col] = (
                    float("nan")
                    if col.startswith("Ret")
                    or col.endswith("Pct")
                    or col.endswith("Rank")
                    else ""
                )

        market_df["VNINDEX_ATR14PctRank"] = market_summary.get("VNINDEX_ATR14PctRank", float("nan"))

        # Đánh dấu thành phần VN30 (1 = thuộc VN30, 0 = ngoài VN30)
        if "Sector" in market_df.columns:
            insert_at = market_df.columns.get_loc("Sector") + 1
        else:
            insert_at = 1
        vn30_flags = (
            market_df["Ticker"].astype(str).str.upper().isin(vn30_members).astype(int)
        )
        market_df.insert(insert_at, "IsVN30", vn30_flags)

        position_column_map = {
            "Quantity": "PositionQuantity",
            "AvgPrice": "PositionAvgPrice",
            "MarketValue_kVND": "PositionMarketValue_kVND",
            "CostBasis_kVND": "PositionCostBasis_kVND",
            "Unrealized_kVND": "PositionUnrealized_kVND",
            "PNLPct": "PositionPNLPct",
        }
        positions_for_merge = positions_df.copy()
        if "Last" in positions_for_merge.columns:
            positions_for_merge = positions_for_merge.drop(columns=["Last"])
        positions_for_merge = positions_for_merge.rename(columns=position_column_map)

        universe_df = market_df.merge(positions_for_merge, on="Ticker", how="left")
        if "PositionQuantity" in universe_df.columns:
            universe_df["PositionQuantity"] = pd.to_numeric(
                universe_df["PositionQuantity"], errors="coerce"
            ).fillna(0.0)
        for col in [
            "PositionMarketValue_kVND",
            "PositionCostBasis_kVND",
            "PositionUnrealized_kVND",
        ]:
            if col in universe_df.columns:
                universe_df[col] = pd.to_numeric(universe_df[col], errors="coerce").fillna(0.0)

        if "PositionQuantity" in universe_df.columns and "ADTV20_shares" in universe_df.columns:
            adtv_shares = pd.to_numeric(universe_df["ADTV20_shares"], errors="coerce")
            position_qty = pd.to_numeric(universe_df["PositionQuantity"], errors="coerce")
            position_pct_adv = pd.Series([float("nan")] * len(universe_df), index=universe_df.index)
            valid_adv = (~adtv_shares.isna()) & (adtv_shares != 0)
            position_pct_adv.loc[valid_adv] = (
                position_qty.loc[valid_adv] / adtv_shares.loc[valid_adv]
            ) * 100.0
            insert_at = universe_df.columns.get_loc("PositionQuantity") + 1
            universe_df.insert(insert_at, "PositionPctADV20", position_pct_adv)

            atr_base = universe_df.get("ATR14")
            if atr_base is not None:
                atr_series = pd.to_numeric(atr_base, errors="coerce")
            else:
                atr_series = pd.Series([float("nan")] * len(universe_df), index=universe_df.index)
            position_atr = atr_series * position_qty
            if "PositionUnrealized_kVND" in universe_df.columns:
                insert_at = universe_df.columns.get_loc("PositionUnrealized_kVND") + 1
            else:
                insert_at = len(universe_df.columns)
            universe_df.insert(insert_at, "PositionATR_kVND", position_atr)

        if "PositionMarketValue_kVND" in universe_df.columns:
            portfolio_values = pd.to_numeric(universe_df["PositionMarketValue_kVND"], errors="coerce").fillna(0.0)
            total_market = float(portfolio_values.sum())
            engine_nav = pd.Series([total_market] * len(universe_df), index=universe_df.index, dtype=float)
            insert_at = universe_df.columns.get_loc("PositionMarketValue_kVND")
            universe_df.insert(insert_at, "EnginePortfolioMarketValue_kVND", engine_nav)
            if total_market:
                weights = (portfolio_values / total_market) * 100.0
            else:
                weights = pd.Series(
                    [0.0] * len(universe_df), index=universe_df.index, dtype=float
                )
            insert_at = universe_df.columns.get_loc("PositionMarketValue_kVND") + 1
            universe_df.insert(insert_at, "PositionWeightPct", weights)

        if "Sector" in universe_df.columns and "PositionWeightPct" in universe_df.columns:
            sector_weight_map = (
                universe_df.groupby("Sector")["PositionWeightPct"].sum().to_dict()
            )
            universe_df["SectorWeightPct"] = universe_df["Sector"].map(sector_weight_map).fillna(0.0)
        else:
            universe_df["SectorWeightPct"] = 0.0

        if "PositionWeightPct" in universe_df.columns and "Beta60_Index" in universe_df.columns:
            weights = pd.to_numeric(universe_df["PositionWeightPct"], errors="coerce").fillna(0.0) / 100.0
            betas = pd.to_numeric(universe_df["Beta60_Index"], errors="coerce").fillna(0.0)
            universe_df["BetaContribution"] = weights * betas
        else:
            universe_df["BetaContribution"] = float("nan")

        universe_df.insert(1, "EngineRunAt", run_started.isoformat())
        priority_order = {ticker: idx for idx, ticker in enumerate(benchmark_tickers)}
        if priority_order:
            ticker_series = universe_df["Ticker"].astype(str).str.upper()
            default_rank = len(priority_order)
            universe_df = (
                universe_df.assign(
                    _priority=ticker_series.map(priority_order).fillna(default_rank),
                    _ticker_sort=ticker_series,
                )
                .sort_values(["_priority", "_ticker_sort"])
                .drop(columns=["_priority", "_ticker_sort"])
                .reset_index(drop=True)
            )
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        universe_df.to_csv(universe_path, index=False)

        market_summary_payload = {**market_summary, "GeneratedAt": run_started.isoformat()}
        market_summary_path.write_text(
            json.dumps(
                {key: _json_safe_value(value) for key, value in market_summary_payload.items()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        sector_summary_output = sector_summary_df.copy()
        if "EngineRunAt" not in sector_summary_output.columns:
            sector_summary_output.insert(0, "EngineRunAt", run_started.isoformat())
        sector_summary_output.to_csv(sector_summary_path, index=False)

        return {
            "tickers": len(tickers),
            "snapshot_rows": len(snapshot),
            "output": str(universe_path),
            "market_summary": str(market_summary_path),
            "sector_summary": str(sector_summary_path),
            "generated_at": run_started.isoformat(),
        }

    def _prepare_directories(self) -> None:
        self._config.output_base_dir.mkdir(parents=True, exist_ok=True)
        self._config.presets_dir.mkdir(parents=True, exist_ok=True)
        self._config.portfolios_dir.mkdir(parents=True, exist_ok=True)
        self._config.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self._config.market_cache_dir.mkdir(parents=True, exist_ok=True)
        self._config.portfolio_dir.mkdir(parents=True, exist_ok=True)
        if self._config.cafef_flow_enabled:
            self._config.cafef_flow_cache_dir.mkdir(parents=True, exist_ok=True)
        if self._config.vietstock_overview_enabled:
            self._config.vietstock_overview_cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_universe(self) -> pd.DataFrame:
        if not self._config.universe_csv.exists():
            raise RuntimeError(f"Universe CSV not found: {self._config.universe_csv}")
        df = pd.read_csv(self._config.universe_csv)
        if "Ticker" not in df.columns:
            raise RuntimeError("Universe CSV must contain 'Ticker' column")
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        if not self._config.include_indices:
            df = df[~df["Ticker"].str.contains("VNINDEX|VN30|VN100", na=False)]
        allowed = set(self._config.industry_ticker_filter or [])
        if allowed:
            before = len(df)
            df = df[df["Ticker"].isin(allowed)].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(
                    f"Ticker filter ({self._config.industry_ticker_filter_source or 'unknown'}) "
                    "loại bỏ toàn bộ universe; kiểm tra cấu hình filter"
                )
            dropped = before - len(df)
            if dropped:
                LOGGER.info(
                    "Ticker filter (%s) đã bỏ %d mã khỏi universe gốc",
                    self._config.industry_ticker_filter_source or "unknown",
                    dropped,
                )
        return df

    def _resolve_tickers(self, universe_df: pd.DataFrame) -> List[str]:
        tickers = set(universe_df["Ticker"].tolist())
        portfolio_tickers = _load_portfolio_ticker_set(self._config.portfolio_dir)
        allowed = set(self._config.industry_ticker_filter or [])
        if allowed:
            filtered = sorted({t for t in tickers if t in allowed})
            if not filtered:
                raise RuntimeError(
                    f"Ticker filter ({self._config.industry_ticker_filter_source or 'unknown'}) "
                    "khiến danh sách mã rỗng sau khi gộp portfolio; kiểm tra cấu hình filter"
                )
            dropped_portfolio = sorted(portfolio_tickers - allowed)
            if dropped_portfolio:
                LOGGER.info(
                    "Ticker filter (%s) bỏ qua các mã trong portfolio: %s",
                    self._config.industry_ticker_filter_source or "unknown",
                    ", ".join(dropped_portfolio),
                )
            return filtered
        tickers |= portfolio_tickers
        clean = sorted({t.strip().upper() for t in tickers if t and isinstance(t, str)})
        return clean

    def _wipe_output_dir(self) -> None:
        out_dir = self._config.output_base_dir
        if not out_dir.exists():
            return
        # Keep caches under out/ (e.g. out/data, out/cafef_flows, out/vietstock_overview).
        # Only remove generated artifacts so repeated runs don't re-fetch everything.
        generated_files = [
            out_dir / "universe.csv",
            out_dir / "positions.csv",
            out_dir / "market_summary.json",
            out_dir / "sector_summary.csv",
        ]
        removed_any = False
        for p in generated_files:
            try:
                if p.exists():
                    p.unlink()
                    removed_any = True
            except Exception as exc:
                raise RuntimeError(f"Failed to remove generated output file {p}: {exc}")
        if removed_any:
            LOGGER.info("Cleared generated output files under: %s", out_dir)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect market data and compute technical indicators")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data_engine.yaml"),
        help="Path to engine YAML configuration",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary as JSON to stdout",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = EngineConfig.from_yaml(Path(args.config))
    service = VndirectMarketDataService(config)
    engine = DataEngine(config, service)
    summary = engine.run()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
