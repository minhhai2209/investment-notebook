from __future__ import annotations

import argparse
import io
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import requests
import yaml


DEFAULT_CONFIG_PATH = Path("config/macro_factors.yaml")
DEFAULT_CACHE_DIR = Path("out/macro_factors")
USER_AGENT = "broker-gpt-3 macro-factor-cache/1.0"
DEFAULT_START_DATE = "2018-01-01"


@dataclass(frozen=True)
class MacroFactorSpec:
    name: str
    label: str
    source: str
    series_id: Optional[str] = None
    symbol: Optional[str] = None
    value_column: str = "Close"
    start_date: str = DEFAULT_START_DATE


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_macro_specs(config_path: Path) -> Dict[str, MacroFactorSpec]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    factors = raw.get("factors", {}) or {}
    specs: Dict[str, MacroFactorSpec] = {}
    for name, payload in factors.items():
        spec = MacroFactorSpec(
            name=str(name).strip(),
            label=str(payload.get("label") or name).strip(),
            source=str(payload.get("source") or "").strip().lower(),
            series_id=str(payload.get("series_id")).strip() if payload.get("series_id") else None,
            symbol=str(payload.get("symbol")).strip() if payload.get("symbol") else None,
            value_column=str(payload.get("value_column") or "Close").strip(),
            start_date=str(payload.get("start_date") or DEFAULT_START_DATE).strip(),
        )
        if spec.source not in {"fred", "stooq"}:
            raise ValueError(f"Unsupported factor source for {name}: {spec.source}")
        specs[spec.name] = spec
    if not specs:
        raise ValueError(f"No macro factors configured in {config_path}")
    return specs


def parse_fred_csv_text(text: str, series_id: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    expected_columns = {"observation_date", series_id}
    if not expected_columns.issubset(frame.columns):
        raise RuntimeError(f"FRED response for {series_id} is not a CSV payload")
    out = frame.rename(columns={"observation_date": "Date", series_id: "Value"})[["Date", "Value"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return out.reset_index(drop=True)


def parse_stooq_csv_text(text: str, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    expected_columns = {"Date", value_column}
    if not expected_columns.issubset(frame.columns):
        raise RuntimeError(f"Stooq response is missing Date/{value_column} columns")
    out = frame[["Date", value_column]].rename(columns={value_column: "Value"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return out.reset_index(drop=True)


def _download_text(url: str, session: requests.Session, timeout: int) -> str:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as first_exc:
        curl = subprocess.run(
            ["curl", "-fsSL", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
        )
        if curl.returncode == 0 and curl.stdout:
            return curl.stdout
        raise first_exc


def fetch_factor_frame(spec: MacroFactorSpec, session: Optional[requests.Session] = None, timeout: int = 30) -> pd.DataFrame:
    owned_session = session is None
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    if spec.source == "fred":
        if not spec.series_id:
            raise ValueError(f"Missing series_id for factor {spec.name}")
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={spec.series_id}&cosd={spec.start_date}"
        frame = parse_fred_csv_text(_download_text(url, session, timeout), spec.series_id)
    elif spec.source == "stooq":
        if not spec.symbol:
            raise ValueError(f"Missing symbol for factor {spec.name}")
        url = f"https://stooq.com/q/d/l/?s={spec.symbol}&i=d"
        frame = parse_stooq_csv_text(_download_text(url, session, timeout), spec.value_column)
    else:
        raise ValueError(f"Unsupported factor source: {spec.source}")
    if owned_session:
        session.close()
    if frame.empty:
        raise RuntimeError(f"No usable data returned for factor {spec.name}")
    frame = frame[frame["Date"] >= pd.Timestamp(spec.start_date)].reset_index(drop=True)
    return frame


def factor_cache_path(cache_dir: Path, factor_name: str) -> Path:
    return cache_dir / f"{factor_name}.csv"


def _is_cache_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    if max_age_hours <= 0:
        return True
    age = _utc_now() - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age <= timedelta(hours=max_age_hours)


def refresh_macro_factor_cache(
    config_path: Path,
    cache_dir: Path,
    factor_names: Optional[Sequence[str]] = None,
    max_age_hours: int = 24,
    timeout: int = 30,
) -> pd.DataFrame:
    specs = load_macro_specs(config_path)
    requested = {name.strip() for name in factor_names or specs.keys() if str(name).strip()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows = []
    try:
        for factor_name in requested:
            if factor_name not in specs:
                raise KeyError(f"Unknown macro factor: {factor_name}")
            spec = specs[factor_name]
            cache_path = factor_cache_path(cache_dir, factor_name)
            refreshed = False
            error = ""
            if not _is_cache_fresh(cache_path, max_age_hours):
                try:
                    frame = fetch_factor_frame(spec, session=session, timeout=timeout)
                    frame.to_csv(cache_path, index=False)
                    refreshed = True
                except Exception as exc:
                    error = str(exc)
                    if not cache_path.exists():
                        rows.append(
                            {
                                "Factor": factor_name,
                                "Label": spec.label,
                                "Source": spec.source,
                                "Refreshed": 0,
                                "RowCount": 0,
                                "LatestDate": "",
                                "LatestValue": float("nan"),
                                "CachePath": str(cache_path),
                                "Error": error,
                            }
                        )
                        continue
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            latest = cached.iloc[-1] if not cached.empty else None
            rows.append(
                {
                    "Factor": factor_name,
                    "Label": spec.label,
                    "Source": spec.source,
                    "Refreshed": int(refreshed),
                    "RowCount": int(len(cached)),
                    "LatestDate": latest["Date"].date().isoformat() if latest is not None else "",
                    "LatestValue": float(latest["Value"]) if latest is not None else float("nan"),
                    "CachePath": str(cache_path),
                    "Error": error,
                }
            )
    finally:
        session.close()
    return pd.DataFrame(rows).sort_values("Factor").reset_index(drop=True)


def load_macro_factor_matrix(cache_dir: Path, factor_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    frames = []
    paths = []
    if factor_names:
        paths = [factor_cache_path(cache_dir, name) for name in factor_names]
    else:
        paths = sorted(cache_dir.glob("*.csv"))
    for csv_path in paths:
        if not csv_path.exists():
            continue
        factor_name = csv_path.stem
        frame = pd.read_csv(csv_path, parse_dates=["Date"])
        if frame.empty or "Date" not in frame.columns or "Value" not in frame.columns:
            continue
        series = frame[["Date", "Value"]].copy().rename(columns={"Value": factor_name}).dropna()
        frames.append(series.set_index("Date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh/cache macro factor time series.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Macro factor YAML config.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory for cached factor CSVs.")
    parser.add_argument("--factor", action="append", default=[], help="Optional factor name(s) to refresh.")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Reuse cache newer than this many hours. 0 means trust cache forever.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--summary-out", default="out/analysis/macro_factor_cache_summary.csv", help="Optional CSV summary output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = refresh_macro_factor_cache(
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        factor_names=args.factor or None,
        max_age_hours=args.max_age_hours,
        timeout=args.timeout,
    )
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    payload = {
        "factor_count": int(len(summary)),
        "refreshed_count": int(summary["Refreshed"].sum()) if not summary.empty else 0,
        "summary_csv": str(summary_out),
    }
    (summary_out.with_suffix(".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary.empty:
        print("No macro factor data cached.")
    else:
        print(summary[["Factor", "Source", "Refreshed", "LatestDate", "LatestValue"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
