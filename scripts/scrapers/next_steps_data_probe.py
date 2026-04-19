#!/usr/bin/env python3
"""Collect NEXT_STEPS data using API -> CafeF -> Vietstock priority."""
from __future__ import annotations

import json
import sys
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.data_fetching.collect_intraday import ensure_intraday_latest_df
from scripts.data_fetching.fetch_ticker_data import fetch_history
from scripts.scrapers.cafef_probe import CafeFProbe
from scripts.scrapers.vietstock_next_steps_probe import VietstockNextStepsProbe

VN_TZ = timezone(timedelta(hours=7))


@dataclass
class BucketResult:
    status: str
    source: str
    detail: Any
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure detail is JSON serializable (convert DataFrames, sets, etc.).
        detail = data.get("detail")
        if isinstance(detail, pd.DataFrame):
            data["detail"] = detail.to_dict(orient="records")
        return data


class NextStepsDataProbe:
    """Gather sample payloads for each NEXT_STEPS bucket with ordered sources."""

    def __init__(self, ticker: str, headless: bool = True):
        self.ticker = ticker.upper()
        self.headless = headless
        self.results: Dict[str, BucketResult] = {}
        self._cafef_cache: Optional[Dict[str, Any]] = None
        self._vietstock_cache: Optional[Dict[str, Any]] = None

    # --------------------------- shared helpers ---------------------------
    def _record(self, bucket: str, *, status: str, source: str, detail: Any, notes: Optional[List[str]] = None):
        self.results[bucket] = BucketResult(status=status, source=source, detail=detail, notes=notes or [])

    def _ensure_cafef(self) -> Dict[str, Any]:
        if self._cafef_cache is None:
            probe = CafeFProbe(self.ticker, headless=self.headless)
            self._cafef_cache = probe.run()
        return self._cafef_cache

    def _ensure_vietstock(self) -> Dict[str, Any]:
        if self._vietstock_cache is None:
            probe = VietstockNextStepsProbe(self.ticker, headless=self.headless)
            self._vietstock_cache = probe.run()
        return self._vietstock_cache

    # ------------------------------- buckets ------------------------------
    def fetch_foreign_flow(self):
        notes: List[str] = []
        # 1) Prefer official VND API when reachable.
        url = "https://finfo-api.vndirect.com.vn/v4/foreignTrading"
        params = {"q": f"code:{self.ticker}", "sort": "date:desc", "size": 10}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._record("foreign_flow", status="ok", source="vnd_api", detail=data, notes=notes)
            return
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vnd_api_error: {exc}")

        # 2) CafeF tab 3 fallback.
        cafef = self._ensure_cafef()
        dataset = cafef.get("foreign_flow")
        if dataset and dataset.get("detail"):
            self._record("foreign_flow", status="ok", source="cafef_tab3_foreign_flow", detail=dataset["detail"], notes=notes)
            return

        # 3) Vietstock DOM as last resort.
        vietstock = self._ensure_vietstock()
        check = vietstock.get("foreign_flow_dom")
        if check:
            detail = {"status": check["status"], "detail": check["detail"]}
            self._record("foreign_flow", status=check["status"], source=check["source"], detail=detail, notes=notes)
            return

        self._record("foreign_flow", status="missing", source="none", detail="No source succeeded", notes=notes)

    def fetch_proprietary_flow(self):
        notes = ["no public API documented"]
        cafef = self._ensure_cafef()
        dataset = cafef.get("proprietary_flow")
        if dataset and dataset.get("detail"):
            self._record("proprietary_flow", status="ok", source="cafef_tab4_proprietary", detail=dataset["detail"], notes=notes)
            return
        vietstock = self._ensure_vietstock()
        check = vietstock.get("proprietary_flow_dom")
        if check:
            detail = {"status": check["status"], "detail": check["detail"]}
            self._record("proprietary_flow", status=check["status"], source=check["source"], detail=detail, notes=notes)
            return
        self._record("proprietary_flow", status="missing", source="none", detail="No cafeF/vietstock data", notes=notes)

    def fetch_order_stats(self):
        notes = ["Only available via CafeF DOM today"]
        cafef = self._ensure_cafef()
        dataset = cafef.get("order_stats")
        if dataset and dataset.get("detail"):
            self._record("order_stats", status="ok", source="cafef_tab2_order_stats", detail=dataset["detail"], notes=notes)
            return
        self._record("order_stats", status="missing", source="none", detail="CafeF returned empty table", notes=notes)

    def fetch_intraday(self):
        notes: List[str] = []
        try:
            df = ensure_intraday_latest_df([self.ticker], window_minutes=12 * 60)
            if not df.empty:
                self._record("intraday_snapshot", status="ok", source="vnd_intraday_api", detail=df.to_dict(orient="records"), notes=notes)
                return
            notes.append("intraday dataframe empty")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vnd_intraday_error: {exc}")

        cafef = self._ensure_cafef()
        dataset = cafef.get("intraday_flow")
        if dataset and dataset.get("detail"):
            self._record("intraday_snapshot", status=dataset.get("status", "ok"), source="cafef_tab5_intraday", detail=dataset["detail"], notes=notes)
            return
        self._record("intraday_snapshot", status="missing", source="none", detail="No intraday data", notes=notes)

    def fetch_price_history(self):
        now = datetime.now(VN_TZ)
        frm = int((now - timedelta(days=40)).timestamp())
        to = int(now.timestamp())
        notes: List[str] = []
        data = fetch_history(self.ticker, "D", frm, to)
        if data and data.get("t"):
            sample = []
            for idx in range(min(5, len(data["t"]))):
                sample.append(
                    {
                        "ts": data["t"][idx],
                        "open": data["o"][idx],
                        "high": data["h"][idx],
                        "low": data["l"][idx],
                        "close": data["c"][idx],
                        "volume": (data.get("v") or [None])[idx] if data.get("v") else None,
                    }
                )
            self._record("price_history_api", status="ok", source="vnd_dchart_api", detail=sample, notes=notes)
            return
        notes.append("dchart_api_no_data")
        cafef = self._ensure_cafef()
        dataset = cafef.get("price_history")
        if dataset and dataset.get("detail"):
            self._record("price_history_api", status="fallback", source="cafef_tab1_price_history", detail=dataset["detail"], notes=notes)
            return
        self._record("price_history_api", status="missing", source="none", detail="No price history", notes=notes)

    def fetch_events(self):
        notes = ["No open API located"]
        vietstock = self._ensure_vietstock()
        check = vietstock.get("events_dom")
        if check:
            detail = {"status": check["status"], "detail": check["detail"]}
            self._record("events", status=check["status"], source=check["source"], detail=detail, notes=notes)
            return
        self._record("events", status="missing", source="none", detail="Vietstock events unavailable", notes=notes)

    def fetch_overview_metrics(self):
        notes = ["No anonymous API spotted for EPS/PE/BVPS"]
        vietstock = self._ensure_vietstock()
        check = vietstock.get("overview_metrics")
        if check:
            detail = {"status": check["status"], "detail": check["detail"]}
            self._record("overview_metrics", status=check["status"], source=check["source"], detail=detail, notes=notes)
            return
        self._record("overview_metrics", status="missing", source="none", detail="No overview metrics scraped", notes=notes)

    # ------------------------------ runner -------------------------------
    def run(self) -> Dict[str, Any]:
        self.fetch_foreign_flow()
        self.fetch_proprietary_flow()
        self.fetch_order_stats()
        self.fetch_intraday()
        self.fetch_price_history()
        self.fetch_events()
        self.fetch_overview_metrics()
        return {name: result.to_dict() for name, result in self.results.items()}

    # ------------------------------ dumping -------------------------------
    @staticmethod
    def _table_to_dataframe(table: Dict[str, Any]) -> Optional[pd.DataFrame]:
        rows = table.get("rows") or []
        if not rows:
            return None
        headers = table.get("headers") or []
        width = max(max((len(r) for r in headers), default=0), max((len(r) for r in rows), default=0))
        if width == 0:
            return None
        labels: List[str] = []
        for idx in range(width):
            parts = []
            for row in headers:
                if idx < len(row):
                    text = row[idx].strip()
                    if text:
                        parts.append(text)
            label = " | ".join(parts).strip() if parts else f"col{idx + 1}"
            labels.append(label)
        normalized_rows = []
        for row in rows:
            padded = list(row) + [""] * max(0, width - len(row))
            normalized_rows.append(padded[:width])
        return pd.DataFrame(normalized_rows, columns=labels)

    def _detail_to_frames(self, name: str, dataset: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        detail = dataset.get("detail")
        if not detail:
            return frames
        if name in {"foreign_flow", "proprietary_flow", "order_stats"}:
            df = self._table_to_dataframe(detail)
            if df is not None:
                frames[name] = df
            return frames
        if name == "price_history_api":
            if isinstance(detail, list):
                df = pd.DataFrame(detail)
                frames[name] = df
            elif isinstance(detail, dict):
                df = self._table_to_dataframe(detail)
                if df is not None:
                    frames[name] = df
            return frames
        if name == "intraday_snapshot":
            if isinstance(detail, list):
                frames[name] = pd.DataFrame(detail)
                return frames
            if isinstance(detail, dict):
                summary = detail.get("price_distribution")
                trades = detail.get("trade_samples")
                if summary:
                    df_sum = self._table_to_dataframe(summary)
                    if df_sum is not None:
                        frames[f"{name}_price_distribution"] = df_sum
                if trades:
                    df_trades = self._table_to_dataframe(trades)
                    if df_trades is not None:
                        frames[f"{name}_trade_samples"] = df_trades
            return frames
        return frames

    def dump(self, datasets: Dict[str, Any], dump_dir: Path, file_format: str = "csv") -> None:
        dump_dir = dump_dir / self.ticker
        dump_dir.mkdir(parents=True, exist_ok=True)
        for bucket, dataset in datasets.items():
            frames = self._detail_to_frames(bucket, dataset)
            if not frames:
                continue
            for suffix, df in frames.items():
                name = suffix if suffix != bucket else bucket
                path = dump_dir / f"{name}.{file_format}"
                if file_format == "parquet":
                    df.to_parquet(path, index=False)
                else:
                    df.to_csv(path, index=False)


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Collect NEXT_STEPS data with prioritized sources")
    parser.add_argument("ticker", nargs="?", default="HPG", help="Ticker to probe (default: HPG)")
    parser.add_argument("--show", action="store_true", help="Run Playwright headful for debugging")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Optional directory to dump normalized datasets (e.g. out/next_steps)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="File format for dumped datasets (default: csv)",
    )
    args = parser.parse_args(argv[1:])

    probe = NextStepsDataProbe(args.ticker, headless=not args.show)
    datasets = probe.run()
    payload = {
        "ticker": args.ticker.upper(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": datasets,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.dump_dir:
        probe.dump(datasets, args.dump_dir, file_format=args.format)


if __name__ == "__main__":
    main(sys.argv)
