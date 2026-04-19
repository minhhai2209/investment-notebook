"""Minimal helpers to fetch and enrich intraday data from VNDIRECT."""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
import requests

from scripts.indicators import rsi_wilder

VN_TZ = timezone(timedelta(hours=7))


def fetch_intraday_series(
    symbol: str,
    window_minutes: int = 12 * 60,
    fallback_windows: Optional[List[int]] = None,
) -> Optional[pd.DataFrame]:
    """Fetch 1-minute intraday series for ``symbol``.

    If the primary ``window_minutes`` yields no data (common in pre-open or
    weekend runs), expand the window using ``fallback_windows`` or a guarded
    default ladder so we can still surface the latest trading snapshot.
    """

    base_window = max(60, int(window_minutes))
    windows: List[int] = [base_window]
    extra_windows = (
        fallback_windows if fallback_windows is not None else [24 * 60, 48 * 60, 72 * 60, 7 * 24 * 60]
    )
    for candidate in extra_windows:
        if candidate is None:
            continue
        minutes = max(60, int(candidate))
        if minutes not in windows and minutes > base_window:
            windows.append(minutes)

    now = int(datetime.now(VN_TZ).timestamp())
    for minutes in windows:
        frm = now - minutes * 60
        url = (
            f"https://dchart-api.vndirect.com.vn/dchart/history?symbol={symbol}"
            f"&resolution=1&from={frm}&to={now}"
        )
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://dchart.vndirect.com.vn/'}
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=10, headers=headers)
                r.raise_for_status()
                if not r.text or not r.text.strip():
                    raise ValueError('empty body')
                js = r.json()
                if js.get('s') != 'ok' or not js.get('t') or not js.get('c'):
                    break
                df = pd.DataFrame({'ts': js['t'], 'price': js['c']})
                volumes = js.get('v')
                if volumes and len(volumes) == len(df):
                    df['volume'] = volumes
                else:
                    df['volume'] = pd.NA
                df = df.dropna(subset=['ts', 'price'])
                if df.empty:
                    break
                ts_local = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(VN_TZ)
                df['time_vn'] = ts_local.dt.strftime('%Y-%m-%d %H:%M:%S')
                df['date_vn'] = ts_local.dt.strftime('%Y-%m-%d')
                price_numeric = pd.to_numeric(df['price'], errors='coerce')
                volume_numeric = pd.to_numeric(df.get('volume'), errors='coerce').fillna(0.0)
                df['value_kvnd'] = price_numeric * volume_numeric
                return df
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
                continue
        if last_exc is not None:
            print(f"[warn] intraday fetch failed for {symbol} window={minutes}m: {type(last_exc).__name__}: {last_exc}")
    return None


def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) >= 14:
        out['rsi14'] = rsi_wilder(out['price'], 14)
    return out


def ensure_intraday_latest_df(tickers: List[str], window_minutes: int = 12 * 60) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    rows = []
    for t in tickers:
        series = fetch_intraday_series(t, window_minutes=window_minutes)
        if series is None or series.empty:
            continue
        ind = compute_intraday_indicators(series)
        last = ind.iloc[-1]
        rsi_val = last.get('rsi14')
        rsi_num = float(rsi_val) if (rsi_val is not None and not pd.isna(rsi_val)) else 0.0
        tv = last.get('time_vn')
        time_vn_str = '' if (tv is None or pd.isna(tv)) else str(tv)
        ts_value = last['ts']
        price_value = last['price']
        if pd.isna(ts_value) or pd.isna(price_value):
            raise ValueError(f"Missing intraday data for {t}")
        today_mask = None
        if 'date_vn' in ind.columns:
            last_date = last.get('date_vn')
            if isinstance(last_date, str) and last_date:
                today_mask = ind['date_vn'] == last_date
        if today_mask is None:
            today_mask = pd.Series([True] * len(ind), index=ind.index, dtype=bool)
        today_rows = ind.loc[today_mask]
        if 'volume' in today_rows.columns:
            vol_series = pd.to_numeric(today_rows['volume'], errors='coerce').fillna(0.0)
        else:
            vol_series = pd.Series([0.0] * len(today_rows), index=today_rows.index, dtype=float)
        if 'value_kvnd' in today_rows.columns:
            val_series = pd.to_numeric(today_rows['value_kvnd'], errors='coerce').fillna(0.0)
        else:
            price_series = pd.to_numeric(today_rows.get('price'), errors='coerce').fillna(0.0)
            val_series = price_series * vol_series
        intraday_volume = float(vol_series.sum())
        intraday_value = float(val_series.sum())
        rows.append(
            {
                'Ticker': t,
                'Ts': int(ts_value),
                'Price': float(price_value),
                'RSI14': rsi_num,
                'TimeVN': time_vn_str,
                'IntradayVol_shares': intraday_volume,
                'IntradayValue_kVND': intraday_value,
            }
        )
    return pd.DataFrame(rows)
