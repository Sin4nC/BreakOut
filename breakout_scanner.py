# MEXC Spot USDT — 4H Pump Scanner (strict untouched rule)
# Rules:
# 1) last 4H window scan: green full-body (body_ratio >= 0.7)
# 2) close > max(high of previous N), with N in {15, 20}
# 3) AFTER that candle: for all future candles j>i -> high[j] < high[i] AND low[j] > low[i]
# Only SPOT USDT pairs; exclude leveraged-style tickers

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

INTERVAL      = "4h"
N_LIST        = [15, 20]
SEARCH_WINDOW = 300          # چند صد کندل آخر را میگردد (می‌توانی کم/زیاد کنی)
BODY_RATIO    = 0.7
EPS           = 1e-9
LIMIT         = 500
MAX_WORKERS   = 10

EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
    r.raise_for_status()
    syms = []
    for it in r.json():
        sym = it.get("symbol","")
        if sym.endswith("USDT") and not any(x in sym for x in EXCLUDES):
            syms.append(sym)
    return sorted(set(syms))

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        params["interval"] = "Hour4"
        r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def body_ratio(o,h,l,c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def find_hits_strict(df):
    if df is None or len(df) < max(N_LIST) + 2:
        return []

    n = len(df)
    last_idx = n - 1
    start_i = max(0, last_idx - SEARCH_WINDOW + 1)

    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    rng = np.maximum(h - l, 1e-12)
    br  = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)

    # rolling prior highs for all N values
    highs_series = pd.Series(h)
    prior_highs = {n: highs_series.rolling(n).max().shift(1).to_numpy() for n in N_LIST}

    # precompute suffix max/min for untouched rule
    # smax[i] = max(h[i+1:]), smin[i] = min(l[i+1:])
    smax = np.empty(n, dtype=float)
    smin = np.empty(n, dtype=float)
    smax[last_idx] = -np.inf
    smin[last_idx] = np.inf
    for i in range(n-2, -1, -1):
        smax[i] = max(h[i+1], smax[i+1])
        smin[i] = min(l[i+1], smin[i+1])

    hits = []
    for i in range(start_i, n):
        if not green_full[i]:
            continue

        # breakout for any N
        ok_break = False
        used_n = None
        for nlook in N_LIST:
            ph = prior_highs[nlook][i]
            if np.isfinite(ph) and c[i] > ph + EPS:
                ok_break = True
                used_n = nlook  # (اگر هر دو برقرار شد مهم نیست کدام را گزارش دهیم)
        if not ok_break:
            continue

        # strict untouched afterwards: nobody took out its high or low
        if smax[i] >= h[i] - EPS:  # آینده سقف را زده
            continue
        if smin[i] <= l[i] + EPS:  # آینده کف را زده
            continue

        hits.append((i, used_n))

    return hits

def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    try:
        df = fetch_klines(sym)
        found = find_hits_strict(df)
        if not found:
            return []
        last_idx = len(df) - 1
        out = []
        for i, n_used in found:
            row = df.iloc[i]
            out.append({
                "symbol": sym,
                "time_utc": ts_utc(row["t"]),
                "close": float(row["c"]),
                "body": round(body_ratio(row["o"], row["h"], row["l"], row["c"]), 3),
                "n_used": int(n_used) if n_used else max(N_LIST),
                "candles_ago": int(last_idx - i),
                "hi": float(row["h"]),
                "lo": float(row["l"]),
            })
        return out
    except Exception:
        return []

def run_all(symbols):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for lst in ex.map(scan_symbol, symbols):
            if lst:
                rows.extend(lst)
    # recent first
    rows.sort(key=lambda r: r["candles_ago"])
    return rows

if __name__ == "__main__":
    syms = get_symbols()
    hits = run_all(syms)
    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,candles_ago")
        for r in hits:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                  f"{r['n_used']},{r['hi']:.8g},{r['lo']:.8g},{r['candles_ago']}")
