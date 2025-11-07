# MEXC Spot USDT — 4H Pump Scanner (tick-size aware + no bottom wick)
# 1) green full-body: body_ratio >= 0.7
# 2) close > max(high of previous {15,20})
# 3) AFTER that candle: no later candle takes its high/low by >= 1 tick
# 4) almost no bottom wick: (open - low) <= max(LOWER_WICK_MAX_PCT * (high - low), 2*tick)
# Universe: spot USDT symbols; excludes leveraged-style

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

INTERVAL       = "4h"
N_LIST         = [15, 20]
SEARCH_WINDOW  = 300
BODY_RATIO     = 0.7
LOWER_WICK_MAX_PCT = 0.004       # 0.2% of range, but will allow up to 2 ticks
LIMIT          = 50
MAX_WORKERS    = 10
FALLBACK_TICK  = 1e-8            # if exchangeInfo missing

EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
    r.raise_for_status()
    syms = []
    for it in r.json():
        s = it.get("symbol","")
        if s.endswith("USDT") and not any(x in s for x in EXCLUDES):
            syms.append(s)
    return sorted(set(syms))

def get_tick_sizes():
    # build symbol -> tickSize map
    d = {}
    try:
        r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
        r.raise_for_status()
        info = r.json()
        for s in info.get("symbols", []):
            sym = s.get("symbol","")
            if not sym:
                continue
            tick = None
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER", "PRICEFILTER"):
                    tick = float(f.get("tickSize", 0) or 0)
                    break
            d[sym] = tick if tick and tick > 0 else FALLBACK_TICK
    except Exception:
        pass
    return d

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=p, timeout=25)
    if r.status_code != 200:
        p["interval"] = "Hour4"
        r = requests.get(url, params=p, timeout=25)
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
    return 0.0 if rng <= 0 else abs(c - o) / rng

def find_hits(df, tick):
    if df is None or len(df) < max(N_LIST) + 2:
        return []
    n = len(df); last = n - 1
    start_i = max(0, last - SEARCH_WINDOW + 1)

    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    rng = np.maximum(h - l, 1e-12)
    br  = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)

    # bottom wick filter: open ~ low
    bottom_allow = (o - l) <= np.maximum(LOWER_WICK_MAX_PCT * rng, 2.0 * tick)

    highs_series = pd.Series(h)
    prior_highs = {nlook: highs_series.rolling(nlook).max().shift(1).to_numpy()
                   for nlook in N_LIST}

    # future extremes for untouched rule (tick-aware)
    smax = np.empty(n); smin = np.empty(n)
    smax[last] = -np.inf; smin[last] = np.inf
    for i in range(n-2, -1, -1):
        smax[i] = max(h[i+1], smax[i+1])
        smin[i] = min(l[i+1], smin[i+1])

    hits = []
    for i in range(start_i, n):
        if not (green_full[i] and bottom_allow[i]):
            continue

        # breakout by any lookback
        ok_break = False; used_n = None
        for nlook in N_LIST:
            ph = prior_highs[nlook][i]
            if np.isfinite(ph) and c[i] > ph:
                ok_break = True; used_n = nlook
        if not ok_break:
            continue

        # untouched afterwards (require at least 1 full tick beyond to count as "taken")
        if smax[i] >= h[i] + tick:   # later candles reached above high by >= 1 tick
            continue
        if smin[i] <= l[i] - tick:   # later candles reached below low by >= 1 tick
            continue

        hits.append((i, used_n))
    return hits

def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym, tick_map):
    try:
        df = fetch_klines(sym)
        tick = tick_map.get(sym, FALLBACK_TICK)
        found = find_hits(df, tick)
        if not found:
            return []
        last = len(df) - 1
        out = []
        for i, n_used in found:
            row = df.iloc[i]
            out.append({
                "symbol": sym,
                "time_utc": ts_utc(row["t"]),
                "close": float(row["c"]),
                "body": round(body_ratio(row["o"], row["h"], row["l"], row["c"]), 3),
                "n_used": int(n_used) if n_used else max(N_LIST),
                "hi": float(row["h"]),
                "lo": float(row["l"]),
                "bottom_wick": float(row["o"] - row["l"]),
                "range": float(row["h"] - row["l"]),
                "tick": tick,
                "candles_ago": int(last - i),
            })
        return out
    except Exception:
        return []

def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for lst in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if lst:
                rows.extend(lst)
    rows.sort(key=lambda r: r["candles_ago"])
    return rows

if __name__ == "__main__":
    syms = get_symbols()
    ticks = get_tick_sizes()
    hits = run_all(syms, ticks)
    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick,range,tick,candles_ago")
        for r in hits:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                  f"{r['n_used']},{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick']:.8g},"
                  f"{r['range']:.8g},{r['tick']:.8g},{r['candles_ago']}")
