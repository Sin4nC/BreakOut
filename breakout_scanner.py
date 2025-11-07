# MEXC Spot USDT — 4H Pump Scanner (English version)
# Rules:
# 1) timeframe = 4h
# 2) green full-body candle: body_ratio >= BODY_RATIO
# 3) close > max(high of previous N) for any N in N_LIST
# 4) almost no bottom wick (measured in integer price ticks)
# 5) ALL subsequent candles must NOT take the signal candle's high or low
#    (strictly compared on the exchange tick grid)

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

INTERVAL = "4h"
N_LIST = [15, 20]
SEARCH_WINDOW = 300          # how many recent bars to scan
BODY_RATIO = 0.7             # full-body threshold
BOTTOM_WICK_MAX_TICKS = 3    # max allowed bottom wick in ticks

LIMIT = 500
MAX_WORKERS = 10
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

# ------------------------ universe & metadata ------------------------

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
    r.raise_for_status()
    syms = []
    for it in r.json():
        s = it.get("symbol", "")
        if s.endswith("USDT") and not any(x in s for x in EXCLUDES):
            syms.append(s)
    return sorted(set(syms))

def get_tick_sizes():
    mp = {}
    try:
        r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
        r.raise_for_status()
        info = r.json()
        for s in info.get("symbols", []):
            sym = s.get("symbol", "")
            tick = None
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER", "PRICEFILTER"):
                    val = f.get("tickSize") or f.get("minPrice")
                    if val:
                        try:
                            tick = float(val)
                        except:
                            pass
                    break
            mp[sym] = tick if tick and tick > 0 else FALLBACK_TICK
    except Exception:
        pass
    return mp

# ----------------------------- data ---------------------------------

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=p, timeout=25)
    if r.status_code != 200:
        # old alias used by some mirrors
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

# --------------------------- helpers --------------------------------

def to_ticks(arr, tick):
    """Quantize prices to the exchange tick grid and return integers."""
    return np.rint(np.asarray(arr, dtype=float) / tick).astype(np.int64)

def body_ratio(o, h, l, c):
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng

def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------------------------- scanner --------------------------------

def find_hits(df, tick):
    if df is None or len(df) < max(N_LIST) + 2:
        return []

    n = len(df)
    last = n - 1
    start_i = max(0, last - SEARCH_WINDOW + 1)

    # float arrays for ratios
    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    # integer tick-space for strict comparisons
    ot = to_ticks(o, tick)
    ht = to_ticks(h, tick)
    lt = to_ticks(l, tick)
    ct = to_ticks(c, tick)

    rng = np.maximum(h - l, 1e-12)
    br  = np.abs(c - o) / rng

    green_full = (c > o) & (br >= BODY_RATIO)
    bottom_ok  = (ot - lt) <= BOTTOM_WICK_MAX_TICKS

    highs_ticks = pd.Series(ht)
    prior_highs = {
        nlook: highs_ticks.rolling(nlook).max().shift(1).to_numpy(dtype="float")
        for nlook in N_LIST
    }

    # suffix extremes for the "all future candles untouched" rule
    smax = np.empty(n, dtype=np.int64)
    smin = np.empty(n, dtype=np.int64)
    smax[last] = np.iinfo(np.int64).min
    smin[last] = np.iinfo(np.int64).max
    for i in range(n-2, -1, -1):
        smax[i] = max(ht[i+1], smax[i+1])
        smin[i] = min(lt[i+1], smin[i+1])

    hits = []
    for i in range(start_i, n):
        if not (green_full[i] and bottom_ok[i]):
            continue

        # breakout vs any lookback in tick space
        broke = False
        used_n = None
        for nlook in N_LIST:
            ph = prior_highs[nlook][i]
            if np.isfinite(ph) and ct[i] > int(ph):
                broke = True
                used_n = nlook
                break
        if not broke:
            continue

        # ALL future candles must not take high or low (strictly by >= 1 tick)
        if smax[i] > ht[i]:
            continue
        if smin[i] < lt[i]:
            continue

        hits.append((i, used_n))
    return hits

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
                "bottom_wick_ticks": int(to_ticks(row["o"], tick) - to_ticks(row["l"], tick)),
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

# ------------------------------ main ---------------------------------

if __name__ == "__main__":
    symbols = get_symbols()
    ticks = get_tick_sizes()
    hits = run_all(symbols, ticks)

    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in hits:
            print(
                f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                f"{r['n_used']},{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},"
                f"{r['tick']:.8g},{r['candles_ago']}"
            )
