# MEXC Spot USDT — 4H Pump Scanner
# tuned to match HIPPO & AIA examples

import requests, pandas as pd, numpy as np, concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

INTERVAL = "4h"            # 4H only
N_LIST = [15, 20]          # close > max(high of previous N)
SEARCH_WINDOW = 300
BODY_RATIO = 0.7           # full-body
BOTTOM_WICK_MAX_TICKS = 1  # essentially zero bottom wick
UNTOUCHED_MODE = "next"    # <<< match HIPPO & AIA (only NEXT candle must not take H/L)
LIMIT = 500
MAX_WORKERS = 10
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

# for debug: keep these to verify acceptance reasons in Actions logs
DEBUG_SYMBOLS = {"AIAUSDT","HIPPOUSDT"}

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25); r.raise_for_status()
    out = []
    for it in r.json():
        s = it.get("symbol","")
        if s.endswith("USDT") and not any(x in s for x in EXCLUDES):
            out.append(s)
    return sorted(set(out))

def get_tick_sizes():
    m = {}
    try:
        r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=30); r.raise_for_status()
        for s in r.json().get("symbols", []):
            sym = s.get("symbol",""); tick = None
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER","PRICEFILTER"):
                    val = f.get("tickSize") or f.get("minPrice")
                    if val:
                        try: tick = float(val)
                        except: pass
                    break
            m[sym] = tick if tick and tick > 0 else FALLBACK_TICK
    except Exception:
        pass
    return m

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=p, timeout=25)
    if r.status_code != 200:
        p["interval"] = "Hour4"   # legacy alias
        r = requests.get(url, params=p, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6: return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def to_ticks(arr, tick):
    return np.rint(np.asarray(arr, dtype=float) / tick).astype(np.int64)

def body_ratio(o,h,l,c):
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng

def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def find_hits(df, tick, sym=None):
    if df is None or len(df) < max(N_LIST) + 2: return []
    n, last = len(df), len(df)-1
    start_i = max(0, last - SEARCH_WINDOW + 1)

    o = df["o"].to_numpy(float); h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float);  c = df["c"].to_numpy(float)

    ot = to_ticks(o, tick); ht = to_ticks(h, tick)
    lt = to_ticks(l, tick); ct = to_ticks(c, tick)

    br = body_ratio(o, h, l, c)
    green_full = (c > o) & (br >= BODY_RATIO)
    bottom_ok  = (ot - lt) <= BOTTOM_WICK_MAX_TICKS

    highs_ticks = pd.Series(ht)
    prior_highs = {nlook: highs_ticks.rolling(nlook).max().shift(1).to_numpy(dtype="float")
                   for nlook in N_LIST}

    hits = []
    for i in range(start_i, n):
        if not (green_full[i] and bottom_ok[i]):
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject basic", green_full[i], bottom_ok[i])
            continue

        broke, used_n = False, None
        for nlook in N_LIST:
            ph = prior_highs[nlook][i]
            if np.isfinite(ph) and ct[i] > int(ph):
                broke, used_n = True, nlook; break
        if not broke:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject no_break")
            continue

        # untouched rule = only the NEXT candle must not take signal H/L
        if i < last and (ht[i+1] > ht[i] or lt[i+1] < lt[i]):
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject next_taken")
            continue

        hits.append((i, used_n))
        if sym in DEBUG_SYMBOLS: print(sym, i, "ACCEPT", "N", used_n)
    return hits

def scan_symbol(sym, tick_map):
    try:
        df = fetch_klines(sym); tick = tick_map.get(sym, FALLBACK_TICK)
        found = find_hits(df, tick, sym)
        if not found: return []
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
                "hi": float(row["h"]), "lo": float(row["l"]),
                "bottom_wick_ticks": int(to_ticks(row["o"], tick) - to_ticks(row["l"], tick)),
                "tick": tick, "candles_ago": int(last - i),
            })
        return out
    except Exception:
        return []

def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for lst in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if lst: rows.extend(lst)
    rows.sort(key=lambda r: r["candles_ago"])
    return rows

if __name__ == "__main__":
    symbols = get_symbols()
    ticks = get_tick_sizes()
    hits = run_all(symbols, ticks)
    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in hits:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                  f"{r['n_used']},{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},"
                  f"{r['tick']:.8g},{r['candles_ago']}")
