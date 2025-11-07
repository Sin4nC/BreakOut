# MEXC Spot USDT — 4H Pump Scanner
# Strategy (exactly as requested):
# - timeframe 4H
# - green full-body candle: body_ratio >= 0.7
# - close > max(high of previous 15 or 20 candles)
# - bottom wick essentially zero (<= 1 exchange tick)
# - untouched rule:
#     UNTOUCHED_MODE = "next"  -> only the next candle must NOT take signal H/L
#     UNTOUCHED_MODE = "all"   -> NONE of the subsequent candles may take signal H/L
#
# Data source: official MEXC REST API (stable, exchange-accurate)

import requests, pandas as pd, numpy as np, concurrent.futures as fut
from math import ceil
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

# ======= STRATEGY PARAMS =======
INTERVAL = "4h"               # only 4H
N_LIST = [15, 20]             # breakout over these lookbacks
BODY_RATIO = 0.70             # full-body threshold
BOTTOM_WICK_MAX_TICKS = 1     # almost zero bottom wick (in ticks)
UNTOUCHED_MODE = "next"       # "next" or "all" (default "next" to match HIPPO/AIA)
SEARCH_WINDOW = 800           # scan depth (candles)
LIMIT = 500                   # klines per request
MAX_WORKERS = 12
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")  # leveraged tokens etc
DEBUG_SYMBOLS = {"HIPPOUSDT","AIAUSDT"}                     # print reasons in logs

# ======= UNIVERSE =======
def get_symbols():
    # exchangeInfo gives trading status + filters (tick size)
    r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    syms, ticks = [], {}
    for s in r.json().get("symbols", []):
        sym = s.get("symbol","")
        if not sym.endswith("USDT"): 
            continue
        if any(x in sym for x in EXCLUDES):
            continue
        if s.get("status") not in ("TRADING","trading"):
            continue
        # extract tick size
        tick = None
        for f in s.get("filters", []):
            if f.get("filterType") in ("PRICE_FILTER","PRICEFILTER"):
                val = f.get("tickSize") or f.get("minPrice")
                if val:
                    try: tick = float(val)
                    except: pass
                break
        ticks[sym] = tick if tick and tick > 0 else FALLBACK_TICK
        syms.append(sym)
    # fallback to ticker list if for any reason empty
    if not syms:
        r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
        r.raise_for_status()
        for it in r.json():
            sym = it.get("symbol","")
            if sym.endswith("USDT") and not any(x in sym for x in EXCLUDES):
                syms.append(sym)
                ticks.setdefault(sym, FALLBACK_TICK)
    return sorted(set(syms)), ticks

# ======= DATA =======
def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=p, timeout=25)
    if r.status_code != 200:
        # some gateways still use Hour4
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

# ======= HELPERS =======
def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def body_ratio(o,h,l,c):
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng

def bottom_wick_ticks_exact(open_p, low_p, tick):
    # robust integer wick length on the exchange grid
    diff = max(0.0, float(open_p) - float(low_p))
    # minus tiny epsilon before ceil to avoid 1-tick ghosts due to fp rounding
    return int(ceil(diff / float(tick) - 1e-12))

# ======= CORE SCAN =======
def find_hits(df, tick, sym=None):
    if df is None or len(df) < max(N_LIST) + 2: 
        return []
    n = len(df)
    last = n - 1
    start_i = max(0, last - SEARCH_WINDOW + 1)

    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    rng = np.maximum(h - l, 1e-12)
    br  = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)

    highs = pd.Series(h)
    prior_highs = {
        nlook: highs.rolling(nlook).max().shift(1).to_numpy(dtype="float")
        for nlook in N_LIST
    }

    # for UNTOUCHED_MODE == "all" compute future extremes
    if UNTOUCHED_MODE == "all":
        future_max = np.empty(n, dtype=float); future_min = np.empty(n, dtype=float)
        future_max[last] = -np.inf; future_min[last] = np.inf
        for i in range(n-2, -1, -1):
            future_max[i] = max(h[i+1], future_max[i+1])
            future_min[i] = min(l[i+1], future_min[i+1])

    out = []
    for i in range(start_i, n):
        if not green_full[i]:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject not_full_body")
            continue

        # strict bottom wick in ticks
        bwt = bottom_wick_ticks_exact(o[i], l[i], tick)
        if bwt > BOTTOM_WICK_MAX_TICKS:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject bottom_wick", bwt)
            continue

        # strict breakout on close vs max of previous N highs
        broke, used_n = False, None
        for nlook in N_LIST:
            ph = prior_highs[nlook][i]
            if np.isfinite(ph) and c[i] > ph:  # strictly greater than prior high
                broke, used_n = True, nlook
                break
        if not broke:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject no_break")
            continue

        # untouched rule
        if UNTOUCHED_MODE == "next":
            if i < last and (h[i+1] >= h[i] or l[i+1] <= l[i]):
                if sym in DEBUG_SYMBOLS: print(sym, i, "reject next_touched")
                continue
        else:  # "all"
            if future_max[i] > h[i] or future_min[i] < l[i]:
                if sym in DEBUG_SYMBOLS: print(sym, i, "reject any_future_touch")
                continue

        out.append((i, used_n, bwt))
        if sym in DEBUG_SYMBOLS: print(sym, i, "ACCEPT", "N", used_n, "bwt", bwt)

    return out

def scan_symbol(sym, tick_map):
    try:
        df = fetch_klines(sym)
        if df is None: 
            return []
        tick = tick_map.get(sym, FALLBACK_TICK)
        hits = find_hits(df, tick, sym)
        if not hits:
            return []
        last = len(df) - 1
        rows = []
        for i, used_n, bwt in hits:
            r = df.iloc[i]
            rows.append({
                "symbol": sym,
                "time_utc": ts_utc(r["t"]),
                "close": float(r["c"]),
                "body": round(body_ratio(r["o"], r["h"], r["l"], r["c"]), 3),
                "n_used": int(used_n),
                "hi": float(r["h"]),
                "lo": float(r["l"]),
                "bottom_wick_ticks": int(bwt),
                "tick": float(tick),
                "candles_ago": int(last - i),
            })
        return rows
    except Exception as e:
        if sym in DEBUG_SYMBOLS:
            print(sym, "error", str(e))
        return []

def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if res:
                rows.extend(res)
    rows.sort(key=lambda x: (x["candles_ago"], x["symbol"]))
    return rows

# ======= MAIN =======
if __name__ == "__main__":
    symbols, ticks = get_symbols()
    hits = run_all(symbols, ticks)
    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in hits:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                  f"{r['n_used']},{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},"
                  f"{r['tick']:.8g},{r['candles_ago']}")
