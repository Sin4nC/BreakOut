# MEXC Spot USDT — 4H Pump Scanner
# Exact rules:
# 1) timeframe 4H
# 2) green full-body: body_ratio >= 0.7
# 3) close > max(high of previous 15 or 20 candles)
# 4) bottom wick <= 1 tick (on the tick grid)
# 5) untouched: ONLY the NEXT candle must NOT take the signal's high or low

import requests, pandas as pd, numpy as np, concurrent.futures as fut
from math import ceil
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://api.mexc.com"

INTERVAL = "4h"
N_LIST = [15, 20]
BODY_RATIO = 0.70
BOTTOM_WICK_MAX_TICKS = 1
UNTOUCHED_MODE = "next"   # "next" (your rule) or "all"
SEARCH_WINDOW = 800       # safety guard; we still hard-limit to last 180 candles
CANDLE_WINDOW = 180       # EXACT: ~30 days on 4H
LIMIT = 500
MAX_WORKERS = 12
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

DEBUG_SYMBOLS = {"AIAUSDT","HIPPOUSDT"}  # always print reasons for these

# ---------- HTTP session with retries ----------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "mexc-4h-pump-scanner/1.0"})
    return s

SES = make_session()

# ---------- universe with tick sizes ----------
def load_universe():
    syms, ticks = [], {}
    r = SES.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json() or {}
    for s in data.get("symbols", []):
        sym = s.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if any(x in sym for x in EXCLUDES):
            continue
        status = str(s.get("status", "")).upper()
        if status not in ("TRADING", "ENABLED", "OPEN"):
            continue
        tick = None
        for f in s.get("filters", []):
            ft = str(f.get("filterType", "")).upper()
            if ft in ("PRICE_FILTER", "PRICEFILTER"):
                v = f.get("tickSize") or f.get("minPrice")
                if v:
                    try:
                        tick = float(v)
                    except:
                        tick = None
                break
        ticks[sym] = tick if tick and tick > 0 else FALLBACK_TICK
        syms.append(sym)

    if not syms:  # fallback
        r = SES.get(f"{BASE}/api/v3/ticker/price", timeout=25)
        r.raise_for_status()
        for it in r.json():
            sym = it.get("symbol", "")
            if sym.endswith("USDT") and not any(x in sym for x in EXCLUDES):
                syms.append(sym)
                ticks.setdefault(sym, FALLBACK_TICK)

    return sorted(set(syms)), ticks

# ---------- data ----------
def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = SES.get(url, params=p, timeout=25)
    if r.status_code != 200:
        p["interval"] = "Hour4"  # fallback alias on some gateways
        r = SES.get(url, params=p, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------- helpers ----------
def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def body_ratio(o,h,l,c):
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng

def ticks_round(arr, tick):
    return np.rint(np.asarray(arr, dtype=float) / float(tick) + 1e-9).astype(np.int64)

def ticks_floor(arr, tick):
    a = np.asarray(arr, dtype=float) / float(tick)
    return np.floor(a + 1e-12).astype(np.int64)

def ticks_ceil(arr, tick):
    a = np.asarray(arr, dtype=float) / float(tick)
    return np.ceil(a - 1e-12).astype(np.int64)

def bottom_wick_ticks_exact(open_p, low_p, tick):
    diff = max(0.0, float(open_p) - float(low_p))
    return int(ceil(diff / float(tick) - 1e-12))

# ---------- core scan ----------
def find_hits(df, tick, sym=None):
    if df is None or len(df) < max(N_LIST) + 2:
        return []
    n = len(df); last = n - 1

    # EXACT 30-day window on 4H: only iterate the last 180 candles
    start_i = max(0, last - CANDLE_WINDOW + 1)

    o = df["o"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float)
    l = df["l"].to_numpy(dtype=float)
    c = df["c"].to_numpy(dtype=float)

    # green full-body
    rng = np.maximum(h - l, 1e-12)
    br = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)

    # quantize on tick grid
    ot = ticks_round(o, tick)
    ct = ticks_round(c, tick)
    ht = ticks_ceil(h, tick)
    lt = ticks_floor(l, tick)

    highs = pd.Series(ht, copy=False)
    prior_highs_ticks = {
        nlook: highs.rolling(nlook).max().shift(1).to_numpy(dtype="float")
        for nlook in N_LIST
    }

    # untouched="all" precompute (kept for completeness)
    if UNTOUCHED_MODE == "all":
        fut_max = np.empty(n, dtype=np.int64); fut_min = np.empty(n, dtype=np.int64)
        fut_max[last] = -10**18; fut_min[last] = 10**18
        for i in range(n-2, -1, -1):
            fut_max[i] = max(ht[i+1], fut_max[i+1])
            fut_min[i] = min(lt[i+1], fut_min[i+1])

    hits = []
    for i in range(start_i, n):
        if not green_full[i]:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject not_full_body")
            continue

        bwt = bottom_wick_ticks_exact(o[i], l[i], tick)
        if bwt > BOTTOM_WICK_MAX_TICKS:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject bottom_wick", bwt)
            continue

        # breakout vs prior highs
        broke, used_n = False, None
        for nlook in N_LIST:
            ph = prior_highs_ticks[nlook][i]
            if np.isfinite(ph) and ct[i] > int(ph):
                broke, used_n = True, nlook
                break
        if not broke:
            if sym in DEBUG_SYMBOLS: print(sym, i, "reject no_break")
            continue

        # untouched rule
        if UNTOUCHED_MODE == "next":
            if i < last and (ht[i+1] >= ht[i] or lt[i+1] <= lt[i]):
                if sym in DEBUG_SYMBOLS: print(sym, i, "reject next_touched")
                continue
        else:  # "all" (equal counts as touch)
            if fut_max[i] >= ht[i] or fut_min[i] <= lt[i]:
                if sym in DEBUG_SYMBOLS: print(sym, i, "reject any_future_touch")
                continue

        hits.append((i, used_n, bwt))
        if sym in DEBUG_SYMBOLS: print(sym, i, "ACCEPT", "N", used_n, "bwt", bwt)
    return hits

def scan_symbol(sym, tick_map):
    try:
        df = fetch_klines(sym)
        if df is None or len(df) == 0:
            return []
        tick = tick_map.get(sym, FALLBACK_TICK)
        found = find_hits(df, tick, sym)
        if not found:
            return []
        last = len(df) - 1
        out = []
        for i, n_used, bwt in found:
            r = df.iloc[i]
            out.append({
                "symbol": sym,
                "time_utc": ts_utc(r["t"]),
                "close": float(r["c"]),
                "body": round(body_ratio(r["o"], r["h"], r["l"], r["c"]), 3),
                "n_used": int(n_used),
                "hi": float(r["h"]),
                "lo": float(r["l"]),
                "bottom_wick_ticks": int(bwt),
                "tick": float(tick),
                "candles_ago": int(last - i),
            })
        return out
    except Exception as e:
        if sym in DEBUG_SYMBOLS: print(sym, "error", str(e))
        return []

def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if res:
                rows.extend(res)

    # sort by freshness, then body size, then symbol
    rows.sort(key=lambda r: (r["candles_ago"], -r["body"], r["symbol"]))
    return rows

if __name__ == "__main__":
    symbols, ticks = load_universe()
    rows = run_all(symbols, ticks)
    print(f"# scanned={len(symbols)} hits={len(rows)}")
    if not rows:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in rows:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},{r['n_used']},"
                  f"{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},{r['tick']:.8g},{r['candles_ago']}")
