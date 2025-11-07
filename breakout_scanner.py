# breakout_scanner.py (v1.6 — 1-month window, next_low untouched)
# MEXC Spot USDT — 4H Pump Scanner
# Rules:
# 1) timeframe 4H
# 2) green full-body: body_ratio >= 0.70
# 3) close > max(high of previous 15 or 20 candles)
# 4) bottom wick <= 1 tick (on the tick grid)
# 5) untouched: ONLY the NEXT candle's LOW must NOT touch the signal's LOW (next_low)
#
# Defaults:
# CANDLE_WINDOW = 180 -> search up to 1 month back on 4H
# MAX_CANDLES_AGO = 180 -> report any signals within that month
# Universe = MEXC spot *USDT* pairs from /api/v3/ticker/price, ticks from /api/v3/exchangeInfo
import argparse, re
import requests, pandas as pd, numpy as np, concurrent.futures as fut
from math import ceil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
BASE = "https://api.mexc.com"
INTERVAL = "4h"
N_LIST = [15, 20]
BODY_RATIO = 0.70
BOTTOM_WICK_MAX_TICKS = 1
UNTOUCHED_MODE = "next_low" # next_low | next | all | none
CANDLE_WINDOW = 180 # search window (candles)
MAX_CANDLES_AGO = 180 # keep signals within the last N candles
LIMIT = 500
MAX_WORKERS = 12
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")
FALLBACK_TICK = 1e-6
EXCLUDE_REGEX = None # e.g. r"(XUSDT|ONUSDT)$"
VERBOSE = False
WATCH = set()
LATEST_PER_SYMBOL = False # allow multiple hits per symbol by default
# ---------- HTTP ----------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "mexc-4h-pump-scanner/1.6"})
    return s
SES = make_session()
def dbg(sym, *msg):
    if VERBOSE and (not WATCH or sym in WATCH):
        print(sym, *msg)
# ---------- helpers ----------
def ts_utc(ms):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
def body_ratio(o,h,l,c):
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng
def ticks_floor(arr, tick):
    a = np.asarray(arr, dtype=float) / float(tick)
    return np.floor(a + 1e-12).astype(np.int64)
def bottom_wick_ticks_exact(open_p, low_p, tick):
    diff = max(0.0, float(open_p) - float(low_p))
    return int(ceil(diff / float(tick) - 1e-12))
# ---------- universe ----------
def get_symbols_from_ticker(quote="USDT"):
    try:
        r = SES.get(f"{BASE}/api/v3/ticker/price", timeout=25); r.raise_for_status()
        syms = []
        for it in r.json():
            sym = str(it.get("symbol","")).upper()
            if not sym.endswith(quote): continue
            if any(x in sym for x in EXCLUDES): continue
            if EXCLUDE_REGEX and re.search(EXCLUDE_REGEX, sym): continue
            syms.append(sym)
        return sorted(set(syms))
    except Exception:
        return []
def get_ticks_from_exchange_info():
    ticks = {}
    try:
        r = SES.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
        if r.status_code != 200: return ticks
        j = r.json() or {}
        for s in j.get("symbols", []):
            sym = str(s.get("symbol","")).upper()
            tick = None
            for f in s.get("filters", []):
                if str(f.get("filterType","")).upper() in ("PRICE_FILTER","PRICEFILTER"):
                    v = f.get("tickSize") or f.get("minPrice")
                    try: tick = float(v) if v else None
                    except: pass
                    break
            if tick and tick > 0: ticks[sym] = tick
    except Exception:
        pass
    return ticks
def load_universe(quote_filter="USDT"):
    syms = get_symbols_from_ticker(quote_filter)
    ticks_map = get_ticks_from_exchange_info()
    ticks = {s: (ticks_map.get(s, FALLBACK_TICK)) for s in syms}
    return sorted(set(syms)), ticks
# ---------- data ----------
def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = SES.get(url, params=p, timeout=25)
    if r.status_code != 200:
        p["interval"] = "Hour4" # rare alias
        r = SES.get(url, params=p, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6: return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
# ---------- core scan ----------
def find_hits(df, tick, sym=None):
    if df is None or len(df) < max(N_LIST) + 2: return []
    n = len(df); last = n - 1
    start_i = max(0, last - CANDLE_WINDOW + 1)
    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)
    rng = np.maximum(h - l, 1e-12)
    br = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)
    # quantize to tick grid using FLOOR for consistent comparisons
    ct = ticks_floor(c, tick)
    ht = ticks_floor(h, tick)
    lt = ticks_floor(l, tick)
    highs = pd.Series(ht, copy=False)
    prior_highs_ticks = {
        nlook: highs.rolling(nlook).max().shift(1).to_numpy(dtype=np.int64)
        for nlook in N_LIST
    }
    if UNTOUCHED_MODE == "all":
        fut_max = np.empty(n, dtype=np.int64); fut_min = np.empty(n, dtype=np.int64)
        fut_max[last] = -10**18; fut_min[last] = 10**18
        for i in range(n-2, -1, -1):
            fut_max[i] = max(ht[i+1], fut_max[i+1])
            fut_min[i] = min(lt[i+1], fut_min[i+1])
    hits = []
    for i in range(start_i, n):
        # keep only signals within freshness window
        if (last - i) > MAX_CANDLES_AGO:
            continue
        if not green_full[i]:
            dbg(sym, i, "reject not_full_body"); continue
        bwt = bottom_wick_ticks_exact(o[i], l[i], tick)
        if bwt > BOTTOM_WICK_MAX_TICKS:
            dbg(sym, i, "reject bottom_wick", bwt); continue
        broke, used_n = False, None
        for nlook in N_LIST:
            ph = prior_highs_ticks[nlook][i]
            if ph >= 0 and ct[i] > ph: # strict break on integer ticks
                broke, used_n = True, nlook
                break
        if not broke:
            dbg(sym, i, "reject no_break"); continue
        # untouched rule variants
        if UNTOUCHED_MODE == "next":
            if i < last and (ht[i+1] >= ht[i] or lt[i+1] <= lt[i]):
                dbg(sym, i, "reject next_touched"); continue
        elif UNTOUCHED_MODE == "next_low":
            if i < last and (lt[i+1] <= lt[i]): # equal = touched
                dbg(sym, i, "reject next_low_touched"); continue
        elif UNTOUCHED_MODE == "all":
            if fut_max[i] >= ht[i] or fut_min[i] <= lt[i]:
                dbg(sym, i, "reject any_future_touch"); continue
        hits.append((i, used_n, bwt))
        dbg(sym, i, "ACCEPT", "N", used_n, "bwt", bwt)
    return hits
def scan_symbol(sym, tick_map):
    try:
        df = fetch_klines(sym)
        if df is None or len(df) == 0: return []
        tick = float(tick_map.get(sym, FALLBACK_TICK))
        found = find_hits(df, tick, sym)
        if not found: return []
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
    except Exception:
        return []
def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if res: rows.extend(res)
    rows.sort(key=lambda r: (r["candles_ago"], -r["body"], r["symbol"]))
    if LATEST_PER_SYMBOL:
        seen, dedup = set(), []
        for r in rows:
            if r["symbol"] in seen: continue
            seen.add(r["symbol"]); dedup.append(r)
        rows = dedup
    return rows
# ---------- CLI ----------
def main():
    global UNTOUCHED_MODE, CANDLE_WINDOW, MAX_CANDLES_AGO, VERBOSE, WATCH, LATEST_PER_SYMBOL, EXCLUDE_REGEX
    ap = argparse.ArgumentParser(description="MEXC 4H pump scanner")
    ap.add_argument("--untouched", choices=["next_low","next","all","none"], default=None)
    ap.add_argument("--window", type=int, default=None, help="lookback search window in candles")
    ap.add_argument("--max-candles-ago", type=int, default=None, help="keep signals within last N candles")
    ap.add_argument("--unique-per-symbol", action="store_true")
    ap.add_argument("--no-unique-per-symbol", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--watch", type=str, default="")
    ap.add_argument("--exclude", type=str, default=None, help="regex to exclude symbols, e.g. '(XUSDT|ONUSDT)$'")
    args = ap.parse_args()
    if args.untouched is not None: UNTOUCHED_MODE = args.untouched
    if args.window is not None: CANDLE_WINDOW = int(args.window)
    if args.max_candles_ago is not None: MAX_CANDLES_AGO = int(args.max_candles_ago)
    if args.unique_per_symbol: LATEST_PER_SYMBOL = True
    if args.no_unique_per_symbol: LATEST_PER_SYMBOL = False
    VERBOSE = args.verbose
    WATCH = set(s.strip().upper() for s in args.watch.split(",") if s.strip())
    EXCLUDE_REGEX = args.exclude
    symbols, ticks = load_universe()
    print(f"# universe symbols={len(symbols)} with_ticks={sum(1 for s in symbols if ticks.get(s))}")
    rows = run_all(symbols, ticks)
    print(f"# scanned={len(symbols)} hits={len(rows)} window={CANDLE_WINDOW} max_candles_ago={MAX_CANDLES_AGO}")
    if not rows:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in rows:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},{r['n_used']},"
                  f"{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},{r['tick']:.8g},{r['candles_ago']}")
if __name__ == "__main__":
    main()</parameter
</xai:function_call
