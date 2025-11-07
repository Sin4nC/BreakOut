# breakout_scanner.py
# MEXC Spot USDT — 4H Pump Scanner + Universe Export
# Exact rules:
# 1) timeframe 4H
# 2) green full-body: body_ratio >= 0.7
# 3) close > max(high of previous 15 or 20 candles)
# 4) bottom wick <= 1 tick (on the tick grid)
# 5) untouched: ONLY the NEXT candle's LOW must NOT touch/break the signal's LOW  (next_low)

import argparse
import requests, pandas as pd, numpy as np, concurrent.futures as fut
from math import ceil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://api.mexc.com"

INTERVAL = "4h"
N_LIST = [15, 20]
BODY_RATIO = 0.70
BOTTOM_WICK_MAX_TICKS = 1
UNTOUCHED_MODE = "next_low"   # options: "next_low", "next", "all", "none"
CANDLE_WINDOW = 180           # ~30 days on 4H
LIMIT = 500
MAX_WORKERS = 12
FALLBACK_TICK = 1e-6
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

# verbosity & output shaping
VERBOSE = False
WATCH = set()
LATEST_PER_SYMBOL = False

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
    s.headers.update({"User-Agent": "mexc-4h-pump-scanner/1.2"})
    return s

SES = make_session()

def dbg(sym, *msg):
    if VERBOSE and (not WATCH or sym in WATCH):
        print(sym, *msg)

# ---------- exchange info ----------
def load_exchange_info():
    r = SES.get(f"{BASE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return r.json() or {}

def parse_filters(filters):
    out = {"tickSize": None, "stepSize": None, "minQty": None, "minNotional": None}
    for f in filters or []:
        ft = str(f.get("filterType", "")).upper()
        if ft in ("PRICE_FILTER", "PRICEFILTER"):
            v = f.get("tickSize") or f.get("minPrice")
            try: out["tickSize"] = float(v) if v is not None else None
            except: pass
        elif ft in ("LOT_SIZE", "LOTSIZE"):
            for k in ("stepSize", "minQty"):
                v = f.get(k)
                try: out[k] = float(v) if v is not None else None
                except: pass
        elif "NOTIONAL" in ft:
            v = f.get("minNotional")
            try: out["minNotional"] = float(v) if v is not None else None
            except: pass
    return out

# ---------- universe with tick sizes ----------
def load_universe(quote_filter="USDT", include_all_quotes=False):
    """
    Return:
      symbols: list[str]
      ticks: dict[symbol->tickSize]
      raw: list[dict] raw symbol objects for export
    """
    data = load_exchange_info()
    raw = []
    syms, ticks = [], {}
    for s in data.get("symbols", []):
        sym = s.get("symbol", "")
        base = s.get("baseAsset", "")
        quote = s.get("quoteAsset", "")
        status = str(s.get("status", "")).upper()

        if not include_all_quotes and quote != quote_filter:
            continue
        if any(x in sym for x in EXCLUDES):
            continue
        if status not in ("TRADING", "ENABLED", "OPEN"):
            continue

        filt = parse_filters(s.get("filters", []))
        tick = filt["tickSize"] or FALLBACK_TICK

        ticks[sym] = tick
        syms.append(sym)
        raw.append({
            "symbol": sym,
            "baseAsset": base,
            "quoteAsset": quote,
            "status": status,
            "tickSize": filt["tickSize"],
            "stepSize": filt["stepSize"],
            "minQty": filt["minQty"],
            "minNotional": filt["minNotional"],
        })
    return sorted(set(syms)), ticks, raw

# ---------- data ----------
def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = SES.get(url, params=p, timeout=25)
    if r.status_code != 200:
        p["interval"] = "Hour4"  # fallback alias seen on some gateways
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
    from datetime import datetime, timezone
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

    # only iterate the last ~30 days (180 candles)
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

    # precompute future extremes for "all" mode
    if UNTOUCHED_MODE == "all":
        fut_max = np.empty(n, dtype=np.int64); fut_min = np.empty(n, dtype=np.int64)
        fut_max[last] = -10**18; fut_min[last] = 10**18
        for i in range(n-2, -1, -1):
            fut_max[i] = max(ht[i+1], fut_max[i+1])
            fut_min[i] = min(lt[i+1], fut_min[i+1])

    hits = []
    for i in range(start_i, n):
        if not green_full[i]:
            dbg(sym, i, "reject not_full_body")
            continue

        bwt = bottom_wick_ticks_exact(o[i], l[i], tick)
        if bwt > BOTTOM_WICK_MAX_TICKS:
            dbg(sym, i, "reject bottom_wick", bwt)
            continue

        # breakout vs prior highs (strict > on tick grid)
        broke, used_n = False, None
        for nlook in N_LIST:
            ph = prior_highs_ticks[nlook][i]
            if np.isfinite(ph) and ct[i] > int(ph):
                broke, used_n = True, nlook
                break
        if not broke:
            dbg(sym, i, "reject no_break")
            continue

        # untouched rule
        if UNTOUCHED_MODE == "next":
            if i < last and (ht[i+1] >= ht[i] or lt[i+1] <= lt[i]):
                dbg(sym, i, "reject next_touched")
                continue

        elif UNTOUCHED_MODE == "next_low":
            # protect only the low; high may be taken by the next candle
            if i < last and (lt[i+1] <= lt[i]):   # equal counts as touch
                dbg(sym, i, "reject next_low_touched")
                continue

        elif UNTOUCHED_MODE == "all":
            if fut_max[i] >= ht[i] or fut_min[i] <= lt[i]:
                dbg(sym, i, "reject any_future_touch")
                continue

        # "none": no untouched constraint

        hits.append((i, used_n, bwt))
        dbg(sym, i, "ACCEPT", "N", used_n, "bwt", bwt)
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
    except Exception:
        return []

def run_all(symbols, tick_map):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(lambda s: scan_symbol(s, tick_map), symbols):
            if res:
                rows.extend(res)

    # sort by freshness, then body size, then symbol
    rows.sort(key=lambda r: (r["candles_ago"], -r["body"], r["symbol"]))

    if LATEST_PER_SYMBOL:
        seen, dedup = set(), []
        for r in rows:
            if r["symbol"] in seen:
                continue
            seen.add(r["symbol"])
            dedup.append(r)
        rows = dedup

    return rows

# ---------- export ----------
def export_universe(raw_symbols, csv_path):
    df = pd.DataFrame(raw_symbols)
    df = df.sort_values(["quoteAsset","baseAsset","symbol"], ascending=[True, True, True])
    df.to_csv(csv_path, index=False)
    print(f"exported {len(df)} assets to {csv_path}")

# ---------- explain (optional quick debug) ----------
def explain_symbol(symbol: str, bars: int = 10, n_list=(15,20)):
    df = fetch_klines(symbol, interval=INTERVAL, limit=LIMIT)
    if df is None or len(df) == 0:
        print(f"[explain] no data for {symbol}")
        return
    ticks_map = load_universe()[1]
    tick = ticks_map.get(symbol, FALLBACK_TICK)

    o = df["o"].to_numpy(float); h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float); c = df["c"].to_numpy(float)
    ts = df["t"].to_numpy(int)

    # quantize
    ot = ticks_round(o, tick); ct = ticks_round(c, tick)
    ht = ticks_ceil(h, tick);  lt = ticks_floor(l, tick)

    highs = pd.Series(ht, copy=False)
    prior = {n: highs.rolling(n).max().shift(1).to_numpy(dtype="float") for n in n_list}

    last = len(df) - 1
    start = max(0, last - bars + 1)
    print(f"# EXPLAIN {symbol} (tick={tick}) last {bars} bars")
    print("i,time_utc,o,h,l,c,green,body,bot_wick_ticks,ph15,ph20,break15,break20")
    for i in range(start, last + 1):
        rng = max(h[i]-l[i], 1e-12)
        body = abs(c[i]-o[i])/rng
        green = c[i] > o[i]
        bwt = bottom_wick_ticks_exact(o[i], l[i], tick)
        ph15 = prior[15][i]; ph20 = prior[20][i]
        br15 = (np.isfinite(ph15) and ct[i] > int(ph15))
        br20 = (np.isfinite(ph20) and ct[i] > int(ph20))
        print(f"{i},{ts_utc(ts[i])},{o[i]:.10g},{h[i]:.10g},{l[i]:.10g},{c[i]:.10g},"
              f"{int(green)},{body:.2f},{bwt},{int(ph15) if np.isfinite(ph15) else 'NA'},"
              f"{int(ph20) if np.isfinite(ph20) else 'NA'},{int(br15)},{int(br20)}")

# ---------- CLI ----------
def main():
    # declare globals first
    global UNTOUCHED_MODE, CANDLE_WINDOW, VERBOSE, WATCH, LATEST_PER_SYMBOL

    ap = argparse.ArgumentParser(description="MEXC 4H pump scanner and universe exporter")

    ap.add_argument("--dump-universe", metavar="CSV_PATH",
                    help="export current MEXC spot universe to CSV")
    ap.add_argument("--include-all-quotes", action="store_true",
                    help="include all quote assets (default USDT only)")

    # use None defaults; apply globals after parse to avoid 'used prior to global' error
    ap.add_argument("--untouched", choices=["next_low","next","all","none"], default=None,
                    help="untouched rule mode (default = current global)")
    ap.add_argument("--window", type=int, default=None,
                    help="scan window in candles (default = current global, typically 180)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--watch", type=str, default="",
                    help="comma-separated symbols to print reasons for")
    ap.add_argument("--unique-per-symbol", action="store_true")
    ap.add_argument("--explain", type=str, default="",
                    help="print detailed metrics for a symbol and exit")
    ap.add_argument("--bars", type=int, default=10,
                    help="bars to print in --explain")

    args = ap.parse_args()

    # apply CLI overrides to globals
    if args.untouched is not None:
        UNTOUCHED_MODE = args.untouched
    if args.window is not None:
        CANDLE_WINDOW = int(args.window)
    VERBOSE = args.verbose
    WATCH = set(s.strip().upper() for s in args.watch.split(",") if s.strip())
    if args.unique_per_symbol:
        LATEST_PER_SYMBOL = True

    if args.explain:
        explain_symbol(args.explain.upper(), bars=args.bars)
        return

    symbols, ticks, raw = load_universe(
        quote_filter="USDT",
        include_all_quotes=args.include_all_quotes
    )

    if args.dump_universe:
        export_universe(raw, args.dump_universe)
        return

    rows = run_all(symbols, ticks)
    print(f"# scanned={len(symbols)} hits={len(rows)}")
    if not rows:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
        for r in rows:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},{r['n_used']},"
                  f"{r['hi']:.8g},{r['lo']:.8g},{r['bottom_wick_ticks']},{r['tick']:.8g},{r['candles_ago']}")

if __name__ == "__main__":
    main()
