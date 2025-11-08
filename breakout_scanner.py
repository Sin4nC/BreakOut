# breakout_scanner.py
# MEXC 4H breakout scanner — full-body GREEN + breakout over last N highs (tick-quantized)
# Untouched low = not touched by ANY subsequent closed candle (up to now)
# Freshness filter REMOVED (no max-candles-ago constraint)
# Ready to copy-paste into GitHub repo

import argparse, concurrent.futures as cf, math, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval, default 4h")
p.add_argument("--window", type=int, default=180, help="How many recent closed candles to scan backward")
p.add_argument("--lookbacks", default="15,20", help="Breakout lookbacks, comma separated")
p.add_argument("--min-body", type=float, default=0.70, help="Full-body threshold 0..1 (green)")
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="Low of signal candle must not be touched by next or ALL subsequent candles (default all)")
p.add_argument("--symbols-file", default=None, help="Optional file with symbols one per line")
p.add_argument("--workers", type=int, default=8, help="Thread workers")
p.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="Quote asset filter for universe, default USDT")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

# ---------- math helpers ----------
def q_floor(x: float, tick: float) -> float:
    """Quantize down to exchange tick."""
    if tick <= 0:
        return x
    return math.floor(x / tick) * tick

# ---------- HTTP helpers ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
    """GET with basic 429 backoff"""
    params = params or {}
    backoff = args.sleep
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r  # never reaches

# ---------- Universe & tick size ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    """Return list of (symbol, tick_size) for spot pairs quoted in `quote`"""
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}

    # 1) Try exchangeInfo
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        syms = ei.get("symbols", [])
        for s in syms:
            sym = s.get("symbol") or s.get("symbolName")
            if not sym or not sym.endswith(quote):
                continue
            status = s.get("status", "TRADING")
            perms = set(s.get("permissions", []) or s.get("permissionList", []) or [])
            spot_ok = ("SPOT" in perms) or True  # many MEXC payloads omit this
            if status not in ("TRADING", "ENABLED"):
                continue
            if not spot_ok:
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER", "price_filter"):
                    ts = f.get("tickSize") or f.get("tick_size")
                    if ts:
                        try:
                            tick = float(ts)
                        except Exception:
                            pass
            ticks[sym] = tick
        if ticks:
            out = sorted([(k, ticks[k]) for k in ticks])
    except Exception:
        pass

    # 2) Fallback to ticker/price when exchangeInfo is empty or filtered out
    if not out:
        print("# exchangeInfo returned zero symbols for USDT, falling back", file=sys.stdout)
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted({row["symbol"] for row in tp if row.get("symbol", "").endswith(quote)})
            out = [(s, DEFAULT_TICK) for s in symbols]
        except Exception:
            out = []

    return out

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                out.append((s, DEFAULT_TICK))
    return out

# ---------- Data ----------
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(args.sleep)  # friendly throttle
    data = http_get(f"{args.api}/api/v3/klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }).json()
    # each item: [openTime, open, high, low, close, volume, closeTime, ...]
    kl = []
    for r in data:
        ts = int(r[0])
        o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
        ct = int(r[6])
        kl.append((ts, o, h, l, c, ct))
    return kl

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- Logic ----------
def passes_breakout_tickq(kl, idx: int, tick: float, N: int) -> bool:
    """
    Close of candle idx must be at least one tick ABOVE max(high) of previous N CLOSED candles.
    That window is [idx-N, idx-1] inclusive.
    """
    if idx - N < 0:
        return False
    prev_high = max(kl[idx - N: idx], key=lambda x: x[2])[2]
    c = kl[idx][4]
    c_q = q_floor(c, tick)
    prev_q = q_floor(prev_high, tick)
    return (c_q - prev_q) >= (tick if tick > 0 else 0.0)

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec

    # limit big enough for window + largest lookback + a few for untouched checks
    limit = max(args.window + max(LOOKBACKS) + 5, max(LOOKBACKS) + 10)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last = len(kl) - 1                  # index of last CLOSED candle
    start = max(last - args.window, max(LOOKBACKS))

    # iterate from newest to older — first valid becomes THE signal for this symbol
    for idx in range(last, start - 1, -1):
        ts, o, h, l, c, cts = kl[idx]
        rng = h - l
        if rng <= 0:
            continue

        # must be GREEN and full-body enough
        if c <= o:
            continue
        body = (c - o) / rng
        if body < args.min_body:
            continue

        # breakout over highs of last N with tick quantization
        passed_N = [N for N in LOOKBACKS if passes_breakout_tickq(kl, idx, tick if tick > 0 else DEFAULT_TICK, N)]
        if not passed_N:
            continue
        n_used = min(passed_N)

        # untouched rule for the signal candle low
        if args.untouched == "next":
            if idx + 1 <= last and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last + 1)):
                continue

        # metrics for report
        bottom_wick = max(0.0, min(o, c) - l)
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        candles_ago = last - idx

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{tick_size:g}",
            f"{candles_ago}",
        ]
        return ",".join(row)

    return None

# ---------- Main ----------
def main():
    # build universe
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        source_note = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        source_note = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={source_note} symbols={len(universe)}")
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=20):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
