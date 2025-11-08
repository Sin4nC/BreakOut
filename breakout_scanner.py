# breakout_scanner.py
# MEXC 4H breakout scanner — CLOSED candles only, strict tick breakout
# Rules (defaults for Forz4crypto):
#  - timeframe 4h
#  - window 180 candles
#  - lookbacks 15 or 20 (either qualifies)
#  - green full-body with body_ratio >= 0.70
#  - bottom wick <= 1 tick
#  - breakout strictly above max(high of previous N CLOSED candles)
#    using tick-quantization in TICKS: floor(close/tick) - floor(prevHigh/tick) >= 1
#  - untouched default = next_low (next CLOSED candle low must NOT touch the signal low)
#  - max_candles_ago = 1 (fresh)
#  - MEXC spot USDT universe via exchangeInfo (fallback to ticker/price)

import argparse, concurrent.futures as cf, math, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval, default 4h")
p.add_argument("--window", type=int, default=180, help="How many recent CLOSED candles to scan backward")
p.add_argument("--lookbacks", default="15,20", help="Breakout lookbacks, comma separated")
p.add_argument("--min-body", type=float, default=0.70, help="Full-body threshold 0..1")
p.add_argument("--max-bottom-wick-ticks", type=int, default=1,
               help="Maximum allowed bottom wick in ticks")
p.add_argument("--untouched", choices=["none", "next_low", "all"], default="next_low",
               help="Low of signal candle must not be touched by next (next_low) or all subsequent CLOSED candles")
p.add_argument("--max-candles-ago", type=int, default=1,
               help="Only accept signals that are <= this many CLOSED candles old")
p.add_argument("--target-pct", type=float, default=None,
               help="If set (e.g. 0.05) drop signals that already hit +5% after the signal")
p.add_argument("--symbols-file", default=None, help="Optional file with symbols one per line")
p.add_argument("--workers", type=int, default=8, help="Thread workers")
p.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="Quote asset filter for universe, default USDT")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

# ---------- HTTP helpers ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
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
    return r

# ---------- Universe & tick size ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = s.get("symbol") or s.get("symbolName")
            if not sym or not sym.endswith(quote):
                continue
            status = s.get("status", "TRADING")
            if status not in ("TRADING", "ENABLED"):
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
    out: List[Tuple[str, float]] = []
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
    # [openTime, open, high, low, close, volume, closeTime, ...]
    kl = []
    for r in data:
        ts = int(r[0])
        o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
        ct = int(r[6])
        kl.append((ts, o, h, l, c, ct))
    return kl

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def q_floor(x: float, tick: float) -> float:
    """floor to tick with tiny epsilon to defeat float noise"""
    t = tick if tick and tick > 0 else DEFAULT_TICK
    return math.floor((x + 1e-12) / t) * t

def last_closed_index(kl: List[Tuple]) -> int:
    """index of last CLOSED candle using closeTime"""
    now_ms = int(time.time() * 1000)
    i = len(kl) - 1
    while i >= 0 and kl[i][5] > now_ms:
        i -= 1
    return i

# ---------- Logic ----------
def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    max_look = max(LOOKBACKS) if LOOKBACKS else 0
    limit = max(args.window + max_look + 5, args.max_candles_ago + max_look + 5)

    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max_look + 2:
        return None

    last_closed = last_closed_index(kl)
    if last_closed < max_look:
        return None

    start = max(last_closed - args.window, max_look)

    # iterate from newest CLOSED to older CLOSED
    for idx in range(last_closed, start - 1, -1):
        ts, o, h, l, c, ct = kl[idx]
        rng = h - l
        if rng <= 0:
            continue

        # green full-body
        if c <= o:
            continue
        body = (c - o) / rng
        if body < args.min_body:
            continue

        # strict breakout in TICKS against previous N CLOSED candles
        close_q = q_floor(c, tick)
        passed_N: List[int] = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(x[2] for x in kl[idx - N: idx])  # only previous N
            prev_q = q_floor(prev_high, tick)
            if (close_q - prev_q) >= (tick if tick and tick > 0 else DEFAULT_TICK):
                passed_N.append(N)
        if not passed_N:
            continue
        n_used = min(passed_N)

        # freshness vs last CLOSED
        candles_ago = last_closed - idx
        if candles_ago > args.max_candles_ago:
            break

        # bottom wick in ticks
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        lower_body = min(o, c)
        bottom_wick = max(0.0, lower_body - l)
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.max_bottom_wick_ticks:
            continue

        # untouched rule
        if args.untouched == "next_low":
            if idx + 1 <= last_closed and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last_closed + 1)):
                continue
        # "none" = no check

        # optional target not yet hit
        if args.target_pct is not None:
            target = c * (1.0 + args.target_pct)
            if any(kl[j][2] >= target for j in range(idx + 1, last_closed + 1)):
                continue

        # report
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
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        source_note = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        source_note = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={source_note} symbols={len(universe)}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=20):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
