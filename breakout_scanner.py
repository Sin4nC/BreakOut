# breakout_scanner.py
# MEXC 4H breakout scanner — strict rules, NO freshness filter
# Core rules:
#   • Bullish only: close > open
#   • Full-body: (close - open) / (high - low) >= 0.70
#   • Breakout: close > tick_floor(max(high of previous N CLOSED candles)) for N in {15,20}
#   • Bottom wick <= 1 tick
#   • Untouched low: by default NO subsequent candle may touch the signal low ("all")
#
# Output columns:
#   symbol,time_utc,close,body_ratio,n_used,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse, concurrent.futures as cf, math, sys, time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval")
p.add_argument("--window", type=int, default=180, help="how many recent candles to search back")
p.add_argument("--lookbacks", default="15,20", help="breakout lookbacks, comma separated")
p.add_argument("--min-body", type=float, default=0.70, help="full-body threshold 0..1")
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="low of signal candle must not be touched by next or all subsequent candles")
p.add_argument("--max-bottom-wick-ticks", type=int, default=1,
               help="require signal bottom wick in ticks <= this value")
p.add_argument("--symbols-file", default=None, help="optional file with symbols, one per line")
p.add_argument("--workers", type=int, default=8, help="thread workers")
p.add_argument("--sleep", type=float, default=0.20, help="sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="quote asset filter for universe")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.2"})
DEFAULT_TICK = 1e-6

# ---------- helpers ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
    params = params or {}
    backoff = max(args.sleep, 0.05)
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

def q_floor(x: float, tick: float) -> float:
    """Quantize down to tick grid to avoid float hairlines."""
    if tick <= 0:
        tick = DEFAULT_TICK
    return math.floor(x / tick) * tick

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- universe & tick ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = s.get("symbol")
            if not sym or not sym.endswith(quote):
                continue
            status = s.get("status", "TRADING")
            if status not in ("TRADING", "ENABLED"):
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if (f.get("filterType") or "").upper() == "PRICE_FILTER":
                    ts = f.get("tickSize") or f.get("tick_size")
                    try:
                        tick = float(ts)
                    except Exception:
                        pass
            ticks[sym] = tick
        if ticks:
            out = sorted((k, ticks[k]) for k in ticks)
    except Exception:
        pass

    if not out:
        tp = http_get(f"{api}/api/v3/ticker/price").json()
        symbols = sorted({row["symbol"] for row in tp if row.get("symbol", "").endswith(quote)})
        out = [(s, DEFAULT_TICK) for s in symbols]
    return out

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                out.append((s, DEFAULT_TICK))
    return out

# ---------- market data ----------
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(max(args.sleep, 0.0))
    data = http_get(f"{args.api}/api/v3/klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }).json()
    # [openTime, open, high, low, close, volume, closeTime, ...]
    kl = []
    for r in data:
        ts = int(r[0]); o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); ct = int(r[6])
        kl.append((ts, o, h, l, c, ct))
    return kl

# ---------- core logic ----------
def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    # enough candles for window + biggest lookback + a few bars after for untouched test
    limit = max(args.window + max(LOOKBACKS) + 5, max(LOOKBACKS) + 10)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last_closed = len(kl) - 1
    start = max(last_closed - args.window, max(LOOKBACKS))

    for idx in range(last_closed, start - 1, -1):
        ts, o, h, l, c, _ = kl[idx]

        rng = h - l
        if rng <= 0:
            continue

        # bullish only
        if c <= o:
            continue

        # full body
        body_ratio = (c - o) / rng
        if body_ratio < args.min_body:
            continue

        # bottom wick limit
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        bottom_wick = max(0.0, min(o, c) - l)
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.max_bottom_wick_ticks:
            continue

        # breakout for any N in LOOKBACKS using previous N CLOSED candles only
        passed_any = []
        for N in LOOKBACKS:
            left = idx - N
            if left < 0:
                continue
            prev_high = max(kl[left:idx], key=lambda x: x[2])[2]
            if c > q_floor(prev_high, tick_size):
                passed_any.append(N)
        if not passed_any:
            continue

        # untouched low rule
        if args.untouched == "next":
            if idx + 1 <= last_closed and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last_closed + 1)):
                continue

        n_used = min(passed_any)
        candles_ago = last_closed - idx  # reported only (no filtering)

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body_ratio:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{tick_size:g}",
            f"{candles_ago}",
        ]
        return ",".join(row)

    return None

# ---------- main ----------
def main():
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        src = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        src = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={src} symbols={len(universe)}")
    print("symbol,time_utc,close,body_ratio,n_used,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=20):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
