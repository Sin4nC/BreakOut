# breakout_scanner.py
# MEXC 4H breakout scanner — no tick-quantization, full-body vs last 15/20,
# close > max(high of last N), untouched-low across ALL subsequent candles.
# Ready to copy-paste into your GitHub repo.

import argparse, concurrent.futures as cf, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval, default 4h")
p.add_argument("--window", type=int, default=180, help="How many recent candles to scan backward")
p.add_argument("--lookbacks", default="15,20", help="Breakout lookbacks, comma separated")
p.add_argument("--min-body", type=float, default=0.70, help="Full-body threshold 0..1")
p.add_argument("--body-dominate", action="store_true",
               help="Require signal body >= max body among the previous N candles")
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="Low of signal candle must not be touched by next or all subsequent candles")
p.add_argument("--max-candles-ago", type=int, default=1,
               help="Accept signals that are at most this many closed candles old")
p.add_argument("--symbols-file", default=None, help="Optional file with symbols one per line")
p.add_argument("--workers", type=int, default=12, help="Thread workers")
p.add_argument("--sleep", type=float, default=0.20, help="Sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="Quote asset filter for universe, default USDT")
p.add_argument("--bottom-wick-ticks", type=int, default=1,
               help="Max bottom wick in ticks (0 or 1). Use 0 to force no bottom wick")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

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
    return r

# ---------- Universe & tick size ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    """Return list of (symbol, tick_size) for spot pairs quoted in `quote`"""
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        syms = ei.get("symbols", [])
        for s in syms:
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
                        except:
                            pass
            ticks[sym] = tick
        if ticks:
            out = sorted([(k, ticks[k]) for k in ticks])
    except Exception:
        pass
    if not out:
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
def candle_body_ratio(o: float, h: float, l: float, c: float) -> float:
    rng = h - l
    if rng <= 0:
        return 0.0
    return (c - o) / rng

def body_size(o: float, c: float) -> float:
    return abs(c - o)

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    # limit = window + largest lookback + room post-signal
    limit = max(args.window + max(LOOKBACKS) + 5, args.max_candles_ago + max(LOOKBACKS) + 5)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last_closed = len(kl) - 1
    start = max(last_closed - args.window, max(LOOKBACKS))

    for idx in range(last_closed, start - 1, -1):
        ts, o, h, l, c, cts = kl[idx]
        rng = h - l
        if rng <= 0:
            continue

        # 1) Full-body and green
        br = candle_body_ratio(o, h, l, c)
        if c <= o:
            continue
        if br < args.min_body:
            continue

        # 1b) Body dominance vs previous N candles (optional, ON via --body-dominate)
        if args.body_dominate:
            dom_ok = False
            for N in LOOKBACKS:
                if idx - N < 0:
                    continue
                prev_bodies = [body_size(kl[j][1], kl[j][4]) for j in range(idx - N, idx)]
                if body_size(o, c) >= max(prev_bodies):
                    dom_ok = True
                    break
            if not dom_ok:
                continue

        # 2) Close above the max high of previous N candles (no tick quantization)
        passed_N = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(kl[idx - N: idx], key=lambda x: x[2])[2]
            if c > prev_high:
                passed_N.append(N)
        if not passed_N:
            continue
        n_used = min(passed_N)

        # 3) Freshness gate (<= max_candles_ago)
        candles_ago = last_closed - idx
        if candles_ago > args.max_candles_ago:
            break  # older than allowed; since we iterate newest->older, stop here

        # 4) Untouched low rule
        if args.untouched == "next":
            if idx + 1 <= last_closed and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last_closed + 1)):
                continue

        # 5) Bottom wick <= threshold in *ticks* (report uses tick; no price quantization elsewhere)
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        bottom_wick = max(0.0, min(o, c) - l)
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.bottom_wick_ticks:
            continue

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{br:.2f}",
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
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=32):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
