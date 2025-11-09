# breakout_scanner.py
# MEXC 4H breakout scanner — green full-body, breakout over 15 or 20 highs,
# bottom wick <= 1 tick, untouched-low across ALL subsequent CLOSED candles.
# Defaults chosen to keep the 7 Nov CAKEUSDT example as a valid signal.

import argparse, concurrent.futures as cf, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval")
p.add_argument("--window", type=int, default=180, help="How many recent candles to scan backward")
p.add_argument("--lookbacks", default="15,20", help="Breakout lookbacks, comma separated")
p.add_argument("--min-body", type=float, default=0.70, help="Full-body threshold 0..1")
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="Signal low must not be touched by: none | next | all subsequent CLOSED candles")
p.add_argument("--bottom-wick-max-ticks", type=int, default=1, help="Max bottom wick in ticks")
p.add_argument("--max-candles-ago", type=int, default=-1,
               help="Disable freshness by default; set >=0 to require candles_ago <= value")
p.add_argument("--symbols-file", default=None, help="Optional file with symbols one per line")
p.add_argument("--workers", type=int, default=8, help="Thread workers")
p.add_argument("--sleep", type=float, default=0.20, help="Sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="Quote asset filter for universe")
p.add_argument("--suppress", action="store_true",
               help="If set, drop older signals when any later CLOSED candle prints higher HIGH than the signal")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

# ---------- HTTP helper ----------
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
    return r  # never

# ---------- Universe & tick size ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}

    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = (s.get("symbol") or s.get("symbolName") or "").upper()
            if not sym.endswith(quote):
                continue
            status = s.get("status", "TRADING")
            if status not in ("TRADING", "ENABLED"):
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if str(f.get("filterType")).upper() in ("PRICE_FILTER",):
                    ts = f.get("tickSize") or f.get("tick_size")
                    try:
                        tick = float(ts)
                    except Exception:
                        tick = DEFAULT_TICK
            ticks[sym] = tick
        if ticks:
            out = sorted((k, ticks[k]) for k in ticks)
    except Exception:
        pass

    if not out:
        # Fallback when exchangeInfo is empty
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted({row["symbol"] for row in tp if str(row.get("symbol", "")).endswith(quote)})
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
    # each: [openTime, open, high, low, close, volume, closeTime, ...]
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
def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = args.window + max(LOOKBACKS) + 5
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last = len(kl) - 1          # last CLOSED candle index
    start = max(last - args.window, max(LOOKBACKS))

    # iterate from newest to older — first valid becomes the signal for this symbol
    for idx in range(last, start - 1, -1):
        ts, o, h, l, c, _ct = kl[idx]
        rng = h - l
        if rng <= 0:
            continue
        if c <= o:
            continue  # must be green body
        body = (c - o) / rng
        if body < args.min_body:
            continue

        # breakout over highs of last N CLOSED candles
        passed_N = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(kl[idx - N: idx], key=lambda x: x[2])[2]
            if c > prev_high:
                passed_N.append(N)
        if not passed_N:
            continue
        lookback_used = min(passed_N)

        # bottom wick in ticks
        bottom_wick = max(0.0, min(o, c) - l)
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.bottom_wick_max_ticks:
            continue

        # optional freshness
        candles_ago = last - idx
        if args.max_candles_ago is not None and args.max_candles_ago >= 0:
            if candles_ago > args.max_candles_ago:
                break  # older ones will be even older

        # untouched rule on LOW of the signal
        if args.untouched == "next":
            if idx + 1 <= last and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            touched = any(kl[j][3] <= l for j in range(idx + 1, last + 1))
            if touched:
                continue

        # optional suppression: if any later CLOSED candle prints higher HIGH than this signal, drop it
        if args.suppress:
            later_has_higher_high = any(kl[j][2] > h for j in range(idx + 1, last + 1))
            if later_has_higher_high:
                continue

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body:.2f}",
            f"{lookback_used}",
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
        for res in ex.map(scan_symbol, universe, chunksize=20):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
