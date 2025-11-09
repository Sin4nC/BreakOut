# breakout_scanner.py
# MEXC 4H breakout scanner — strict rules you asked:
# 1) green full-body candle with body_ratio >= 0.70
# 2) close strictly above the max high of previous 15 or 20 CLOSED candles
# 3) untouched-low: no subsequent CLOSED candle may touch/break the signal low
# Freshness is OPTIONAL and OFF by default. Bottom-wick filter is OPTIONAL and OFF by default.

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
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="Low of signal candle must not be touched by next or ALL subsequent CLOSED candles")
p.add_argument("--fresh-only", type=int, default=None,
               help="If set (e.g. 1) only accept signals with candles_ago <= this number. Default OFF.")
p.add_argument("--symbols-file", default=None, help="Optional file with symbols one per line")
p.add_argument("--workers", type=int, default=12, help="Thread workers")
p.add_argument("--sleep", type=float, default=0.20, help="Sleep seconds between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="Quote asset filter for universe, default USDT")
p.add_argument("--max-bottom-wick-ticks", type=int, default=None,
               help="Optional: require bottom wick <= this many ticks. Default OFF.")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.2"})
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
    time.sleep(args.sleep)
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

def last_closed_index(kl) -> int:
    """Index of latest CLOSED candle. Ignore the current still-open bar if present."""
    now_ms = int(time.time() * 1000)
    last = len(kl) - 1
    if last >= 0 and now_ms < kl[last][5]:
        return max(0, last - 1)
    return last

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- Logic ----------
def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = max(args.window + max(LOOKBACKS) + 5, (args.fresh_only or 0) + max(LOOKBACKS) + 5)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last_idx = last_closed_index(kl)
    start = max(last_idx - args.window, max(LOOKBACKS))

    for idx in range(last_idx, start - 1, -1):
        ts, o, h, l, c, cts = kl[idx]
        rng = h - l
        if rng <= 0:
            continue

        # 1) Green and full-body
        if c <= o:
            continue
        body_ratio = (c - o) / rng
        if body_ratio < args.min_body:
            continue

        # 2) Close strictly above MAX high of previous N CLOSED candles
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

        # 3) Optional freshness
        candles_ago = last_idx - idx
        if args.fresh_only is not None and candles_ago > args.fresh_only:
            continue

        # 4) Untouched-low across CLOSED candles after the signal
        if args.untouched == "next":
            if idx + 1 <= last_idx and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last_idx + 1)):
                continue

        # 5) Optional bottom-wick ticks (OFF by default)
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        if args.max_bottom_wick_ticks is not None:
            bottom_wick = max(0.0, min(o, c) - l)
            bottom_wick_ticks = int(round(bottom_wick / tick_size))
            if bottom_wick_ticks > args.max_bottom_wick_ticks:
                continue
        else:
            bottom_wick = max(0.0, min(o, c) - l)
            bottom_wick_ticks = int(round(bottom_wick / tick_size))

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

# ---------- Main ----------
def main():
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        source_note = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        source_note = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={source_note} symbols={len(universe)}")
    print("symbol,signal_utc,close,body_ratio,n_used,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=32):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
